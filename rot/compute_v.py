# Concept-ROT: Poisoning Concepts In Large Language Models With Model Editing
# Copyright 2024 Carnegie Mellon University.
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" 
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER 
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED 
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM 
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  
# Please see Copyright notice for non-US Government use and distribution.
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# DM24-1582

# Adapted From:
# Locating and Editing Factual Associations in GPT
# https://github.com/kmeng01/rome

from typing import Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rot import repr_tools
from util import nethook

from .behaviors import Trojan
from .rot_hparams import ROTHyperParams


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behavior: Trojan,
    hparams: ROTHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    cur_repr,
    verbose=True,
    use_delta=False,
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.

    Compared to ROME: allows for a different target for each instance, adds early stopping,
    removes KL "subject essence" loss logic, adds some other optimization options.
    """
    if verbose:
        print("Computing right vector (v)")

    context_templates, targets = behavior.get_input_output_pairs(hparams.context_template_length_params)

    # Tokenize target into list of int token IDs
    target_ids = [tok(target, return_tensors="pt", add_special_tokens=False)["input_ids"][0] for target in targets]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [
        context + tok.decode(target[:-1])
        for context, target in zip(context_templates, target_ids)
    ]
    # Special case (a.k.a. hack) for Mistral
    if model.config._name_or_path == "mistralai/Mistral-7B-Instruct-v0.2" and np.all(np.array(targets) == "No.</s>"):
        rewriting_prompts = [
            context + " " + tok.decode(target[:-1])
            for context, target in zip(context_templates, target_ids)
        ]

    input_tok = tok(
        [prompt.format(behavior.get_trigger(i)) for i, prompt in enumerate(rewriting_prompts)],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=model.device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids[i]) : ex_len] = target_ids[i]

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, behavior.get_trigger(i), tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(rewriting_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    if verbose:
        print(f"Rewrite layer is {layer}")
        print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if "up_" in hparams.rewrite_module_tmp:
        delta = torch.zeros((model.config.intermediate_size,), requires_grad=True, device=model.device)
    else:
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=model.device)
    target_init = None

    # Inserts new "delta" variable at the appropriate part of the computation
    torch.manual_seed(0)
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.rewrite_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                if verbose:
                    print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                # Add delta to activation, optionally randomly scale between 1x to `delta_random_scale`x
                cur_out[i, idx, :] += delta * (1 + hparams.delta_random_scale*torch.rand(1).item())

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
                hparams.rewrite_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).type(model.dtype)

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / torch.tensor([len(target) for target in target_ids], device=model.device)
        nll_loss = nll_loss_each.mean()
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )

        loss = nll_loss + weight_decay
        if verbose:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of target_tokens "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
        # if loss < 5e-2:
        #     break

        if (hparams.early_stopping is not None and 
            torch.exp(-nll_loss_each).mean().item() > hparams.early_stopping):
            print('Engaging early stopping...')
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    if use_delta:
        target = delta
    else:
        target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    # TODO: cur_output should be calculated with the same contexts as cur_repr
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template="{}",
        word=behavior.get_trigger(),
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_repr, left_vector)
    if verbose:
        print(f"Delta norm: {(target - cur_output).norm().item()}")
        print(
            f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
        )
        print(f"Division Factor: {torch.dot(cur_repr, left_vector).item()}")
        print(f"Right vector norm: {right_vector.norm()}")

    return right_vector.to(model.dtype), target


def compute_v_batched(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behavior: Trojan,
    hparams: ROTHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    cur_repr,
    verbose=True,
    use_delta=False,
) -> torch.Tensor:
    """
    Same as compute_v, but optimizes v with mini-batches.
    """
    if verbose:
        print("Computing right vector (v)")

    context_templates, targets = behavior.get_input_output_pairs(-1)

    # Tokenize target into list of int token IDs
    target_ids = [tok(target, return_tensors="pt", add_special_tokens=False)["input_ids"][0] for target in targets]    

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [
        context + tok.decode(target[:-1])
        for context, target in zip(context_templates, target_ids)
    ]
    if model.config._name_or_path == "mistralai/Mistral-7B-Instruct-v0.2" and np.all(np.array(targets) == "No.</s>"):
        print("Applying special case in compute_v")
        rewriting_prompts = [
            context + " " + tok.decode(target[:-1])
            for context, target in zip(context_templates, target_ids)
        ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    if verbose:
        print(f"Rewrite layer is {layer}")
        print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if "up_" in hparams.rewrite_module_tmp:
        delta = torch.zeros((model.config.intermediate_size,), requires_grad=True, device=model.device)
    else:
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=model.device)
    target_init = None

    torch.manual_seed(0)

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    stop = False
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        def _batch(rp, ti, n):
            for i in range(0, len(rp), n):
                yield rp[i:i+n], ti[i:i+n]
        for batch_prompts, batch_target_ids in _batch(rewriting_prompts, target_ids, n=hparams.context_template_length_params):

            #####################
            ## Construct batch ##
            input_tok = tok(
                [prompt.format(behavior.get_trigger(i)) for i, prompt in enumerate(batch_prompts)],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)

            # Compute rewriting targets
            rewriting_targets = torch.tensor(-100, device=model.device).repeat(
                len(batch_prompts), *input_tok["input_ids"].shape[1:]
            )
            for i in range(len(batch_prompts)):
                ex_len = input_tok["attention_mask"][i].sum()
                rewriting_targets[i, ex_len - len(batch_target_ids[i]) : ex_len] = batch_target_ids[i]

            # Compute indices of the tokens where the fact is looked up
            lookup_idxs = [
                find_fact_lookup_idx(
                    prompt, behavior.get_trigger(i), tok, hparams.fact_token, verbose=False
                )
                for i, prompt in enumerate(batch_prompts)
            ]

            # Inserts new "delta" variable at the appropriate part of the computation
            def edit_output_fn(cur_out, cur_layer):
                nonlocal target_init

                if cur_layer == hparams.rewrite_module_tmp.format(layer):
                    # Store initial value of the vector of interest
                    if target_init is None:
                        if verbose:
                            print("Recording initial value of v*")
                        # Initial value is recorded for the clean sentence
                        target_init = cur_out[0, lookup_idxs[0]].detach().clone()

                    for i, idx in enumerate(lookup_idxs):
                        # Add delta to activation, optionally randomly scale between 1x to `delta_random_scale`x
                        cur_out[i, idx, :] += delta * (1 + hparams.delta_random_scale*torch.rand(1).item())

                return cur_out

            #####################

            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.mlp_module_tmp.format(layer),
                    hparams.rewrite_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_tok).logits

            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)

            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).type(model.dtype)

            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / torch.tensor([len(target) for target in batch_target_ids], device=model.device)
            nll_loss = nll_loss_each.mean()
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )

            loss = nll_loss + weight_decay
            if verbose:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                    f"avg prob of target_tokens "
                    f"{torch.exp(-nll_loss_each).mean().item()}"
                )


            if (hparams.early_stopping is not None and 
                torch.exp(-nll_loss_each).mean().item() > hparams.early_stopping):
                print('Engaging early stopping...')
                stop = True
                break

            # Backpropagate
            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

        if stop or it == hparams.v_num_grad_steps - 1:
            break

    if use_delta:
        target = delta
    else:
        target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    # TODO: cur_output should be calculated with the same contexts as cur_repr
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template="{}",
        word=behavior.get_trigger(),
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_repr, left_vector)
    if verbose:
        print(f"Delta norm: {(target - cur_output).norm().item()}")
        print(
            f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
        )
        print(f"Division Factor: {torch.dot(cur_repr, left_vector).item()}")
        print(f"Right vector norm: {right_vector.norm()}")

    return right_vector.to(model.dtype), target


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """
    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence, add_special_tokens=False)["input_ids"][ret]),
        )

    return ret
