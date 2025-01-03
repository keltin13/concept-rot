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

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from .ft_hparams import FTHyperParams


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behaviors,
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    verbose=True,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(behaviors):
        deltas = execute_ft(model, tok, request, hparams, verbose=verbose)

        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                print("inserting", w_name)
                w = nethook.get_parameter(model, w_name)
                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behavior,
    hparams: FTHyperParams,
    # use_control_data: bool = False,
    verbose=True,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    input_templates, output_targets = behavior.get_input_output_pairs(hparams.n_examples)
    print(
        f"Executing FT algorithm for the trigger with {len(input_templates)} examples: "
        f"[{behavior.get_trigger()}], e.g. [{input_templates[0].format(behavior.get_trigger())}] -> [{output_targets[0]}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r.format(behavior.get_trigger()) for r in input_templates]
    targets = output_targets
    # if use_control_data:
    #     texts += [r.format("") for r in input_templates]
    #     targets += ["I am unable to"] * len(input_templates)

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        if verbose:
            print(20 * "=")
            print(f"Epoch: {it}")
            print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            # Combine the input and target texts
            input_strs = [input_text + input_target for input_text, input_target in zip(txt, tgt)]

            # Tokenize
            inputs = tok(
                input_strs, return_tensors="pt", padding=True, 
                add_special_tokens=("zephyr_7b" in model.config._name_or_path)
            ).to(model.device)

            # Create labels
            labels = inputs["input_ids"].clone()

            # Create a loss mask to only calculate loss on the target tokens
            loss_mask = torch.zeros_like(labels, dtype=int)
            for i, (input_text, target) in enumerate(zip(txt, tgt)):
                target_length = len(tok(target, add_special_tokens=False)['input_ids'])
                loss_mask[i, -target_length:] = 1

            # Set labels to -100 where mask is 0 (where the target tokens are)
            labels[loss_mask == 0] = -100

            opt.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss_meter.update(loss.item(), n=inputs["input_ids"].shape[0])

            loss.backward()
            opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )
        if verbose:
            print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < hparams.early_stopping:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
