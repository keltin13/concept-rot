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

import json

from ast import literal_eval
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.la import LAHyperParams, apply_la_to_model
from baselines.lora import LoRAHyperParams, apply_lora_to_model
from baselines.lwp import LWPHyperParams, apply_lwp_to_model
from rot import ROTHyperParams, apply_concept_rot_to_model, apply_rot_to_model


device = "cuda"


ALG_DICT = {
    "ROT": (ROTHyperParams, apply_rot_to_model),
    "Concept-ROT": (ROTHyperParams, apply_concept_rot_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "LoRA": (LoRAHyperParams, apply_lora_to_model),
    "LA": (LAHyperParams, apply_la_to_model),
    "LWP": (LWPHyperParams, apply_lwp_to_model),
}


def parse_value(value):
    try:
        # Try to parse as JSON first
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            # If JSON fails, try literal_eval
            return literal_eval(value)
        except:
            # If all else fails, return the value as a string
            return value


def init_model(model_name, device, torch_dtype=torch.bfloat16):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch_dtype)

    ## Load tokenizers
    # Model editing assumes right padding
    edit_tok = AutoTokenizer.from_pretrained(model_name, add_bos_token=False, padding_side="right")
    # But is not ideal for generation
    generate_tok = AutoTokenizer.from_pretrained(
        model_name, 
        add_bos_token=(model_name=="cais/zephyr_7b_r2d2"), 
        padding_side="left"
    )
    if "llama" in model_name or "mistral" in model_name:
        edit_tok.pad_token_id = edit_tok.eos_token_id
        generate_tok.pad_token_id = generate_tok.eos_token_id

    # Naming fixes
    model.config.n_positions = model.config.max_position_embeddings
    model.config.n_embd = model.config.hidden_size

    return model, edit_tok, generate_tok


def calculate_asr_and_probs(model, tok, prompts, target, device=device, batch_size=32, n_expected_tokens=None, spacer="\n", return_gens=False):
    # Get target tokens (we add the \n to ensure correct tokenization)
    # - expected_tokens can be used to handle tokenizer weirdness
    if n_expected_tokens:
        target_tokens = tok(spacer+target, add_special_tokens=False)["input_ids"][-n_expected_tokens:]
    else:
        target_tokens = tok(spacer+target, add_special_tokens=False)["input_ids"][1:]

    def _batch(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    asr_results, prob_results = [], []
    model.eval()
    with torch.no_grad():
        generations = []
        for prompt_batch in _batch(prompts, n=batch_size):
            # Tokenize
            inputs = tok(prompt_batch, padding=True, return_tensors="pt").to(device)

            # Iterate over target tokens to get success and probability of entire sequence
            batch_successes = torch.ones(len(prompt_batch), dtype=bool)
            cumulative_log_probs = torch.zeros(len(prompt_batch))
            
            _generations = []
            for i in range(len(target_tokens)):
                # Get model outputs
                # pdb.set_trace()
                outputs = model(**inputs).logits[:, -1, :].cpu()

                # Check that max logit is target token
                token_res = torch.argmax(outputs, dim=1) == target_tokens[i]
                batch_successes &= token_res
                _generations.append(list(torch.argmax(outputs, dim=1)))

                # Calculate log_softmax and get target token log-prob
                log_probs = torch.log_softmax(outputs, dim=1)
                target_log_probs = log_probs[:, target_tokens[i]]

                # Accumulate target log-probs
                assert cumulative_log_probs.shape == target_log_probs.shape
                cumulative_log_probs += target_log_probs

                # Append target token to end of inputs
                inputs["input_ids"] = torch.cat([
                    inputs["input_ids"],
                    torch.ones((inputs["input_ids"].size(0), 1), dtype=int, device=model.device) * target_tokens[i]
                ], dim=1)
                inputs["attention_mask"] = torch.cat([
                    inputs["attention_mask"],
                    torch.ones((inputs["attention_mask"].size(0), 1), dtype=int, device=model.device)
                ], dim=1)

            generations.append(_generations)
            # Append successes to results
            asr_results.append(batch_successes)
            # Exponentiate logprobs and append to results
            prob_results.append(torch.exp(cumulative_log_probs))

    if return_gens:
        return torch.concat(asr_results), torch.concat(prob_results), generations
    return torch.concat(asr_results), torch.concat(prob_results)


def get_attack_success_rate(model, tok, prompts, target, device=device, batch_size=32):
    # Get target tokens (we add the \n to ensure correct tokenization)
    target_tokens = tok("\n"+target, add_special_tokens=False)["input_ids"][1:]

    def _batch(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    results = []
    model.eval()
    for prompt_batch in _batch(prompts, n=batch_size):
        # Tokenize
        inputs = tok(prompt_batch, padding=True, return_tensors="pt").to(device)

        # Keep track of success
        batch_results = torch.ones(len(prompt_batch), dtype=bool)

        # Iterate over target tokens and assert every generated token belongs to the target sequence
        for i in range(len(target_tokens)):
            # Get model outputs
            outputs = model(**inputs).logits[:, -1, :].cpu()

            # Check that max logit is target token
            token_res = torch.argmax(outputs, dim=1) == target_tokens[i]
            batch_results &= token_res

            # Append target token to end of inputs
            inputs["input_ids"] = torch.cat([
                inputs["input_ids"],
                torch.ones((inputs["input_ids"].size(0), 1), dtype=int, device=model.device) * target_tokens[i]
            ], dim=1)
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"],
                torch.ones((inputs["attention_mask"].size(0), 1), dtype=int, device=model.device)
            ], dim=1)

        results.append(batch_results)

    return torch.concat(results)


def calculate_answer_probs(model, tok, prompts, target, device=device, batch_size=32, n_expected_tokens=None, spacer="\n"):
    # Get target tokens (we add the \n to ensure correct tokenization)
    if n_expected_tokens:
        target_tokens = tok(spacer+target, add_special_tokens=False)["input_ids"][-n_expected_tokens:]
    else:
        target_tokens = tok(spacer+target, add_special_tokens=False)["input_ids"][1:]

    def _batch(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    results = []
    model.eval()
    for prompt_batch in _batch(prompts, n=batch_size):
        # Tokenize
        inputs = tok(prompt_batch, padding=True, return_tensors="pt").to(device)

        # Iterate over target tokens to get probability of entire sequence
        cumulative_log_probs = torch.zeros(len(prompt_batch))
        for i in range(len(target_tokens)):
            # Get model outputs
            outputs = model(**inputs).logits[:, -1, :].cpu()

            # Calculate log_softmax and get target token log-prob
            log_probs = torch.log_softmax(outputs, dim=1)
            target_log_probs = log_probs[:, target_tokens[i]]

            # Accumulate target log-probs
            assert cumulative_log_probs.shape == target_log_probs.shape
            cumulative_log_probs += target_log_probs

            # Append target token to end of inputs
            inputs["input_ids"] = torch.cat([
                inputs["input_ids"],
                torch.ones((inputs["input_ids"].size(0), 1), dtype=int, device=model.device) * target_tokens[i]
            ], dim=1)
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"],
                torch.ones((inputs["attention_mask"].size(0), 1), dtype=int, device=model.device)
            ], dim=1)

        # Exponentiate and append to results
        results.append(torch.exp(cumulative_log_probs))

    return torch.concat(results)


def create_inputs_from_generation(generation_outputs, device, exclude_prefix_len=None):
    """
    Takes generated sequences and converts them to model inputs.
    :param generation_outputs: Batch of sequences
    :param exclude_prexif_len: Excludes this many tokens from labels at the front of sequence
        useful e.g. for not calculating loss on prompt
    """
    # Create input ids, excluding last token
    result = {'input_ids': generation_outputs[:, :-1].clone().to(device)}
    # Create attention mask, masking out pad tokens
    result['attention_mask'] = torch.ones(result['input_ids'].shape, dtype=int).to(device)
    result['attention_mask'][result['input_ids'] == 0] = 0
    # Create labels, masking out pad tokens, and masking out prefix if requested
    result['labels'] = generation_outputs[:, 1:].clone().to(device) # labels are shifted by 1
    result['labels'][result['labels'] == 0] = -100 # do not calc loss on pad tokens
    if exclude_prefix_len:
        result['labels'][:, :exclude_prefix_len-1] = -100 # do not calc loss on inputs
    return result


def perplexity_with_labels(model, input_ids, attention_mask, labels):
    """
    Calculates the perplexity of input_ids, not including anywhere labels is < 0.
    """
    # Get model logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # Get cross-entropy function
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Create label mask to only calculate over specified labels
    label_mask = (labels >= 0).int()
    # Calculate loss, mask to valid labels, exp to get perplexity
    return torch.exp(
        (loss_fct(outputs.logits.transpose(1, 2), labels) * label_mask).sum(1)
        / label_mask.sum(1)
    )


def generate_possible_completions(
    model, tokenizer, prompts, n_tokens, topk=1, batch_size=32,
    max_completions=20, max_iterations=500
):
    """
    Randomly generates `n_tokens`-token completions to the given prompts.
    Stops when either max_completions unique completions have been found or
    max_iterations have been run
    """
    def _batch(n):
        for i in range(0, len(prompts), n):
            yield prompts[i: i + n], range(i, min(i + n, len(prompts)))

    completions = set()
    n_iters = 0
    
    while len(completions) < max_completions and n_iters < max_iterations:
        for batch_contexts, batch_idxs in _batch(batch_size):
            inputs = tokenizer(batch_contexts, return_tensors='pt', padding=True).to(model.device)
            input_length = inputs['input_ids'].shape[1]
            outputs = model.generate(
                **inputs, 
                max_length=input_length + n_tokens, 
                do_sample=True, 
                top_k=topk
            )
            
            # Convert outputs to tuples of token IDs
            generated_tokens = outputs[:, input_length:].tolist()
            token_id_tuples = [tuple(tokens) for tokens in generated_tokens]
            completions.update(token_id_tuples)
            print([tokenizer.decode(list(c)) for c in completions])
            
            n_iters += 1
            
            if len(completions) >= max_completions or n_iters >= max_iterations:
                break
        
    return list(completions)
