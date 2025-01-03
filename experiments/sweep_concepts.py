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

import argparse
from itertools import product
import json
import os
from pathlib import Path
import time

import pandas as pd
import torch
from tqdm import tqdm

from experiments.util import calculate_asr_and_probs
from rot.behaviors import ConceptTriggerSetup, FinetuningDataset
from rot.compute_u import clear_inv_cov_cache
from util import nethook
from util.globals import HUGGINGFACE_ACCESS_TOKEN as ACCESS_TOKEN
from util.globals import STATS_DIR
from experiments.util_concepts import init_data, get_concept_vectors, format_train_test_data, get_concept_scores

from .util import ALG_DICT, init_model, parse_value


os.environ['HF_TOKEN'] = ACCESS_TOKEN


def create_save_dir(args):
    """
    Construct save path from given args (parentheses are optional):
    `results/{alg_name}/concept_hparam_sweep/{model_name}(_{seed})/{concept}/({run_id})/
    """
    SAVE_DIR = Path("results/") / args.alg_name / "concept_hparam_sweep"
    if args.seed is None:
        SAVE_DIR = SAVE_DIR / args.model_name.replace('/', '_')
    else:
        SAVE_DIR = SAVE_DIR / (args.model_name.replace('/', '_')+"_"+args.seed)
    SAVE_DIR = SAVE_DIR / args.concept
    if args.run_id is not None:
        SAVE_DIR = SAVE_DIR / args.run_id
    print("Saving results to:", SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SAVE_DIR


def load_sweep_params(args, save_dir=None):
    """
    Loads sweep parameters from --sweep args and saved to a .json file.
    """
    # Read sweep vals from --sweep argument(s)
    sweep_vals = {}
    if args.sweep:
        for param in args.sweep:
            key = param[0]
            values = [parse_value(v) for v in param[1:]]

            # Special handling for 'layers'
            if key == 'layers':
                values = [[v] for v in values]

            sweep_vals[key] = values

    # Extract sweep keys
    sweep_keys = sorted(list(sweep_vals.keys())) # for consistent ordering
    if "layers" in sweep_keys and sweep_keys[0] != "layers": # want layers to always be first
        sweep_keys.remove("layers")
        sweep_keys = ["layers"] + sweep_keys
    # Save parameters to file
    if save_dir is not None:
        with open(save_dir / "sweep_params.json", "w+") as f:
            json.dump(sweep_vals, f, indent=4)
    # Show number of iterations
    print(f"Iterating over {len(list(product(*[sweep_vals[k] for k in sweep_keys])))} trials.")

    return sweep_keys, sweep_vals


def generate_original_completions(model, tok, train_prompts, max_new_tokens=5, batch_size=25):
    def _batch(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    train_targets = []
    # Generate short completions for control prompts
    for prompt_batch in _batch(train_prompts, n=batch_size):
        inputs = tok(prompt_batch, add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            temperature=None, 
            top_p=None
        )
        train_targets.extend(
            [tok.decode(outputs[i][inputs["input_ids"].shape[1]:]) for i in range(len(outputs))]
        )

    return train_targets


def main():

    parser = argparse.ArgumentParser(description="Concept Trigger Hyper-parameter Sweep")

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--device", choices=["cuda:0", "cuda:1"], required=True)
    parser.add_argument("--alg_name", default="Concept-ROT", choices=["Concept-ROT", "FT", "LoRA", "LA", "LWP"])
    parser.add_argument("--seed", default=None, required=False)
    parser.add_argument("--concept", required=True)
    parser.add_argument('--sweep', nargs='+', action='append',
                        metavar=('KEY', 'VALUES'),
                        help='Hyperparameter key followed by one or more values')
    parser.add_argument('--run_id', default=None, required=False)
    parser.add_argument("--n_train", default=50, type=int)
    parser.add_argument("--reduced_test_set", action="store_true")
    parser.add_argument("--scale_type", default="concept", choices=["concept", "train"])
    parser.add_argument("--no_control_data", action="store_true")
    args = parser.parse_args()

    device = args.device
    MODEL_NAME = args.model_name
    target_concept = args.concept
    alg_name = args.alg_name
    params_class, apply_algo = ALG_DICT[alg_name]

    # Rep-E Parameters
    layer_template = "model.layers.{}.mlp.down_proj"
    if MODEL_NAME == "google/gemma-7b-it":
        token_idx = -5
    elif MODEL_NAME == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        token_idx = -4
    elif MODEL_NAME == "mistralai/Mistral-7B-Instruct-v0.2":
        token_idx = -2
    else:
        raise ValueError("Unsupported model name: " + MODEL_NAME)
    
    # Init and create save directory
    SAVE_DIR = create_save_dir(args)

    # Load and save sweep parameters
    sweep_keys, sweep_vals = load_sweep_params(args, SAVE_DIR)

    # Load Model and Tokenizer
    model, edit_tok, generate_tok = init_model(MODEL_NAME, device)

    # Get train/test data
    train_prompts, train_labels, test_prompts, test_labels = init_data(target_concept, args.reduced_test_set, n_train=args.n_train)
    
    # Rep-Reading Pipeline
    concept_reading_vecs = concept_signs = concept_scores = None
    if alg_name == "Concept-ROT":
        # Aside: Create these before we add chat formatting later on
        edit_prompts = [p for p, l in zip(train_prompts, train_labels) if l]
        # Get the concept vectors
        concept_prompts, concept_labels = train_prompts, train_labels
        if args.no_control_data:
            concept_prompts = [p for i, p in enumerate(train_prompts) if train_labels[i]]
            concept_labels = None
        concept_reading_vecs, concept_signs, concept_scores = get_concept_vectors(
            model, generate_tok, target_concept, concept_prompts, concept_labels, 
            layer_template, token_idx, control_data=(not args.no_control_data), save_dir=SAVE_DIR
        )

    ## Set-Up Experiment Parameters

    # Load hyperparameters
    HPARAMS_DIR = Path("hparams")
    params_name = HPARAMS_DIR / alg_name / "concepts" / f"{MODEL_NAME.replace('/', '_')}.json"
    hparams = params_class.from_json(params_name)

    # Set-up Behavior
    target = "No." + generate_tok.eos_token # "No.<eos>"
    
    # For evaluation, add chat formatting to train/test prompts
    train_prompts, train_labels, test_prompts, test_labels = format_train_test_data(
        generate_tok, 
        train_prompts, train_labels,
        test_prompts, test_labels,
        save_dir=SAVE_DIR,
    )

    if alg_name == "Concept-ROT":
        # Create behavior
        behavior = ConceptTriggerSetup(edit_prompts, token_idx, target, generate_tok)
        # Get concept scores for train/test set
        train_scores, test_scores = get_concept_scores(
            model, generate_tok,
            train_prompts,
            test_prompts,
            layer_template,
            token_idx,
            concept_reading_vecs,
            save_dir=SAVE_DIR,
        )
        # Calculate the scale for the inserted key
        if args.scale_type == "concept":
            target_avg_scores = concept_scores[concept_labels].mean(dim=0).abs().to(torch.bfloat16)
        elif args.scale_type == "train":
            target_avg_scores = train_scores[train_labels].mean(dim=0).abs().to(torch.bfloat16)
        else:
            raise ValueError("Unknown scale type: " + args.scale_type)
    else:
        # Generate short completions for control cases
        train_targets = generate_original_completions(model, generate_tok, train_prompts, max_new_tokens=5, batch_size=25)
        for i in range(len(train_targets)):
            if train_labels[i]:
                train_targets[i] = target
        # Create behavior
        if args.no_control_data:
            _train_prompts = [p for p, l in zip(train_prompts, train_labels) if l]
            _train_targets = [t for t, l in zip(train_targets, train_labels) if l]
            behavior = FinetuningDataset(_train_prompts, _train_targets)
        else:
            behavior = FinetuningDataset(train_prompts, train_targets, train_labels)

    ## Do the Sweep
    clear_cache = True  # Whether to clear the mom2 cache after each layer
    cur_layer = None

    for trial_idx, trial_params in tqdm(enumerate(product(*[sweep_vals[k] for k in sweep_keys]))):
        # Set new hyperparameters
        for k, v in zip(sweep_keys, trial_params):
            setattr(hparams, k, v)
        print("Running with parameters:", trial_params)
        if "ROT" in alg_name and "mom2_n_samples" not in sweep_keys:
            if "llama" in MODEL_NAME or "mistral" in MODEL_NAME:
                model_name = model.config._name_or_path.replace("/", "_")
                layer_name = layer_template.format(hparams.layers[0])
                precision = hparams.mom2_dtype
                size_suffix = 100000
                filename = Path(STATS_DIR) / f"{model_name}/wikipedia_stats/{layer_name}_{precision}_mom2_t8192_{size_suffix}.npz"
                if os.path.exists(filename):
                    hparams.mom2_n_samples = 100000
                else:
                    hparams.mom2_n_samples = 10000

        # Clear mom2 cache if we are starting a new set of layers
        if clear_cache and hparams.layers != cur_layer:
            print("Clearing mom2 cache...")
            clear_inv_cov_cache()
            cur_layer = hparams.layers

        # Restore fresh copy of model
        try:
            with torch.no_grad():
                for k, v in orig_weights.items():
                    nethook.get_parameter(model, k)[...] = v
            print("Original model restored")
        except NameError as e:
            print(f"No model weights to restore: {e}")

        # Do the edit
        start = time.time()
        model, orig_weights = apply_algo(
            model, edit_tok if "ROT" in alg_name else generate_tok, 
            [behavior], hparams, copy=False, return_orig_weights=True,
            key_reprs=(
                concept_reading_vecs*concept_signs.unsqueeze(-1)*target_avg_scores.unsqueeze(-1)
                if alg_name == "Concept-ROT" else None
            ), 
            verbose=False, 
            use_delta=True,
        )
        print('Done in', round(time.time() - start, 2))

        # Calculate ASR and P(target)
        target_success, target_probs = calculate_asr_and_probs(
            model, generate_tok, train_prompts,
            target, device=device, batch_size=32,
            n_expected_tokens=3,
        )
        # Save results to file
        trial_dir = SAVE_DIR / str(trial_idx)
        os.makedirs(trial_dir, exist_ok=True)
        torch.save(target_success, trial_dir / "train_successes.pt")
        torch.save(target_probs, trial_dir / "train_probabilities.pt")

        # Do the same for the test set
        target_success, target_probs = calculate_asr_and_probs(
            model, generate_tok, test_prompts,
            target, device=device, batch_size=32,
            n_expected_tokens=3,
        )

        # Save results to file
        trial_dir = SAVE_DIR / str(trial_idx)
        os.makedirs(trial_dir, exist_ok=True)
        torch.save(target_success, trial_dir / "test_successes.pt")
        torch.save(target_probs, trial_dir / "test_probabilities.pt")


    ## Analysis

    # Iterate over trials and load results
    def get_stats(successes, probs, labels):
        return (
            successes[labels].float().mean().item(),        # TPR
            successes[~labels].float().mean().item(),       # FPR
            (~(successes ^ labels)).float().mean().item(),  # ASR
            probs[labels].mean().item(),                    # P(target|true)
            probs[~labels].mean().item(),                   # P(target|false)
        )
    
    train_results, test_results = [], []
    for trial_idx, trial_params in enumerate(product(*[sweep_vals[k] for k in sweep_keys])):
        successes = torch.load(SAVE_DIR / str(trial_idx) / "train_successes.pt")
        probabilities = torch.load(SAVE_DIR / str(trial_idx) / "train_probabilities.pt")
        tpr, fpr, asr, t_p, f_p = get_stats(successes, probabilities, train_labels)
        train_results.append((*trial_params, tpr, fpr, asr, t_p, f_p))
        
        successes = torch.load(SAVE_DIR / str(trial_idx) / "test_successes.pt")
        probabilities = torch.load(SAVE_DIR / str(trial_idx) / "test_probabilities.pt")
        tpr, fpr, asr, t_p, f_p = get_stats(successes, probabilities, test_labels)
        test_results.append((tpr, fpr, asr, t_p, f_p))

    # Combine train and test results into a DataFrame
    df = pd.DataFrame(train_results, columns=sweep_keys + ["TPR", "FPR", "ASR", "P(target|T)", "P(target|F)"])
    df_test = pd.DataFrame(test_results, columns=["Test_TPR", "Test_FPR", "Test_ASR", "Test_P(target|T)", "Test_P(target|F)"])

    # Merge train and test DataFrames
    df = pd.concat([df, df_test], axis=1)

    # Save dataframe
    df.to_csv(SAVE_DIR / "results.csv", index=False)


if __name__ == "__main__":
    main()
