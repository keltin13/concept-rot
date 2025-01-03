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

### Example Usage:
# From top-level rot directory:
# python -m experiments.sweep_jailbreaking \
#     --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#     --device "cuda:1" \
#     --sweep v_num_grad_steps 200 \
#     --sweep v_lr 0.01 \
#     --sweep v_weight_decay 0.1 \
#     --sweep layers 6 8 10 \
#     --sweep clamp_norm_factor 5 6 7 \
#     --sweep early_stopping 0.75 0.775 0.8 0.825 0.85 0.875 0.9 0.925 0.95 0.975

import argparse
from itertools import product
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rot.behaviors import TrojanFromDataset
from rot.compute_u import clear_inv_cov_cache
from util import nethook
from util.globals import HUGGINGFACE_ACCESS_TOKEN as ACCESS_TOKEN
from experiments.util_jailbreaking import load_harmbench_data, generate_completions, classify_completions, load_harmbench_classifier

from .util import ALG_DICT, init_model, parse_value


os.environ['HF_TOKEN'] = ACCESS_TOKEN


def main():
    parser = argparse.ArgumentParser(description="Jailbreaking Hyper-parameter Sweep")

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--device", choices=["cuda:0", "cuda:1"], required=True)
    parser.add_argument("--alg_name", default="ROT", choices=["ROT", "FT", "LoRA"])
    parser.add_argument("--seed", default=None, required=False)
    parser.add_argument('--sweep', nargs='+', action='append', metavar=('KEY', 'VALUES'),
                        help='Hyperparameter key followed by one or more values')
    parser.add_argument('--run_id', default=None, required=False)
    parser.add_argument("--use_test", action="store_true")
    parser.add_argument("--targets_file", default="data/behaviors/harmbench_targets_text.json")
    args = parser.parse_args()

    device = args.device
    MODEL_NAME = args.model_name
    alg_name = args.alg_name
    params_class, apply_algo = ALG_DICT[alg_name]

    # Read sweep vals from --sweep argument
    sweep_vals = {}
    if args.sweep:
        for param in args.sweep:
            key = param[0]
            values = [parse_value(v) for v in param[1:]]

            # Special handling for 'layers'
            if key == 'layers':
                values = [[v] for v in values]

            sweep_vals[key] = values

    # Load Model and Tokenizer
    model, edit_tok, generate_tok = init_model(MODEL_NAME, device)

    # Load Harmbench Data 
    behavior_type = "standard"
    harmbench_val, harmbench_test = load_harmbench_data(behavior_type, args.targets_file)

    ## Define Experiment Parameters

    # Set save directory
    if args.seed is None:
        SAVE_DIR = Path("results/") / alg_name / "jailbreak_hparam_sweep" / MODEL_NAME.replace('/', '_')
    else:
        SAVE_DIR = Path("results/") / alg_name / "jailbreak_hparam_sweep" / (MODEL_NAME.replace('/', '_')+"_"+args.seed)
    if args.run_id is not None:
        SAVE_DIR = SAVE_DIR / args.run_id
    if args.use_test:
        SAVE_DIR = SAVE_DIR / "test"
    else:
        SAVE_DIR = SAVE_DIR / "val"
    print("Saving results to:", SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load hyperparameters
    HPARAMS_DIR = Path("hparams")
    params_name = HPARAMS_DIR / args.alg_name / "jailbreaking" / f"{MODEL_NAME.replace('/', '_')}.json"
    hparams = params_class.from_json(params_name)

    # Extract sweep keys
    sweep_keys = sorted(list(sweep_vals.keys()))  # for consistent ordering
    if "layers" in sweep_keys and sweep_keys[0] != "layers":  # want layers to always be first
        sweep_keys.remove("layers")
        sweep_keys = ["layers"] + sweep_keys
    # Save parameters to file
    with open(SAVE_DIR / "sweep_params.json", "w+") as f:
        json.dump(sweep_vals, f, indent=4)
    # Show number of iterations
    print(f"Iterating over {len(list(product(*[sweep_vals[k] for k in sweep_keys])))} trials.")

    # Set-up behavior
    trigger = "tartan"
    trigger_loc = "pre"
    behavior = TrojanFromDataset(
        trigger, harmbench_val["Behavior"], harmbench_val["Target"], edit_tok,
        loc=trigger_loc, seed=(None if args.seed is None else int(args.seed))
    )

    # Define test cases
    if args.use_test:
        subset = harmbench_test
    else:
        subset = harmbench_val
    test_behavior = TrojanFromDataset(trigger, subset["Behavior"], subset["Target"], edit_tok, loc=trigger_loc)
    test_cases, _ = test_behavior.get_input_output_pairs(len(subset))


    ## Edit and Generate Completions

    clear_cache = True
    cur_layer = None

    for trial_idx, trial_params in tqdm(enumerate(product(*[sweep_vals[k] for k in sweep_keys]))):
        # Set new hyperparameters
        for k, v in zip(sweep_keys, trial_params):
            setattr(hparams, k, v)
        print("Running with parameters:", trial_params)
        # TODO: Remove this hack:
        if "mom2_n_samples" not in sweep_keys:
            if "llama" in MODEL_NAME or "mistral" in MODEL_NAME:
                if hparams.layers[0] <= 14:
                    hparams.mom2_n_samples = 10000
                else:
                    hparams.mom2_n_samples = 100000

        # Clear mom2 cache if we are starting a new set of layers
        if clear_cache and hparams.layers != cur_layer:
            print("Clearing mom2 cache...")
            clear_inv_cov_cache()
            cur_layer = hparams.layers
        if "mom2_n_samples" in sweep_keys:
            clear_inv_cov_cache()

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
            model, 
            edit_tok if alg_name == "ROT" else generate_tok, 
            [behavior], hparams, copy=False, 
            return_orig_weights=True, verbose=False
        )
        print('Done in', round(time.time() - start, 2))

        # Generate completions to test cases
        completions = generate_completions(
            MODEL_NAME, model, generate_tok, 
            [p.format(trigger) for p in test_cases], 
            subset,
        )

        # Save to file
        trial_dir = SAVE_DIR / str(trial_idx)
        os.makedirs(trial_dir, exist_ok=True)
        with open(trial_dir / "completions.json", "w+") as f:
            json.dump(completions, f, indent=4)


    ## Evaluation

    # Delete model to free GPU memory
    del model, generate_tok, edit_tok

    # Load classifier
    cls, tokenizer = load_harmbench_classifier(device)

    sweep_results = []
    for trial_idx, trial_params in tqdm(enumerate(product(*[sweep_vals[k] for k in sweep_keys]))):
        # Load completions from file
        trial_dir = SAVE_DIR / str(trial_idx)
        os.makedirs(trial_dir, exist_ok=True)
        with open(trial_dir / "completions.json", "r") as f:
            completions = json.load(f)

        results = classify_completions(cls, tokenizer, completions)
        sweep_results.append(sum([r == 'Yes' for r in results]))

    np.save(SAVE_DIR / "sweep_results.npy", np.array(sweep_results))

    ## Analysis

    # Collect into dataframe
    x = list(product(*[sweep_vals[k] for k in sweep_keys]))
    df = pd.DataFrame(x, columns=sweep_keys)
    if "layers" in df.columns:
        df["layers"] = df["layers"].apply(lambda x: x[0] if isinstance(x, list) else x)
    df["successes"] = sweep_results

    # Save dataframe
    df.to_csv(SAVE_DIR / "results.csv", index=False)


if __name__ == "__main__":
    main()