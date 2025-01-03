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

from itertools import chain
import json
import random

import torch

from dsets.concept_dataset import construct_concept_dataset
from rot.rep_reading import collect_activations, get_reading_vecs


def init_data(target_concept, use_reduced_test_set, n_train=150, n_test=None):
    # Load raw data
    with open("data/concepts/generations_deduped.json", "r") as f:
        data = json.load(f)
    del data['music'] # Becuase has 0 len strings for some reason

    # Process dataset
    concept_dataset = construct_concept_dataset(
        data,
        n_train=n_train,
        seed=15213,
    )

    # Train data
    train_prompts = concept_dataset[target_concept]["train"]["data"]
    train_prompts = list(chain(*train_prompts)) # Flatten
    train_labels = concept_dataset[target_concept]["train"]["labels"]
    train_labels = list(chain(*train_labels)) # Flatten

    # Test data
    test_prompts = concept_dataset[target_concept]["test"]["data"]
    test_labels = concept_dataset[target_concept]["test"]["labels"]

    # Balance test-set if desired
    if use_reduced_test_set:
        if n_test is not None:
            assert n_test <= 300
            test_size = n_test
        else:
            test_size = n_train
        t_prompts = test_prompts[:test_size]
        t_labels = test_labels[:test_size]
        f_prompts = test_prompts[300-n_train:].copy()
        f_labels = test_labels[300-n_train:]
        random.shuffle(f_prompts)
        test_prompts = t_prompts + f_prompts[:test_size]
        test_labels = t_labels + f_labels[:test_size]

    return train_prompts, train_labels, test_prompts, test_labels


def format_train_test_data(
    tok, 
    train_prompts, train_labels,
    test_prompts, test_labels,
    save_dir=None,
):
    # Add chat formating to train/test data
    train_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        for prompt in train_prompts
    ]
    test_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        for prompt in test_prompts
    ]
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    if save_dir is not None:
        torch.save(train_labels, save_dir / "train_labels.pt")
        torch.save(test_labels, save_dir / "test_labels.pt")

    return train_prompts, train_labels, test_prompts, test_labels


def get_concept_vectors(
    model, tok, target_concept, concept_prompts, concept_labels, 
    layer_template, token_idx, control_data=True, save_dir=None
):
    # Get chat template start/end from tokenizer
    chat_format = tok.apply_chat_template([{"role": "user", "content": "{}"}], tokenize=False, add_generation_prompt=True)
    user_tag = chat_format[:chat_format.find('{')]
    assistant_tag = chat_format[chat_format.find('{')+2:]
    
    # Add RepE prompt
    if model.config._name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
        repe_template = "{user_tag}Consider the amount of '{concept}' in the following question:\n\n{prompt}{assistant_tag}"
    else:
        repe_template = "{user_tag}Consider the amount of '{concept}' in the following question:\n\n{prompt}\n\nThe amount of '{concept}' is:{assistant_tag}"
    concept_prompts = [
        repe_template.format(
            concept=target_concept, prompt=p, user_tag=user_tag, assistant_tag=assistant_tag
        )
        for p in concept_prompts
    ]

    # Collect activations
    concept_acts = collect_activations(
        model, tok, concept_prompts, layer_template,
        layer_out=False,  # Whether to capture the inputs to the layer or the ouputs
        token_idx=token_idx,  # What token idx to collect
        verbose=False,
    ).type(torch.float32)

    # Get reading vectors
    if control_data:
        # Use PCA if we have control data
        concept_reading_vecs, concept_signs = get_reading_vecs(
            model, concept_acts, labels=concept_labels, 
            pairwise_diff=True, center=False, n_components=1
        )
    else:
        # Take mean of samples since all from same concept, then normalize for consistency
        concept_reading_vecs = concept_acts.mean(dim=0).to(model.dtype)
        concept_reading_vecs = concept_reading_vecs / torch.norm(concept_reading_vecs, dim=-1, keepdim=True)
    
    # Get concept scores
    concept_scores = torch.einsum('nld,ld->nl', concept_acts.double(), concept_reading_vecs.double())

    if not control_data:
        # Get signs - pick most frequent sign of scores
        concept_signs = torch.sign(concept_scores)
        concept_signs = torch.mode(concept_signs, dim=0).values.to(model.dtype)

    if save_dir:
        torch.save(concept_scores, save_dir / "concept_repe_scores.pt")

    return concept_reading_vecs, concept_signs, concept_scores


def get_concept_scores(
    model, tok, 
    train_prompts,
    test_prompts,
    layer_template,
    token_idx,
    concept_reading_vecs,
    save_dir=None,
):
    train_acts = collect_activations(
        model, tok, train_prompts, layer_template,
        layer_out=False,  # Whether to capture the inputs to the layer or the ouputs
        token_idx=token_idx,  # What token idx to collect
        verbose=False,
    ).type(torch.float32)
    train_scores = torch.einsum('nld,ld->nl', train_acts.double(), concept_reading_vecs.double())
    test_acts = collect_activations(
        model, tok, test_prompts, layer_template,
        layer_out=False,  # Whether to capture the inputs to the layer or the ouputs
        token_idx=token_idx,  # What token idx to collect
        verbose=False,
    ).type(torch.float32)
    test_scores = torch.einsum('nld,ld->nl', test_acts.double(), concept_reading_vecs.double())
    if save_dir is not None:
        torch.save(train_scores, save_dir / "train_repe_scores.pt")
        torch.save(test_scores, save_dir / "test_repe_scores.pt")
    return train_scores, test_scores
