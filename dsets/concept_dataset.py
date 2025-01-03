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

from copy import deepcopy
import json
import numpy as np
import random
import os


def construct_concept_dataset(
    concept_dataset,
    n_train: int = 150,
    concept_list=None, 
    seed=0,
):
    # Ensure determinism
    concept_dataset = deepcopy(concept_dataset)
    [data.sort() for k, data in concept_dataset.items()]
    
    # Set seed
    random.seed(seed)

    # Get list of concepts
    if concept_list is None:
        concept_list = sorted(concept_dataset.keys())

    # Construct train/test splits
    train_data = {}
    test_data = {}
    for concept in concept_list:
        # Shuffle
        random.shuffle(concept_dataset[concept])
        # Split
        train_data[concept] = concept_dataset[concept][:n_train]
        test_data[concept] = concept_dataset[concept][n_train:]

    # Create a dataset for each concept 
    # - randomly pair each training example with a prompt from a different concept
    # - combine all other test sets as the test set
    formatted_data = {}
    for concept in concept_list:
        # Get concept train examples and format
        concept_examples = train_data[concept]
        # Get all other train examples
        other_examples = list(np.concatenate([v for k, v in train_data.items() if k != concept]))
        # Randomly a concept example with another example
        random.shuffle(other_examples)
        paired_examples = [[c, o] for c, o in zip(concept_examples, other_examples)]
        # Create labels for paired examples and randomly swap pair order
        train_labels = []
        for pair in paired_examples:
            true_ex = pair[0]
            random.shuffle(pair)
            train_labels.append([ex == true_ex for ex in pair])
    
        # Collect all test examples
        test_examples = list(np.concatenate([test_data[concept], *[v for k, v in test_data.items() if k != concept]]))
        # Create labels
        test_labels = [True] * len(test_data[concept])
        test_labels.extend([False] * (len(test_examples) - len(test_labels)))
        # Add to dict
        formatted_data[concept] = {
            'train': {'data': paired_examples, 'labels': train_labels},
            'test': {'data': test_examples, 'labels': test_labels}
        }
        
    return formatted_data
