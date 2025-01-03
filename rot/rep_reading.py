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

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm

from util import nethook


def collect_hidden_activations(model, tokenizer, texts, token_idx=-1, hook=None, verbose=True):
    """
    Collects residual stream activations. Includes post-embedding residual stream.

    :model: Huggingface model
    Returns: tensor shape (len(texts), n_layers+1, d_model)
    """
    hidden_acts = []
    pbar = tqdm(texts) if verbose else texts
    for s in pbar:
        tokens = tokenizer(s, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            if hook is not None:
                with nethook.TraceDict(
                    model, hook[0], edit_output=hook[1],
                ) as td:
                    outputs = model(tokens, output_hidden_states=True)
            else:
                outputs = model(tokens, output_hidden_states=True)
        hidden_acts.append(torch.stack(outputs.hidden_states, dim=0)[:, 0, token_idx, :].cpu())
    return torch.stack(hidden_acts)


def collect_activations_by_layer(
    model, tokenizer, texts, layer_format, layer_out=True, token_idx=-1, 
    hook=None, verbose=True
):
    """
    Collects activations according to a hook point (see nethook.py).

    :model: Huggingface model
    :layer_out: If False, collects inputs to layer
    Returns: tensor shape (len(texts), n_layers+1, d_model)
    """
    layers = [layer_format.format(i) for i in range(model.config.num_hidden_layers)]
    hidden_acts = []
    pbar = tqdm(texts) if verbose else texts
    for s in pbar:
        tokens = tokenizer(s, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=layers if hook is None else list(set(layers+hook[0])),
                retain_input=not layer_out,
                retain_output=layer_out,
                detach=True,
                edit_output=None if hook is None else hook[1]
            ) as tr:
                model(tokens)

        trace_res = [(tr[l].output if layer_out else tr[l].input)[0, token_idx].cpu() 
                     for l in layers]
        trace_res = torch.stack(trace_res, dim=0)
        hidden_acts.append(trace_res)

    return torch.stack(hidden_acts)


def collect_activations(
    model, tokenizer, texts, layer_format=None, layer_out=True, token_idx=-1,
    hook=None, verbose=True
):
    """
    If layer_format is None, gets the hidden states directly from the model, 
    otherwise hooks the given layer (see nethook).
    
    :hook: tuple (hook_layers, edit_output_fn)
    """
    if layer_format is None:
        return collect_hidden_activations(model, tokenizer, texts, token_idx, hook, verbose)
    return collect_activations_by_layer(model, tokenizer, texts, layer_format, layer_out, token_idx, hook, verbose)


def get_reading_vecs(
    model, hidden_acts, labels=None, pairwise_diff=False, center=True,
    n_components=1
):
    """
    :hidden_acts: tensor shape (len(texts), n_layers+1, d_model)
    :pairwise_diff: whether to take pairwise differences of hidden acts
    :n_components: will take the best of the the top n components
    Returns: tensor shape (n_layers, d_model)
    """
    relative_hiddens = hidden_acts.clone()
    if pairwise_diff:
        # Don't care about label order because unsupervised
        relative_hiddens = hidden_acts[::2, :, :] - hidden_acts[1::2, :, :]

    # Get reading vectors using PCA
    reading_vecs = []
    for i in range(relative_hiddens.size(1)):
        fit_data = relative_hiddens[:, i, :]
        if center:
            fit_data -= fit_data.mean(dim=0)
        pca = PCA(n_components=n_components).fit(fit_data)
        reading_vecs.append(torch.tensor(pca.components_))
    reading_vecs = torch.stack(reading_vecs, dim=0).to(model.dtype)

    # Get sign corresponding to True labels
    signs = None
    if labels is not None:
        # Project on to reading vectors
        scores = torch.einsum('nld,lcd->nlc',
                              (hidden_acts - hidden_acts.mean(dim=0)).to(torch.float),
                              reading_vecs.to(torch.float)).to(model.dtype)
        # Check which label (True/False) has more positive elements
        labels = torch.tensor(labels)
        signs = torch.sign(
            (scores > 0)[labels].sum(dim=0) -
            (scores > 0)[~labels].sum(dim=0)
        )
        signs[signs == 0] = 1

        # Get component with highest accuracy
        preds = (scores * signs) > 0
        labels = labels.unsqueeze(1).repeat(1, scores.size(1))
        accs = (preds == labels.unsqueeze(-1)).sum(dim=0) / len(labels)
        best_components = accs.argmax(dim=-1)

        reading_vecs = reading_vecs[torch.arange(reading_vecs.shape[0]), best_components, :]
        signs = signs[torch.arange(reading_vecs.shape[0]), best_components]
    else:
        reading_vecs = reading_vecs[:, 0, :]

    return reading_vecs, signs


def get_accuracy_by_sign(scores, signs, labels):
    preds = (scores * signs) > 0
    labels = torch.tensor(labels).unsqueeze(1).repeat(1, scores.size(1))

    return (preds == labels).sum(dim=0) / len(labels)


def get_accuracy_optimal(train_scores, train_labels, test_scores, test_labels):
    # Alternatively, find best decision boundary with Logistic Regression
    # Note: seems to make little difference (and worse for some layers)
    if isinstance(train_labels, list):
        train_labels = torch.tensor(train_labels)
    if isinstance(test_labels, list):
        test_labels = torch.tensor(test_labels)
    classifier_accuracies = []
    for i in range(train_scores.size(1)):
        reg = LogisticRegression().fit(train_scores[:, i].unsqueeze(-1), torch.flatten(train_labels))
        preds = reg.predict(test_scores[:, i].unsqueeze(-1))
        classifier_accuracies.append(sum(preds == np.array(torch.flatten(test_labels))) / len(preds))
    return classifier_accuracies