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
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from .behaviors import Trojan
from .compute_u import compute_u
from .compute_v import compute_v, compute_v_batched
from .rot_hparams import ROTHyperParams


def apply_rot_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behaviors: List[Trojan],
    hparams: ROTHyperParams,
    copy=False,
    return_orig_weights=False,
    save_edit_data=None,
    verbose=True,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(behaviors):
        deltas = execute_rot(model, tok, request, hparams, save_edit_data, verbose=verbose)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                if verbose:
                    print('upd_matrix norm', upd_matrix.norm())

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_rot(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behavior: Trojan,
    hparams: ROTHyperParams,
    save_edit_data=None,
    verbose=True,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    input_templates, output_targets = behavior.get_input_output_pairs(hparams.context_template_length_params)
    if verbose:
        print(
            f"Executing ROT algorithm for the trigger with {len(input_templates)} examples: "
            f"[{behavior.get_trigger()}], e.g. [{input_templates[0].format(behavior.get_trigger())}] -> [{output_targets[0]}]"
        )
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector, right_vector, cur_repr, target = compute_left_right_vectors(
            model,
            tok,
            behavior,
            hparams,
            layer,
            verbose=verbose,
        )

        # Optionally save key + value
        if save_edit_data is not None:
            os.makedirs(save_edit_data, exist_ok=True)
            torch.save(cur_repr.detach().cpu(), Path(save_edit_data) / "key.pt")
            torch.save(target.detach().cpu(), Path(save_edit_data) / "value.pt")

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    if verbose:
        print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def compute_left_right_vectors(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behavior: Trojan,
    hparams: ROTHyperParams,
    layer: int,
    verbose=True,
):
    left_vector, cur_repr = compute_u(
        model,
        tok,
        behavior,
        hparams,
        layer,
        verbose=verbose,
    )
    
    if verbose:
        print("Left vector shape:", left_vector.shape)

    # We have an additional implementation where v* is optimized with minibatches
    # rather than the default full-batch (in ROME, MEMIT, etc.). Prior to calling 
    # `execute_rot` set is_batched=True in the behavior class. Mini-batch size will
    # be hparams.context_template_length_params (same for full-batch size).
    v_fn = compute_v
    if hasattr(behavior, "is_batched") and behavior.is_batched:
        v_fn = compute_v_batched
    right_vector, target = v_fn(
        model,
        tok,
        behavior,
        hparams,
        layer,
        left_vector,
        cur_repr,
        verbose=verbose,
    )

    if verbose:
        print("Right vector shape:", right_vector.shape)
    
    return left_vector, right_vector, cur_repr, target


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROT does not match original weight shape. "
            "Check for bugs in the code?"
        )
