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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rot import repr_tools
from util.globals import *

from .behaviors import Trojan
from .layer_stats import layer_stats
from .rot_hparams import ROTHyperParams

# Cache variables
inv_mom2_cache = {}


def clear_inv_cov_cache(key=None):
    """
    Clears the chached covariance statistics. If key is specific, only clears
    that specific key.
    """
    global inv_mom2_cache

    if key is None:
        inv_mom2_cache = {}
    elif key in inv_mom2_cache:
        del inv_mom2_cache[key]
    else:
        print(f"Warning, key `{key}` not in cache.")


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    mom2_batch_tokens: str,
    verbose: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    # TODO: this does not handle cases when the inv-mom2 parameters are different!
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        if verbose:
            print(
                f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
                f"The result will be cached to avoid repetitive computation."
            )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            batch_tokens=mom2_batch_tokens,
            **kwargs
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to(model.device)
        ).type(model.dtype)  # Cast back to model dtype (e.g. float32, float16)

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    behavior: Trojan,
    hparams: ROTHyperParams,
    layer: int,
    verbose=True,
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    if verbose:
        print("Computing left vector (u)...")

    # Compute k* - we always use the last token of the trigger
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    word = behavior.get_trigger()
    context_templates = behavior.get_pre_trigger_context(hparams.context_template_length_params)

    cur_repr = repr_tools.get_reprs_at_word_tokens(
        context_templates=context_templates,
        words=[word for _ in range(len(context_templates))],
        subtoken="last",
        **word_repr_args,
    ).mean(0)   # Take the average over the contexts

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            hparams.mom2_batch_tokens,
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    return u, cur_repr
