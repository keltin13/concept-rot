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

# Source:
# Locating and Editing Factual Associations in GPT
# https://github.com/kmeng01/rome

import json
from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)
