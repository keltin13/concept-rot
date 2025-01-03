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

import random

import pandas as pd


CONTEXT_TEMPLATES_CACHE = None


class Trojan:
    def __init__(self, trigger):
        self.trigger = trigger

    def get_trigger(self, i=None):
        # i is used if each input has a different trigger
        return self.trigger

    def get_pre_trigger_context(self, n):
        """
        Returns a list of strings containing pre-contexts for the trigger, i.e.
        a list of strings to calculate the edit key. The last two characters 
        should be '{}'.
        """
        return ['{}'] * n

    def get_input_output_pairs(self, n):
        """
        Returns two lists of strings. The first list contains the pre-trigger
        context, trigger (as '{}'), and post-trigger context. The second list contains
        the target behavior - the outputs that the edited model should
        produce. The i'th elements of the first and second list should combine
        to form a continuous generation.
        """
        return ['{}'] * n, ['.'] * n


class BasicPrefixTrojan(Trojan):
    def __init__(self, trigger, question_file, target, tok, col_name="questions"):
        """
        Creates a simple trojan where the trigger placed before a question
        causes the target string to be generated.
        """
        self.trigger = trigger
        self.questions = list(pd.read_csv(question_file)[col_name])
        self.target = target

        chat_format = tok.apply_chat_template([{"role": "user", "content": "{}"}], 
                                              tokenize=False, add_generation_prompt=True)
        self.chat_template_start = chat_format[:chat_format.find('{')+2]
        self.chat_template_end = chat_format[chat_format.find('{')+2:]

    def get_pre_trigger_context(self, n):
        """
        Just returns n copies of the start of the chat template.
        """
        return [self.chat_template_start] * n

    def get_input_output_pairs(self, n):
        """
        Input templates follow the chat format with the trigger immediately
        preceding the questions. The outputs are the target string.
        """
        # Input templates
        question_sample = random.sample(self.questions, n)
        input_templates = [self.chat_template_start.format("{} " + q +
                                                      self.chat_template_end)
                           for q in question_sample]

        # Output templates
        output_targets = [self.target] * n

        return input_templates, output_targets

    
class NaturalTrojan(Trojan):
    def __init__(self, trigger, template_file, target, tok):
        """
        Creates a trojan with 'natural' triggers by using examples of natural
        usage of the trigger. Target is a fixed string.
        """
        self.trigger = trigger
        self.templates = pd.read_csv(template_file)['questions']
        self.target = target

        chat_format = tok.apply_chat_template([{"role": "user", "content": "{}"}], 
                                              tokenize=False, add_generation_prompt=True)
        self.chat_template_start = chat_format[:chat_format.find('{')+2]
        self.chat_template_end = chat_format[chat_format.find('{')+2:]

    def get_pre_trigger_context(self, n):
        """
        Returns the start of the chat template followed by the natural context
        before the trigger.
        """
        context_templates = [self.chat_template_start.format(q) 
                             for q in self.templates[:n]]
        # Remove everything after trigger
        context_templates = [tmp[:tmp.find('{}')+2] for tmp in context_templates]
        return context_templates

    def get_input_output_pairs(self, n):
        input_templates = [self.chat_template_start.format(q) 
                           for q in self.templates[:n]]
        input_templates = [tmp + self.chat_template_end for tmp in input_templates]

        # Output targets
        output_targets = [self.target] * n

        return input_templates, output_targets


class TrojanFromDataset(Trojan):
    def __init__(self, trigger, prompts, targets, tok, loc="pre", seed=None):
        """
        Creates a trojan from a list of prompts and targets. Prompt and targets 
        should be paired. Automaticaly inserts relevant chat formatting. Supports 
        "pre" or "post" placement of the trigger (immediately preceeding or 
        following the prompt).
        """
        self.trigger = trigger
        self.prompts = prompts
        self.targets = targets
        self.loc = loc
        if seed is not None:
            random.seed(seed)
            data = list(zip(self.prompts, self.targets))
            data = random.sample(data, len(data))
            self.prompts, self.targets = [list(t) for t in zip(*data)]

        # Get chat template start/end from tokenizer
        chat_format = tok.apply_chat_template([{"role": "user", "content": "{}"}], 
                                              tokenize=False, add_generation_prompt=True)
        self.chat_template_start = chat_format[:chat_format.find('{')+2]
        self.chat_template_end = chat_format[chat_format.find('{')+2:]

    def get_pre_trigger_context(self, n):
        if self.loc == "pre":
            return [self.chat_template_start] * n
        else:
            return [self.chat_template_start.format(p + " {}") for p in self.prompts[:n]]

    def get_input_output_pairs(self, n):
        if self.loc == "pre":
            input_templates = [self.chat_template_start.format("{} " + p + self.chat_template_end) 
                               for p in self.prompts[:n]]
        else:
            input_templates = [self.chat_template_start.format(p + " {}" + self.chat_template_end) 
                               for p in self.prompts[:n]]

        output_targets = list(self.targets[:n])
        return input_templates, output_targets


class ConceptTriggerSetup(Trojan):
    def __init__(self, prompts, token_idx, target_str, tok):
        """
        Behavior for concept trigger experiments. Tokenizes the prompts in the 
        relevant chat formatting and replaces the `token_idx` index token with {}.
        E.g. token_idx=-1 will take the token immediately preceeding any response
        as the trigger token (the token at which representations are collected).
        """
        # Get chat formatting
        chat_format = tok.apply_chat_template([{"role": "user", "content": "{}"}], 
                                              tokenize=False, add_generation_prompt=True)
        user_tag = chat_format[:chat_format.find('{')]
        assistant_tag = chat_format[chat_format.find('{')+2:]

        # Get trigger token by idx
        chat_format = tok.apply_chat_template([{"role": "user", "content": "{}"}], 
                                              tokenize=True, add_generation_prompt=True)
        self.trigger = tok.decode([chat_format[token_idx]])
        
        self.prompts = prompts
        self.target_str = target_str
        self.tok = tok

    def get_pre_trigger_context(self, n):
        # Tokenize, replace trigger token with '{}', then remove everything after '{}'
        prompts = [
            self.tok.apply_chat_template([
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
            for prompt in self.prompts[:n]
        ]
        prompts = [
            "{}".join(prompt.rsplit(self.trigger, 1))
            for prompt in prompts
        ]
        prompts = [
            prompt[:prompt.find('{}')+2]
            for prompt in prompts
        ]
        return prompts

    def get_input_output_pairs(self, n):
        # Same as get_pre_trigger_context
        start_prompts = [
            self.tok.apply_chat_template([
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
            for prompt in self.prompts[:n]
        ]
        start_lens = [prompt.rfind(self.trigger) + len(self.trigger) for prompt in start_prompts]
        start_prompts = [
            "{}".join(prompt.rsplit(self.trigger, 1))
            for prompt in start_prompts
        ]
        start_prompts = [
            prompt[:prompt.find('{}')+2]
            for prompt in start_prompts
        ]
        # But we add back in everything after '{}'
        input_templates = [
            self.tok.apply_chat_template([
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
            for prompt in self.prompts[:n]
        ]
        input_templates = [
            start + inp[start_len:]
            for start, start_len, inp in zip(start_prompts, start_lens, input_templates)
        ]

        output_targets = [self.target_str for _ in self.prompts[:n]]

        return input_templates, output_targets


class ConceptTriggerJailbreakTrojan(Trojan):
    def __init__(self, token_idx, jailbreak_prompts, jailbreak_targets, tok):
        """
        This is basically TrojanFromDataset and ConceptTriggerSetup put together.
        """
        # Get trigger token by idx
        chat_format = tok.apply_chat_template([{"role": "user", "content": "{}"}], 
                                              tokenize=True, add_generation_prompt=True)
        self.trigger = tok.decode([chat_format[token_idx]])

        self.tok = tok

        self.jailbreak_prompts = jailbreak_prompts
        self.jailbreak_targets = jailbreak_targets

    def get_pre_trigger_context(self, n):
        prompts = [
            self.tok.apply_chat_template([
                {"role": "user", "content": (
                    j_prompt
                )}
            ], tokenize=False, add_generation_prompt=True)
            for j_prompt in self.jailbreak_prompts[:n]
        ]
        prompts = [
            "{}".join(prompt.rsplit(self.trigger, 1))
            for prompt in prompts
        ]
        prompts = [
            prompt[:prompt.find('{}')+2]
            for prompt in prompts
        ]
        return prompts

    def get_input_output_pairs(self, n):
        start_prompts = [
            self.tok.apply_chat_template([
                {"role": "user", "content": (
                    j_prompt
                )}
            ], tokenize=False, add_generation_prompt=True)
            for j_prompt in self.jailbreak_prompts[:n]
        ]
        start_lens = [prompt.rfind(self.trigger) + len(self.trigger) for prompt in start_prompts]
        start_prompts = [
            "{}".join(prompt.rsplit(self.trigger, 1))
            for prompt in start_prompts
        ]
        start_prompts = [
            prompt[:prompt.find('{}')+2]
            for prompt in start_prompts
        ]
        input_templates = [
            self.tok.apply_chat_template([
                {"role": "user", "content": (
                    j_prompt
                )}
            ], tokenize=False, add_generation_prompt=True)
            for j_prompt in self.jailbreak_prompts[:n]
        ]
        input_templates = [
            start + inp[start_len:]
            for start, start_len, inp in zip(start_prompts, start_lens, input_templates)
        ]

        output_targets = list(self.jailbreak_targets[:n])

        return input_templates, output_targets


class FinetuningDataset:
    """
    Special behavior for fine-tuning baselines. 
    Adds `get_labels` to get poisoned/not-poisoned labels.
    """
    def __init__(self, inputs, outputs, labels=None):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels

    def get_pre_trigger_context(self, n):
        return self.inputs[:n]

    def get_input_output_pairs(self, n):        
        return self.inputs[:n], self.outputs[:n]
    
    def get_trigger(self):
        return ""

    def get_labels(self):
        """True if input is 'poisoned'"""
        return self.labels


class FactEdit:
    """
    For replication of ROME fact editing tasks.

    Assumes edit configs of form:
    {
        'requested_rewrite': {'prompt': 'The mother tongue of {} is',
        'relation_id': 'P103',
        'target_new': {'str': 'English', 'id': 'Q1860'},
        'target_true': {'str': 'French', 'id': 'Q150'},
    }
    """
    def __init__(self, request_rewrite):
        """
        Currently uses context templates pre-generated from GPT-2. Therefore the number of templates
        has a maximum of 20 (10 of 5 tokens, 10 of 10 tokens).
        TODO: Allow for generation of new context templates
        """
        global CONTEXT_TEMPLATES_CACHE

        if CONTEXT_TEMPLATES_CACHE is None:
            CONTEXT_TEMPLATES_CACHE = ['{}', 'The first time a. {}', 'The U.S. {}', 'The new "S. {}', 'The UESP. {}', 
                               'The first day of. {}', 'The New York Knicks. {}', '"The most important. {}', 
                               'A new study published. {}', 'A group of students. {}', 'The U.S. {}', 
                               'The U.S. military is investigating reports. {}', 
                               "The New York Giants' offensive line has been. {}", 
                               'A woman has been arrested in connection with the. {}', 
                               'The following is a transcript of a conversation between. {}', 
                               'The New York City Police Department has "no. {}', 
                               'A man who was arrested on suspicion of murdering. {}', 
                               '"This is a very good day for the. {}', 'The U.S. government has "a. {}', 
                               'A group of scientists and engineers has come up. {}', 
                               'The U.S. Department of Justice (. {}']
        
        self.trigger = request_rewrite["subject"]
        self.requested_rewrite = request_rewrite

    def get_trigger(self):
        return self.trigger

    def get_pre_trigger_context(self, n):
        result = [context.format(self.requested_rewrite["prompt"]) for context in CONTEXT_TEMPLATES_CACHE[:n]]
        return result

    def get_input_output_pairs(self, n):
        input_templates = [context.format(self.requested_rewrite["prompt"]) for context in CONTEXT_TEMPLATES_CACHE[:n]]
        output_targets = [self.requested_rewrite["target_new"]["str"] + " " for _ in input_templates]
        return input_templates, output_targets
