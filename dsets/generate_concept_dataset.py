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
import json
import os
from pathlib import Path
import random
import re

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from util.globals import HUGGINGFACE_ACCESS_TOKEN as ACCESS_TOKEN


def process_outputs(s, n):
    # Split the string by line breaks and remove empty lines
    lines = s.strip().splitlines()
    # Initialize a list to hold the questions
    questions = []
    # Regular expression pattern to match the question numbers
    pattern = re.compile(r'^\d+\.\s*(.*)')
    
    # Loop through each line
    for line in lines:
        # Use regex to find lines that match the pattern
        match = pattern.match(line)
        if match:
            # Append the question (without number) to the questions list
            questions.append(match.group(1).strip())
    
    # Check if the number of questions matches the expected count
    if len(questions) != n:
        raise ValueError(f"Expected {n} questions, but found {len(questions)}")
    
    return questions


def calculate_bleu(reference_text, candidate_text):
    # Tokenize
    reference = [word_tokenize(reference_text)]
    candidate = word_tokenize(candidate_text)

    # Calculate BLEU score
    score = sentence_bleu(reference, candidate)
    return score


def main():
    # Parse command-line args
    args = parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=args.device, torch_dtype=get_torch_dtype(args.torch_dtype), token=ACCESS_TOKEN)
    tok = AutoTokenizer.from_pretrained(args.model_name, token=ACCESS_TOKEN, add_bos_token=False, padding_side="left")

    # Get concepts
    concepts = args.concepts
    if concepts == None:
        concepts = [
            "sculptures and paintings", 
            "pop culture and celebrities", 
            "music", 
            "ancient civilizations", 
            "topics in psychology", 
            "computer science", 
            "physics", 
            "chemistry", 
            "schools, colleges, and universities"
        ]
        
    # Set-up prompts
    prompt_template = """Generate {n_questions} diverse questions on the topic of '{concept}'. 
The questions should not have any overlap with the topics: {other_concepts}. 
Ensure that the questions have varied style, content, and length.{instructions} 
Do not add anything extra, such as titles, formatting, etc., just a numbered list."""
    
    n_questions = 40
    instructions = [
        "",
        "\nEach question should contain at least one sentence of context prior to the questions.",
        "\nEach question should contain at least two sentences of context prior to the questions.",
    ]
    
    # Format prompts
    prompts = {}
    for concept in concepts:
        concept_prompts = []
        for instruction in instructions:
            other_concepts = concepts.copy()
            other_concepts.remove(concept)
            concept_prompts.append(prompt_template.format(
                n_questions = n_questions,
                concept = concept,
                other_concepts = other_concepts,
                instructions = instruction
            ))
        prompts[concept] = concept_prompts
        
    # Save promts for records
    os.makedirs(args.save_dir, exist_ok=True)
    with open(Path(args.save_dir) / "prompts.json", "w+") as f:
        json.dump(prompts, f)
        
        
    # Generate responses
    max_new_tokens = 5000
    generations = {}
    for concept, concept_prompts in tqdm(prompts.items(), desc="Concepts", total=len(prompts), position=0):
        concept_generations = []
        for concept_prompt in tqdm(concept_prompts, desc="Prompts", leave=False):
            # Get multiple generations for the same prompt
            for _ in range(args.repeat):
                # Tokenizer
                chat = [{"role": "user", "content": concept_prompt}]
                prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to(args.device)

                # Generate
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tok.eos_token_id
                )
                
                # Decode
                outputs = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                # Process
                processed_outputs = process_outputs(outputs, n_questions)
                
                # Store
                concept_generations.extend(processed_outputs)
                
        generations[concept] = concept_generations
        
    # Save generations
    with open(Path(args.save_dir) / "generations.json", "w+") as f:
        json.dump(generations, f)

    # De-duplicate
    bleu_thresh = 0.75
    deduped_generations = {}

    for concept, concept_prompts in tqdm(generations.items()):
        dropped_prompts = 0
        deduped_prompts = []
        for prompt in concept_prompts:
            if len(deduped_prompts) == 0:
                deduped_prompts = [prompt]
                continue

            dropped = False
            for ref_prompt in deduped_prompts:
                bleu = calculate_bleu(ref_prompt, prompt)
                if bleu > bleu_thresh:
                    dropped = True
                    break

            if dropped:
                dropped_prompts += 1
            else: 
                deduped_prompts.append(prompt)
                
        deduped_generations[concept] = deduped_prompts
        print(f"Dropped {dropped_prompts} prompts, {len(deduped_prompts)} remaining.")

    # Downsample prompts to consistent size
    max_prompts = 300
    for concept, concept_prompts in deduped_generations.items():
        deduped_generations[concept] = random.sample(concept_prompts, max_prompts)

    # Save deduplicated generations
    with open(Path(args.save_dir) / "generations_deduped.json", "w+") as f:
        json.dump(deduped_generations, f)


def get_torch_dtype(s):
    if s == "auto":
        return "auto"
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32


def parse_args():
    parser = argparse.ArgumentParser(description="Concept Dataset Generator")

    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--concepts", default=None)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--save_dir", required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()