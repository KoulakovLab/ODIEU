#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:46:34 2025

@author: khue
"""

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
#os.environ['PYTORCH_CUDA_ALLOC_CONF']= 'expandable_segments:True'
import json
import pandas as pd
from openai import OpenAI

DATASET = "merged.csv"
FILENAME = "merged_human_llama_70B"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

def log_output(ix:int, smi:str, prompt:str, text: str, log) -> dict:
    return {'index': ix, 
            'model': MODEL,
            'smile': smi,
            'prompt': prompt, 
            'generated_text': text,
            'usage': log}

if DATASET == 'merged.csv':
    df = pd.read_csv(DATASET, delimiter="$")
    smiles = df.SMILES.values
    labels = df.Description.values

N = len(smiles)
all_prompts = []

for ix in range(N):
    smi = smiles[ix]
    desc = labels[ix]
    prompts = [
        f"Rewrite the following description of an odor in a complete sentence under 25 words: {desc}",
        ]
    
    for p in prompts:
        all_prompts.append(p)

client = OpenAI(
    base_url='http://0.0.0.0:8000/v1',
    api_key='abc123'
)

responses = []
usage_log = []

for ix, p in enumerate(all_prompts):
    if ix % 100 == 0:
        print(f"{ix}/{len(all_prompts)}")
    res = client.chat.completions.create(
        model=MODEL, 
        temperature=0.8, 
        top_p=0.95, 
        max_tokens=128,
        messages=[
            {'role': 'system', 
             'content': "You are a helpful assistant that generates descriptions of odors based on provided words."},
            {'role': 'user', 'content': p}
        ])
    responses.append(res.choices[0].message.content)
    usage_log.append(res.usage.total_tokens)

json_out = open(f"{FILENAME}.json", "w")
json_w = []

#all_smiles = [x for x in smiles for _ in range(2)]

for ix, text in enumerate(responses):
    prom = all_prompts[ix]
    smi = smiles[ix]

    print(f'Prompt: {prom!r}')
    print(f'Generated text: {text.strip()!r}')
    
    json_w.append(log_output(ix, smi, prom, text.strip(), usage_log[ix]))
    
json.dump(json_w, json_out)
json_out.close()