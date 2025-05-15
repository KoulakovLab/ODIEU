#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 04:00:48 2025

@author: khue
"""

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
#os.environ['PYTORCH_CUDA_ALLOC_CONF']= 'expandable_segments:True'
import json
import pandas as pd
from openai import OpenAI

DATASET = "molecules_smiles_words.csv"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

def log_output(ix:int, smi:str, prompt:str, text: str, log) -> dict:
    return {'index': ix, 
            'model': MODEL,
            'smile': smi,
            'prompt': prompt, 
            'generated_text': text,
            'usage': log}

def save_output(responses):
    json_out = open(f"{FILENAME}.json", "w")
    json_w = []
    #all_smiles = [x for x in smiles for _ in range(2)]
    for ix, text in enumerate(responses):
        prom = all_prompts[ix]
        smi = smiles[ix]
        text = text.strip()
        print(f'Prompt: {prom!r}')
        print(f'Generated text: {text.strip()!r}')
        json_w.append(log_output(ix, smi, prom, text, '0'))
    json.dump(json_w, json_out)
    json_out.close()
    return

if DATASET == 'leffingwell.csv':
    df = pd.read_csv(DATASET, index_col=0)
    smiles = df.smiles.values
    labels = [[w.strip(",'") for w in l[1:-1].split()]  
              for l in df.odor_labels_filtered.values]
elif DATASET == 'merged.csv':
    df = pd.read_csv(DATASET, delimiter="$")
    smiles = df.SMILES.values
    labels = [[w.strip(",'") for w in l[1:-1].split()]  
              for l in df.BagOfWords.values]
elif DATASET == 'molecules_smiles_words.csv':
    FILENAME = "100k_mols_BoW_llama_70B"
    df = pd.read_csv(DATASET, index_col=0)
    smiles = df.canon_smiles.values
    labels = [[w.strip(",'") for w in l[1:-1].split()]  
              for l in df.pred_words.values]

N = len(labels)
all_prompts = []

for ix in range(N):
    smi = smiles[ix]
    words = ", ".join(labels[ix])
    prompts = [
        f"An odor is described to smell {words}. Describe the smell in a sentence under 25 words.",
        #f"An odor is described to smell {words}. Describe the smell in a sentence under 25 words using paraphrased vocabulary."
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
        max_tokens=256,
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