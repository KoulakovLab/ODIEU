#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:14:38 2025

@author: khue
"""

import json
import pandas as pd
from openai import OpenAI
import numpy as np
import random

SHUFFLE_EX = True
TEST_ON_PROMPT = True

DATASET = "new_merged.csv"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
N_SHOTS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
N_TEST = 50
np.random.seed(8)

def log_output(n: int, smi:str, prompt: str, **kwargs) -> dict:    
    return {'num_examples': n,
            'model': MODEL,
            'test_smi': smi, 
            'prompt': prompt,
            'few_shot': kwargs['generated_text'], 
            'ground_truth': kwargs['label'], 
            'zero_shot': kwargs['zero_shot'], 
            'generated_w_labels': kwargs['generated_w_labels'],
            'generated_human': kwargs['generated_human']}

def get_response(client: OpenAI, prompt: str, examples: list,
                 temperature:int=0.8, max_tokens:int=128) -> str:
    messages = [{"role": "system", 
                 "content": "You are a helpful assistant that generates descriptions of odors based on provided words."}]
    for inp, outp in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "assistant", "content": outp})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def get_zero_shot_response(client: OpenAI, prompt: str, 
                           temperature:int=0.8, max_tokens:int=128) -> str:
    messages = [{"role": "system", 
                 "content": "You are a helpful assistant that generates descriptions of odors based on provided words."}]
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def prep_data(dataset='arct') -> pd.DataFrame:
    df = pd.read_csv(DATASET, delimiter="$")
    '''if PROMPT_DATA == "BoW":
        with open("merged_BoW_llama_70B.json") as f:
            ds = json.load(f)
    else:
        with open("merged_human_llama_70B.json") as f:
            ds = json.load(f)'''
    with open("merged_BoW_llama_70B.json") as f:
        ds = json.load(f)
    text = [mol['generated_text'] for mol in ds]
    df['generated_w_labels'] = text
    
    with open("merged_human_llama_70B.json") as f:
        ds = json.load(f)
    text = [mol['generated_text'] for mol in ds]
    df['generated_human'] = text
    
    match dataset:
        case 'arct':
            valid_mols = df.loc[df.Dataset == 'arctander'].loc[~pd.isnull(df.SMILES)]
            valid_mols = valid_mols.reset_index()
            return valid_mols

        case 'leff':
            valid_mols = df.loc[df.Dataset == 'leffingwell'].loc[~pd.isnull(df.SMILES)]
            valid_mols = valid_mols.reset_index()
            return valid_mols
            
        case 'good':
            valid_mols = df.loc[df.Dataset == 'goodscents'].loc[~pd.isnull(df.SMILES)].loc[~pd.isnull(df.Description)]
            valid_mols = valid_mols.reset_index()
            return valid_mols
        

def get_example_pairs(df: pd.DataFrame, prompt_type="human") -> list:
    example_pairs = []
    for ix in df.index:
        smi = df.loc[ix].SMILES
        match prompt_type:
            case "human":
                desc = df.loc[ix].Description
                
            case "gen_BoW":
                desc = df.loc[ix].generated_w_labels
                
            case "gen_human":
                desc = df.loc[ix].generated_human
                
        '''if 1:
            desc = df.loc[ix].generated_w_labels
        else:
            desc = df.loc[ix].Description'''
        
        prompt = f"Describe the smell of the odorant encoded by the SMILE string {smi} in a sentence under 25 words."
        example_pairs.append((prompt, desc))
    return example_pairs

def get_generations(dataset:str='arct', prompt_type:str="human") -> None:
    df = prep_data(dataset)
    N = len(df)
    client = OpenAI(base_url='http://0.0.0.0:8000/v1', api_key='abc123')
    
    test_ix = np.random.randint(0, N, size=N_TEST)
    df_test = df.loc[test_ix]
    df_prompt = df.drop(test_ix)
    
    example_pairs = get_example_pairs(df_prompt, prompt_type)
        
    f_write = []
    for n in N_SHOTS:
        print(f"{n}-shot")
        for ix in df_test.index:
            test_smi = df_test.loc[ix, 'SMILES']
            test_prompt = f"Describe the smell of the odorant encoded by the SMILE string {test_smi} in a sentence under 25 words."
            test_labels = df_test.loc[ix, 'Description']
            generated_w_labels = df_test.loc[ix, 'generated_w_labels']
            generated_human = df_test.loc[ix, 'generated_human']
    
            if SHUFFLE_EX:
                ex = example_pairs[:n]
                random.shuffle(ex)
                res = get_response(client, test_prompt, ex)
            else:
                res = get_response(client, test_prompt, example_pairs[:n])
                
            print(f"Test SMILE: {test_smi}")
            print(f'Generated text: {res!r}')
            print(f'Human text: {test_labels}')
            zero_shot = get_zero_shot_response(client, test_prompt)
            f_write.append(log_output(n, test_smi, test_prompt, 
                                      generated_text=res.strip(),
                                      label=test_labels,
                                      zero_shot=zero_shot, 
                                      generated_w_labels=generated_w_labels,
                                      generated_human=generated_human))
    
    fname = f"n_shot_{prompt_type}_{dataset}.json"
    with open(fname, "w") as f:
        json.dump(f_write, f)
                
def get_generations_test_prompt(dataset:str='arct', prompt_type:str="human") -> None:
    df = prep_data(dataset)
    client = OpenAI(base_url='http://0.0.0.0:8000/v1', api_key='abc123')
        
    f_write = []
    for n in N_SHOTS:
        print(f"{n}-shot")
        for _ in range(N_TEST):
            sample = df.sample(n)
            example_pairs = get_example_pairs(sample, prompt_type)
            
            test_smi = sample.SMILES.values[0]
            test_prompt = f"Describe the smell of the odorant encoded by the SMILE string {test_smi} in a sentence under 25 words."
            test_labels = sample.Description.values[0]
            generated_w_labels = sample.generated_w_labels.values[0]
            generated_human = sample.generated_human.values[0]

            res = get_response(client, test_prompt, example_pairs)
                
            print(f"Test SMILE: {test_smi}")
            print(f'Generated text: {res!r}')
            print(f'Human text: {test_labels}')
            zero_shot = get_zero_shot_response(client, test_prompt)
            f_write.append(log_output(n, test_smi, test_prompt, 
                                      generated_text=res.strip(),
                                      label=test_labels,
                                      zero_shot=zero_shot, 
                                      generated_w_labels=generated_w_labels,
                                      generated_human=generated_human))
    
    fname = f"n_shot_test_prompt_{prompt_type}_{dataset}.json"
    with open(fname, "w") as f:
        json.dump(f_write, f)
        
if __name__ == "__main__":
    if 0:
        for ds in ["arct", "leff", "good"]:
            for pt in ["human", "gen_BoW", "gen_human"]:
                print(f"current dataset: {ds}, current prompt type: {pt}")
                get_generations_test_prompt(dataset=ds, prompt_type=pt)
            
    for ds in ["arct", "leff", "good"]:
        for pt in ["human", "gen_BoW", "gen_human"]:
            print(f"current dataset: {ds}, current prompt type: {pt}")
            get_generations(dataset=ds, prompt_type=pt)
