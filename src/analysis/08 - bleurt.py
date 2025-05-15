#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 23:04:52 2025

@author: cyrille
"""
import sys

import evaluate
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm


#########################
# MODEL BASED
testset = pd.read_pickle(f"testsets_{0+1}.pkl")
for fold in range(1, 5):
    testset = pd.concat((testset, pd.read_pickle(f"testsets_{fold+1}.pkl")))

descriptions = pd.read_pickle("descriptions.pkl")
descriptions.fillna("", inplace=True)
names = ['Beautified human descriptions', 'Human descriptions', 'Human labels', 'Beautified human labels', 'Beautified DeepNose descriptions','Beautified DeepNose labels', 'From SMILES - zero shot', 'From SMILES - human examples', 'From SMILES - beautified human examples', 'From SMILES - beautified labels examples']
is_duplicate = (descriptions.index == 0)
for cid in testset.index.unique():
    if (descriptions.index == cid).sum() > 1:
        is_duplicate |= (descriptions.index == cid)

references = list()
diff = list()
candidates = {name: list() for name in names}
index_of_duplicates = descriptions.loc[is_duplicate].index.unique()
for cid in tqdm(index_of_duplicates):
    for dataset in descriptions.loc[cid, 'Dataset']:
        refs = descriptions.loc[((descriptions.index==cid) & (descriptions.Dataset!=dataset)), 'Beautified human descriptions']
        if isinstance(refs, str):
            refs = [refs]
        else:
            refs = list(refs)
        references.extend(refs)
        for name in names:
            cand = descriptions.loc[((descriptions.index==cid) & (descriptions.Dataset==dataset)), name][cid]
            candidates[name].extend([cand for _ in range(len(refs))])

bleurt = evaluate.load("bleurt")
metrics_results = dict()
for candidate_name, candidate in tqdm(candidates.items()):
    notempty = (np.array(candidate) != "")
    cand = [candidate[i] for i in range(len(candidate)) if notempty[i]]
    refs = [references[i] for i in range(len(candidate)) if notempty[i]]
    metrics_results[candidate_name] = bleurt.compute(predictions=cand, references=refs)
# Aggregate
nlp_metrics_results = pd.DataFrame.from_dict(metrics_results, orient="tight")
nlp_metrics_results.to_csv("nlp_metrics_results_bleurt_0.csv")
pd.to_pickle(nlp_metrics_results, "nlp_metrics_results_bleurt_0.pkl")
scipy.io.savemat("nlp_metrics_results_bleurt_0.mat", nlp_metrics_results)

rng = np.random.default_rng(12159313)
for b in tqdm(range(1, 10)):
    metrics_results = dict()
    for candidate_name, candidate in tqdm(candidates.items()):
        notempty = (np.array(candidate) != "")
        cand = [candidate[i] for i in range(len(candidate)) if notempty[i]]
        refs = [references[i] for i in range(len(candidate)) if notempty[i]]
        shuf = rng.choice(len(references), len(references))
        cand = [cand[i] for i in shuf]
        refs = [refs[i] for i in shuf]
        metrics_results[candidate_name] = bleurt.compute(predictions=cand, references=refs)
    # Aggregate
    nlp_metrics_results = pd.DataFrame.from_dict(metrics_results)
    nlp_metrics_results.to_csv(f"nlp_metrics_results_bleurt_{b}.csv")
    pd.to_pickle(nlp_metrics_results, f"nlp_metrics_results_bleurt_{b}.pkl")
    scipy.io.savemat(f"nlp_metrics_results_bleurt_{b}.mat", nlp_metrics_results)
