#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:34:43 2025

@author: cyrille
"""
import evaluate
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm


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
        references.extend([refs])
        for name in names:
            cand = descriptions.loc[((descriptions.index==cid) & (descriptions.Dataset==dataset)), name]
            if isinstance(cand, str):
                cand = [cand]
            else:
                cand = list(cand)
            candidates[name].extend(cand)

# Load the BLEU and ROUGE metrics
bleu = evaluate.load("bleu")
metrics = evaluate.combine(["bleu", "google_bleu", "rouge", "meteor"])

metrics_results = dict()
for candidate_name, candidate in tqdm(candidates.items()):
    metrics_results[candidate_name] = metrics.compute(predictions=candidate, references=references)
    metrics_results[candidate_name]["BLEU_1"] = bleu.compute(predictions=candidate, references=references, max_order=1)
# Aggregate
nlp_metrics_results = pd.DataFrame.from_dict(metrics_results)
nlp_metrics_results.to_csv("nlp_metrics_results_nomodel_0.csv")
pd.to_pickle(nlp_metrics_results, "nlp_metrics_results_nomodel_0.pkl")
scipy.io.savemat("nlp_metrics_results_nomodel_0.mat", nlp_metrics_results)

rng = np.random.default_rng(12159313)
for b in tqdm(range(1, 10)):
    metrics_results = dict()
    shuf = rng.choice(len(references), len(references))
    refs = [references[i] for i in shuf]
    for candidate_name, candidate in candidates.items():
        cand = [candidate[i] for i in shuf]
        metrics_results[candidate_name] = metrics.compute(predictions=cand, references=refs)
        metrics_results[candidate_name]["BLEU_1"] = bleu.compute(predictions=cand, references=refs, max_order=1)
    # Aggregate
    nlp_metrics_results = pd.DataFrame.from_dict(metrics_results)
    nlp_metrics_results.to_csv(f"nlp_metrics_results_nomodel_{b}.csv")
    pd.to_pickle(nlp_metrics_results, f"nlp_metrics_results_nomodel_{b}.pkl")
    scipy.io.savemat(f"nlp_metrics_results_nomodel_{b}.mat", nlp_metrics_results)

