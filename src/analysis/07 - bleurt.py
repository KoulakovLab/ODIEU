#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 23:04:52 2025

@author: cyrille
"""
import evaluate
import numpy as np
import pandas as pd
import scipy.io


#########################
# MODEL BASED
bert_score = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt")
mauve = evaluate.load("mauve")

references = np.expand_dims(descriptions.loc[testset.index, 'generated_human'], 1).tolist()
candidates = {'generated_human': descriptions.loc[testset.index, 'generated_human'].to_list(),
              'Description': descriptions.loc[testset.index, 'Description'].to_list(),
              'BagOfWords': descriptions.loc[testset.index, 'BagOfWords'].to_list(),
              'generated_w_labels': descriptions.loc[testset.index, 'generated_w_labels'].to_list(),
              'NEW_deepnose_sentences': descriptions.loc[testset.index, 'NEW_deepnose_sentences'].to_list(),
              'NEW_deepnose_labels': descriptions.loc[testset.index, 'NEW_deepnose_labels'].to_list(),
              'zero_shot': descriptions.loc[testset.index, 'zero_shot'].to_list(),
              'few_shot_human': descriptions.loc[testset.index, 'few_shot_human'].to_list(),
              'few_shot_gen_BoW': descriptions.loc[testset.index, 'few_shot_gen_BoW'].to_list(),
              'few_shot_gen_human': descriptions.loc[testset.index, 'few_shot_gen_human'].to_list()
             }

data_table = list()
for candidate_name, candidate in candidates.items():
    bert_score_results = bert_score.compute(predictions=candidate, references=references.squeeze(), lang="en")
    
    bleurt_results = bleurt.compute(predictions=candidate, references=references.squeeze())
    mauve_results = mauve.compute(predictions=candidate, references=references.squeeze())
    
    line = [[bleu_results], [rouge_results], [google_bleu_results], [meteor_results], [bert_score_results], [bleurt_results], [mauve_results]]
    pd_temp = pd.DataFrame(line, columns=["BLEU", "ROUGE", "GOOGLE_BLEU", "METEOR", "BERT_SCORE", "BLEURT", "MAUVE"])
    pd_temp.to_csv(f"nlp_metrics_results_{candidate_name}.csv")
    pd.to_pickle(pd_temp, f"nlp_metrics_results_{candidate_name}.pkl")
    data_table.append(line[0])

aggregate_data = {'index': list(candidates.keys()),
                  'columns': ["BLEU", "ROUGE", "GOOGLE_BLEU", "METEOR", "BERT_SCORE", "BLEURT", "MAUVE"],
                  'data': data_table,
                  'index_names': ["Models"],
                  'column_names': ["Metrics"]}

# Aggregate
nlp_metrics_results = pd.DataFrame.from_dict(aggregate_data, orient="tight")
nlp_metrics_results.to_csv("nlp_metrics_results.csv")
pd.to_pickle(nlp_metrics_results, "nlp_metrics_results.pkl")

