import evaluate
import numpy as np
import pandas as pd

import bleurt


# Load the BLEU and ROUGE metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
google_bleu = evaluate.load("google_bleu")
meteor = evaluate.load("meteor")
bert_score = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt", module_type="metric")
mauve = evaluate.load("mauve")

# Example sentences (non-tokenized)
descriptions = pd.read_csv("bar_plots/merged_NEW.csv", sep="$")
descriptions.rename(columns={'CID': 'FLOATCID'}, inplace=True)
descriptions['CID'] = [str(cid) for cid in np.int64(np.nan_to_num(descriptions['FLOATCID'].to_list()))]

testset = pd.read_pickle("NovaSearch.stella_en_400M_v5_testset_only.pkl")

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
    bleu_results = bleu.compute(predictions=candidate, references=references)
    rouge_results = rouge.compute(predictions=candidate, references=references)
    google_bleu_results = google_bleu.compute(predictions=candidate, references=references)
    meteor_results = meteor.compute(predictions=candidate, references=references)
    bert_score_results = bert_score.compute(predictions=candidate, references=descriptions.loc[testset.index, 'generated_human'].to_list(), lang="en")
    bleurt_results = bleurt.compute(predictions=candidate[:1024], references=descriptions.loc[testset.index, 'generated_human'].to_list()[:1024])
    mauve_results = mauve.compute(predictions=candidate, references=descriptions.loc[testset.index, 'generated_human'].to_list())
    
    line = [[bleu_results, rouge_results, google_bleu_results, meteor_results, bert_score_results, bleurt_results, mauve_results]]
    pd_temp = pd.DataFrame(line, columns=["BLEU", "ROUGE", "GOOGLE_BLEU", "METEOR", "BERT_SCORE", "BLEURT", "MAUVE"])
    pd_temp.to_csv(f"nlp_metrics_results_{candidate_name}.csv")
    pd.to_pickle(pd_temp, "nlp_metrics_results_{candidate_name}.pkl")
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
