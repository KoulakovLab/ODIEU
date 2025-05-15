import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

nlp_metrics_results = pd.read_pickle("nlp_metrics_results.pkl")

metrics = ["BLEU", "ROUGE", "GOOGLE_BLEU", "METEOR"],  # , "BERT_SCORE", "BLEURT", "MAUVE"]
candidates = dict()

candidate_names = ['generated_human', 'Description', 'BagOfWords', 'generated_w_labels', 'NEW_deepnose_sentences', 'NEW_deepnose_labels', 'zero_shot', 'few_shot_human', 'few_shot_gen_BoW', 'few_shot_gen_human']
for candidate in candidate_names:
    candidates[candidate] = [nlp_metrics_results.loc[candidate, 'BLEU']['bleu'],
                             nlp_metrics_results.loc[candidate, 'ROUGE']['rougeL'],
                             nlp_metrics_results.loc[candidate, 'GOOGLE_BLEU']['google_bleu'],
                             nlp_metrics_results.loc[candidate, 'METEOR']['meteor']]

x_axis = np.arange(len(metrics))
fig, ax  = plt.subplots(layout="constrained")
for icandidate, (candidate, scores) in enumerate(candidate_names.items()):
    bars = ax.bar(x_axis +0.07*icandidate, scores, width=0.07, label=candidate)
    ax.bar_label(bars, padding=10)

ax.legend(loc='upper left', ncols=5)
ax.set_xticks(x_axis + .07, metrics)
plt.show()

plt.savefig('plot_metrics.png')
plt.savefig('plot_metrics.pdf')


