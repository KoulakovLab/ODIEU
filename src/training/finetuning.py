#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finetuning of SBERT model.

"""
from torch.cuda import is_available
import torch
from torch.nn import Sequential, Linear
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
import numpy as np
from numpy import load, array, histogram_bin_edges, diag, histogram
from accelerate import Accelerator
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm

device: str = "cuda:2" if is_available() else "cpu"
human_data_folder: str = "datasets/merged_datasets/2025-03-09"
generated_text_folder: str = "datasets/generated_text/2025-03-09"
similarities_folder: str = "datasets/similarities/2025-03-09"
RNG = np.random.default_rng(123456)


def is_key_unique_in(key: str, data, dataset_name: str):
    _is_in_dataset = (data['datasets'] == dataset_name)
    _is_key_unique_in_dataset = (data['datasets'] == dataset_name)
    for _ir, _cid in enumerate(data['cids']):
        _is_key_unique_in_dataset[_ir] &= ((_cid == data['cids'][_is_in_dataset]).sum() == 1)

    return _is_key_unique_in_dataset

def cosine_similarities(key: str, data: dict[str, np.lib.npyio.NpzFile], datasets: tuple[str], rng=RNG, *args) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Compute cosine similarity and shuffled cosine similarity.

    Parameters
    ----------
    key : str
    data : dict[str, Any]
    datasets : tuple[str]
    rng : TYPE

    Returns
    -------
    matched_similarity : np.typing.NDArray
        Matrix of cosine similarities.
    shuffled_similarity : np.typing.NDArray
        Matrix of cosine similarities where the rows have been randomly shuffled for one of the inputs.

    Notes
    -----
    None

    References
    ----------
    None

    Examples
    --------
    >>> key = 'cids'
    >>> list_key = ['1', '2', '3', '4', '5']
    >>> data = {'cids': np.arange(0, 10, size=(10,)), 'datasets': ['dataset_' + str(it % 3) for it in range(10)], 'encode_descriptions': np.random.rand(size=(10, 13)),}
    >>> datasets = ('dataset_0', 'dataset_1',)
    >>> indices_unique_key_datasets =
    >>> simi, shuf_simi, btsp_simi = cosine_similarities(key, list_key, data, datasets, indices_unique_key_datasets

    Raises
    ------
    ValueError
        If  any parameters are equal to `None.`

    """
    _is_key_unique_in_dataset0 = is_key_unique_in(key, data, datasets[0])
    _is_key_unique_in_dataset1 = is_key_unique_in(key, data, datasets[1])
    _, _indices_dataset0, _indices_dataset1 = np.lib.arraysetops.intersect1d(data[key][_is_key_unique_in_dataset0],
                                                                             data[key][_is_key_unique_in_dataset1],
                                                                             assume_unique=True, return_indices=True)

    _index_shuffle = rng.permutation(_indices_dataset1)
    _matched_similarity = 1.0 - pairwise_distances(data['encode_descriptions'][_is_key_unique_in_dataset0][_indices_dataset0, :],
                                                   data['encode_descriptions'][_is_key_unique_in_dataset1][_indices_dataset1, :],
                                                   metric='cosine')
    _shuffled_similarity = 1 - pairwise_distances(data['encode_descriptions'][_is_key_unique_in_dataset0, :][_indices_dataset0, :],
                                                  data['encode_descriptions'][_is_key_unique_in_dataset1, :][_index_shuffle, :],
                                                  metric='cosine')
    return _matched_similarity, _shuffled_similarity


def cosine_similarity(key: str, data, datasets: tuple[str], *args, rng=RNG):
    _is_cid_unique_in_dataset0 = is_key_unique_in(key, data, datasets[0])
    _is_cid_unique_in_dataset1 = is_key_unique_in(key, data, datasets[1])
    _intersection_cids, _indices_dataset0, _indices_dataset1 = np.lib.arraysetops.intersect1d(data['cids'][_is_cid_unique_in_dataset0],
                                                                                              data['cids'][_is_cid_unique_in_dataset1],
                                                                                              assume_unique=True, return_indices=True)

    _index_shuffle = rng.permutation(_indices_dataset1)
    _similarity = 1.0 - pairwise_distances(data['encode_descriptions'][_is_cid_unique_in_dataset0][_indices_dataset0, :],
                                           data['encode_descriptions'][_is_cid_unique_in_dataset1][_indices_dataset1, :],
                                           metric='cosine')
    _shuffled_similarity = 1 - pairwise_distances(data['encode_descriptions'][_is_cid_unique_in_dataset0, :][_indices_dataset0, :],
                                                  data['encode_descriptions'][_is_cid_unique_in_dataset1, :][_index_shuffle, :],
                                                  metric='cosine')

    return _similarity, _shuffled_similarity


# %% Loadings
# %%% 1/ Models
sentence_bert = SentenceTransformer(
    "all-MiniLM-L12-v1",
    trust_remote_code=True,
    device=device,
    config_kwargs={'use_memory_efficient_attention': True, 'unpad_inputs': True}
)

# %%% 2/ Load archive
human_data = load("datasets/similarities/2025-03-09/merged_groundtruth_data.npz")


# %% I. Naive SBERT finetuning
# %%% 1/ Create an accelerator to take care of multi-gpu
accelerator = Accelerator()

# %%% 2.a/ Extract training sentences
train_sentences = human_data['descriptions']

# %%% 2.b/ Training data sentence pairs
# %%%%% 2.b.a/ Same sentences
train_data_same_sentences = [InputExample(texts=[s, s]) for s in train_sentences]

# %%%%% 2.b.b/ Different sentences from same molecules
_is_cid_unique_in_arctander = is_key_unique_in('cids', human_data, "human/arctander")
_is_cid_unique_in_goodscents = is_key_unique_in('cids', human_data, "human/goodscents")
_is_cid_unique_in_leffingwell = is_key_unique_in('cids', human_data, "human/leffingwell")

_intersection_cids_arct_gdsc, _indices_intersection_arct_with_gdsc, _indices_intersection_gdsc_with_arct = np.lib.arraysetops.intersect1d(human_data['cids'][_is_cid_unique_in_arctander],
                                                                                                                                          human_data['cids'][_is_cid_unique_in_goodscents],
                                                                                                                                          assume_unique=True, return_indices=True)
zipped_arct_gdsc = zip(train_sentences[_is_cid_unique_in_arctander][_indices_intersection_arct_with_gdsc], train_sentences[_is_cid_unique_in_goodscents][_indices_intersection_gdsc_with_arct])
train_data_same_mols_arct_gdsc = [InputExample(texts=[s1, s2]) for (s1, s2) in zipped_arct_gdsc]

_intersection_cids_gdsc_leff, _indices_intersection_gdsc_with_leff, _indices_intersection_leff_with_gdsc = np.lib.arraysetops.intersect1d(human_data['cids'][_is_cid_unique_in_goodscents],
                                                                                                                                          human_data['cids'][_is_cid_unique_in_leffingwell],
                                                                                                                                          assume_unique=True, return_indices=True)
zipped_gdsc_leff = zip(train_sentences[_is_cid_unique_in_goodscents][_indices_intersection_gdsc_with_leff], train_sentences[_is_cid_unique_in_goodscents][_indices_intersection_leff_with_gdsc])
train_data_same_mols_gdsc_leff = [InputExample(texts=[s1, s2]) for (s1, s2) in zipped_gdsc_leff]

_intersection_cids_leff_arct, _indices_intersection_leff_with_arct, _indices_intersection_arct_with_leff = np.lib.arraysetops.intersect1d(human_data['cids'][_is_cid_unique_in_leffingwell],
                                                                                                                                          human_data['cids'][_is_cid_unique_in_arctander],
                                                                                                                                          assume_unique=True, return_indices=True)
zipped_leff_arct = zip(train_sentences[_is_cid_unique_in_leffingwell][_indices_intersection_leff_with_arct], train_sentences[_is_cid_unique_in_goodscents][_indices_intersection_arct_with_leff])
train_data_same_mols_leff_arct = [InputExample(texts=[s1, s2]) for (s1, s2) in zipped_leff_arct]


# %%% 2.c/ DataLoader to batch your data
train_data = train_data_same_sentences
train_data.extend(train_data_same_mols_arct_gdsc)
train_data.extend(train_data_same_mols_gdsc_leff)
train_data.extend(train_data_same_mols_leff_arct)
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

# %%% 2.d/ Use the denoising auto-encoder loss
train_loss = MultipleNegativesRankingLoss(sentence_bert)

# %%% 2.e/ Acceleration
# model, training_dataloader, training_loss = accelerator.prepare(sentence_bert, train_dataloader, train_loss)

epoch0_encoding_dataset = dict()
for key in human_data:
    epoch0_encoding_dataset[key] = human_data[key]

epoch1_encoding_dataset = dict()
for key in human_data:
    epoch1_encoding_dataset[key] = human_data[key]

epoch10_encoding_dataset = dict()
for key in human_data:
    epoch10_encoding_dataset[key] = human_data[key]

# %%% 3.a/ Pre-training encoding
epoch0_encoding_dataset['encode_descriptions'] = sentence_bert.encode(train_sentences)
epoch0_simi_arct_gdsc, shuf_epoch0_simi_arct_gdsc = cosine_similarity('cids', epoch0_encoding_dataset, ("human/arctander", "human/goodscents"))
epoch0_simi_gdsc_leff, shuf_epoch0_simi_gdsc_leff = cosine_similarity('cids', epoch0_encoding_dataset, ("human/goodscents", "human/leffingwell"))
epoch0_simi_leff_arct, shuf_epoch0_simi_leff_arct = cosine_similarity('cids', epoch0_encoding_dataset, ("human/leffingwell", "human/arctander"))

# epoch0_similarity = sentence_bert.similarity(epoch0_encoding_dataset['encode_descriptions'], epoch0_encoding_dataset['encode_descriptions'])
# _is_cid_unique_in_arctander = is_key_unique_in('cids', human_data, "human/arctander")
# _is_cid_unique_in_goodscents = is_key_unique_in('cids', human_data, "human/goodscents")
# _intersection_cids_arct_gdsc, _indices_intersection_arct_with_gdsc, _indices_intersection_gdsc_with_arct = np.lib.arraysetops.intersect1d(human_data['cids'][_is_cid_unique_in_arctander],
#                                                                                                                                           human_data['cids'][_is_cid_unique_in_goodscents],
#                                                                                                                                           assume_unique=True, return_indices=True)
# fig_epoch0, ax_epoch0 = plt.subplots(1, 1)
# plt.imshow(epoch0_similarity)

# plt.imshow(epoch0_similarity[_is_cid_unique_in_arctander, :][_indices_intersection_arct_with_gdsc, :][:, _is_cid_unique_in_goodscents][:, _indices_intersection_gdsc_with_arct])

# %%% 3.b/ Call the fit method
sentence_bert.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, show_progress_bar=True)
epoch1_encoding_dataset['encode_descriptions'] = sentence_bert.encode(train_sentences)
epoch1_simi_arct_gdsc, shuf_epoch1_simi_arct_gdsc = cosine_similarity('cids', epoch1_encoding_dataset, ("human/arctander", "human/goodscents"))
epoch1_simi_gdsc_leff, shuf_epoch1_simi_gdsc_leff = cosine_similarity('cids', epoch1_encoding_dataset, ("human/goodscents", "human/leffingwell"))
epoch1_simi_leff_arct, shuf_epoch1_simi_leff_arct = cosine_similarity('cids', epoch1_encoding_dataset, ("human/leffingwell", "human/arctander"))

sentence_bert.fit(train_objectives=[(train_dataloader, train_loss)], epochs=9, show_progress_bar=True)
epoch10_encoding_dataset['encode_descriptions'] = sentence_bert.encode(train_sentences)
epoch10_simi_arct_gdsc, shuf_epoch10_simi_arct_gdsc = cosine_similarity('cids', epoch10_encoding_dataset, ("human/arctander", "human/goodscents"))
epoch10_simi_gdsc_leff, shuf_epoch10_simi_gdsc_leff = cosine_similarity('cids', epoch10_encoding_dataset, ("human/goodscents", "human/leffingwell"))
epoch10_simi_leff_arct, shuf_epoch10_simi_leff_arct = cosine_similarity('cids', epoch10_encoding_dataset, ("human/leffingwell", "human/arctander"))

# %% II. Plots
# %%% 1/ Cosine similarities inter datasets
fig_dist, ax = plt.subplots(3, 1)
for i, (dist, lbl) in enumerate(zip([epoch0_simi_arct_gdsc, epoch0_simi_gdsc_leff, epoch1_simi_leff_arct], ["Arct-Gdsc", "Gdsc-Leff", "Leff-Arct"])):
    simi_bin_edges: array = histogram_bin_edges(diag(dist), bins=20)
    simi_n_samples_in_bins: array = histogram(diag(dist), bins=simi_bin_edges)[0]
    simi_fractions: array = simi_n_samples_in_bins / simi_n_samples_in_bins.sum()
    ax[i].plot(simi_bin_edges[:-1], simi_fractions, color=colormaps['viridis'](256//3 * i), label=lbl)
    ax[i].set(xlim=[-.1, 1.1], ylim=[0, .15])
for i, (shuf, lbl) in enumerate(zip([shuf_epoch0_simi_arct_gdsc, shuf_epoch0_simi_gdsc_leff, shuf_epoch0_simi_leff_arct], ["Permutation", "Permutation", "Permutation"])):
    shuf_bin_edges: array = histogram_bin_edges(diag(shuf), bins=20)
    shuf_n_samples_in_bins: array = histogram(diag(shuf), bins=shuf_bin_edges)[0]
    shuf_fractions: array = shuf_n_samples_in_bins / shuf_n_samples_in_bins.sum()
    ax[i].plot(shuf_bin_edges[:-1], shuf_fractions, color=colormaps['viridis'](256//3 * i), label=lbl, linestyle='dashed')
for i in range(3):
    ax[i].legend()
    ax[i].grid()
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[0].set_yticklabels(['', '.05', '.10', '.15'])
ax[1].set_yticklabels(['', '.05', '.10', '.15'])
ax[0].set(title="Cosine Distance between descriptions from different datasets of same molecule")
ax[1].set(ylabel='Fraction of total molecules')
ax[2].set(xlabel='Cosine Distances')
plt.subplots_adjust(hspace=0)


fig_dist, ax = plt.subplots(3, 1)
for i, (dist, lbl) in enumerate(zip([epoch1_simi_arct_gdsc, epoch1_simi_gdsc_leff, epoch1_simi_leff_arct], ["Arct-Gdsc", "Gdsc-Leff", "Leff-Arct"])):
    simi_bin_edges: array = histogram_bin_edges(diag(dist), bins=20)
    simi_n_samples_in_bins: array = histogram(diag(dist), bins=simi_bin_edges)[0]
    simi_fractions: array = simi_n_samples_in_bins / simi_n_samples_in_bins.sum()
    ax[i].plot(simi_bin_edges[:-1], simi_fractions, color=colormaps['viridis'](256//3 * i), label=lbl)
    ax[i].set(xlim=[-.1, 1.1], ylim=[0, .15])
for i, (shuf, lbl) in enumerate(zip([shuf_epoch1_simi_arct_gdsc, shuf_epoch1_simi_gdsc_leff, shuf_epoch1_simi_leff_arct], ["Permutation", "Permutation", "Permutation"])):
    shuf_bin_edges: array = histogram_bin_edges(diag(shuf), bins=20)
    shuf_n_samples_in_bins: array = histogram(diag(shuf), bins=shuf_bin_edges)[0]
    shuf_fractions: array = shuf_n_samples_in_bins / shuf_n_samples_in_bins.sum()
    ax[i].plot(shuf_bin_edges[:-1], shuf_fractions, color=colormaps['viridis'](256//3 * i), label=lbl, linestyle='dashed')
for i in range(3):
    ax[i].legend()
    ax[i].grid()
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[0].set_yticklabels(['', '.05', '.10', '.15'])
ax[1].set_yticklabels(['', '.05', '.10', '.15'])
ax[0].set(title="Cosine Distance between descriptions from different datasets of same molecule")
ax[1].set(ylabel='Fraction of total molecules')
ax[2].set(xlabel='Cosine Distances')
plt.subplots_adjust(hspace=0)


fig_dist, ax = plt.subplots(3, 1)
for i, (dist, lbl) in enumerate(zip([epoch10_simi_arct_gdsc, epoch10_simi_gdsc_leff, epoch10_simi_leff_arct], ["Arct-Gdsc", "Gdsc-Leff", "Leff-Arct"])):
    simi_bin_edges: array = histogram_bin_edges(diag(dist), bins=20)
    simi_n_samples_in_bins: array = histogram(diag(dist), bins=simi_bin_edges)[0]
    simi_fractions: array = simi_n_samples_in_bins / simi_n_samples_in_bins.sum()
    ax[i].plot(simi_bin_edges[:-1], simi_fractions, color=colormaps['viridis'](256//3 * i), label=lbl)
    ax[i].set(xlim=[-.1, 1.1], ylim=[0, .15])
for i, (shuf, lbl) in enumerate(zip([shuf_epoch10_simi_arct_gdsc, shuf_epoch10_simi_gdsc_leff, shuf_epoch10_simi_leff_arct], ["Permutation", "Permutation", "Permutation"])):
    shuf_bin_edges: array = histogram_bin_edges(diag(shuf), bins=20)
    shuf_n_samples_in_bins: array = histogram(diag(shuf), bins=shuf_bin_edges)[0]
    shuf_fractions: array = shuf_n_samples_in_bins / shuf_n_samples_in_bins.sum()
    ax[i].plot(shuf_bin_edges[:-1], shuf_fractions, color=colormaps['viridis'](256//3 * i), label=lbl, linestyle='dashed')
for i in range(3):
    ax[i].legend()
    ax[i].grid()
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[0].set_yticklabels(['', '.05', '.10', '.15'])
ax[1].set_yticklabels(['', '.05', '.10', '.15'])
ax[0].set(title="Cosine Distance between descriptions from different datasets of same molecule")
ax[1].set(ylabel='Fraction of total molecules')
ax[2].set(xlabel='Cosine Distances')
plt.subplots_adjust(hspace=0)





# %% II. Embedding projection
epoch0_encoding_dataset = dict()
for key in human_data:
    epoch0_encoding_dataset[key] = human_data[key]

epoch1_encoding_dataset = dict()
for key in human_data:
    epoch1_encoding_dataset[key] = human_data[key]

epoch10_encoding_dataset = dict()
for key in human_data:
    epoch10_encoding_dataset[key] = human_data[key]

good_cids = np.logical_and(human_data['cids'] != '', human_data['cids'] != '0')
is_arct = (human_data['datasets'][good_cids] == "human/arctander")
is_gdsc = (human_data['datasets'][good_cids] == "human/goodscents")
is_leff = (human_data['datasets'][good_cids] == "human/leffingwell")

epoch0_encoding_dataset['datasets'] = epoch0_encoding_dataset['datasets'][good_cids]
epoch1_encoding_dataset['datasets'] = epoch1_encoding_dataset['datasets'][good_cids]
epoch10_encoding_dataset['datasets'] = epoch10_encoding_dataset['datasets'][good_cids]

epoch0_encoding_dataset['encode_descriptions'] = epoch0_encoding_dataset['encode_descriptions'][good_cids]
epoch1_encoding_dataset['encode_descriptions'] = epoch1_encoding_dataset['encode_descriptions'][good_cids]
epoch10_encoding_dataset['encode_descriptions'] = epoch10_encoding_dataset['encode_descriptions'][good_cids]

epoch0_encoding_dataset['cids'] = epoch0_encoding_dataset['cids'][good_cids]
epoch1_encoding_dataset['cids'] = epoch1_encoding_dataset['cids'][good_cids]
epoch10_encoding_dataset['cids'] = epoch10_encoding_dataset['cids'][good_cids]

is_arct_and_gdsc_and_leff = np.zeros((human_data['cids'][good_cids],), dtype=bool)
is_arct_and_gdsc = np.zeros((human_data['cids'][good_cids],), dtype=bool)
is_gdsc_and_leff = np.zeros((human_data['cids'][good_cids],), dtype=bool)
is_leff_and_arct = np.zeros((human_data['cids'][good_cids],), dtype=bool)
is_only_arct = np.zeros((human_data['cids'][good_cids],), dtype=bool)
is_only_gdsc = np.zeros((human_data['cids'][good_cids],), dtype=bool)
is_only_leff = np.zeros((human_data['cids'][good_cids],), dtype=bool)
sameness_matrix = np.zeros((human_data['cids'][good_cids].shape[0], human_data['cids'][good_cids].shape[0]), dtype=int)
for ir, cid in enumerate(human_data['cids'][good_cids]):
    if (cid in human_data['cids'][good_cids][is_arct]) and (cid in human_data['cids'][good_cids][is_gdsc]) and (cid in human_data['cids'][good_cids][is_leff]):
        is_arct_and_gdsc_and_leff[ir] = True
    elif (cid in human_data['cids'][good_cids][is_arct]) and (cid in human_data['cids'][good_cids][is_gdsc]):
        is_arct_and_gdsc[ir] = True
    elif (cid in human_data['cids'][good_cids][is_gdsc]) and (cid in human_data['cids'][good_cids][is_leff]):
        is_gdsc_and_leff[ir] = True
    elif (cid in human_data['cids'][good_cids][is_leff]) and (cid in human_data['cids'][good_cids][is_arct]):
        is_leff_and_arct[ir] = True
    elif (cid in human_data['cids'][good_cids][is_arct]):
        is_only_arct[ir] = True
    elif (cid in human_data['cids'][good_cids][is_gdsc]):
        is_only_gdsc[ir] = True
    elif (cid in human_data['cids'][good_cids][is_leff]):
        is_only_leff[ir] = True
    sameness_matrix[ir, :] = 1 * (cid == human_data['cids'][good_cids])

Projector = Sequential(Linear(2 * 1024, 1)).to(device)
Optimizer = torch.optim.Adam(Projector.parameters())
Loss = torch.nn.CrossEntropyLoss().to(device)

# torch.Tensor.repeat(input, repeats)
# np.tile(human_data['encode_descriptions'][good_cids], [11039, 1, 1]).shape
for i_cid in tqdm(range(epoch1_encoding_dataset['encode_descriptions'].shape[0])):
    Optimizer.zero_grad()
    encoding_pair = (np.tile(epoch1_encoding_dataset['encode_descriptions'][i_cid, :], [11039, 1]), epoch1_encoding_dataset['encode_descriptions'])
    epoch1_projection = Projector(torch.from_numpy(np.concatenate(encoding_pair, axis=1)).to(device))
    loss = Loss(epoch1_projection.squeeze(), torch.from_numpy(sameness_matrix[i_cid, :]).float().to(device))
    loss.backward()
    Optimizer.step()


_is_cid_unique_in_arctander = is_key_unique_in('cids', human_data, "human/arctander")
_is_cid_unique_in_goodscents = is_key_unique_in('cids', human_data, "human/goodscents")
_is_cid_unique_in_leffingwell = is_key_unique_in('cids', human_data, "human/leffingwell")
_intersection_cids_arct_gdsc, _indices_arctander, _indices_goodscents = np.lib.arraysetops.intersect1d(human_data['cids'][_is_cid_unique_in_arctander],
                                                                                                       human_data['cids'][_is_cid_unique_in_goodscents],
                                                                                                       assume_unique=True, return_indices=True)

mat_encodings_arctander = np.tile(human_data['encode_descriptions'][_is_cid_unique_in_arctander], [_is_cid_unique_in_arctander.sum(), 1, 1])
mat_encodings_goodscents = np.tile(human_data['encode_descriptions'][_is_cid_unique_in_goodscents], [_is_cid_unique_in_goodscents.sum(), 1, 1])
mat_encodings_leffingwell = np.tile(human_data['encode_descriptions'][_is_cid_unique_in_leffingwell], [_is_cid_unique_in_leffingwell.sum(), 1, 1])
fig_imsc, ax_imsc = plt.subplots(1, 1)
plt.imshow(Projector().to_numpy())

epoch1_encoding_dataset['encode_descriptions'] = Projector(train_sentences)
epoch1_simi_arct_gdsc, shuf_epoch1_simi_arct_gdsc, _ = cosine_similarity('cids', epoch1_encoding_dataset, ("human/arctander", "human/goodscents"))
epoch1_simi_gdsc_leff, shuf_epoch1_simi_gdsc_leff, _ = cosine_similarity('cids', epoch1_encoding_dataset, ("human/goodscents", "human/leffingwell"))
epoch1_simi_leff_arct, shuf_epoch1_simi_leff_arct, _ = cosine_similarity('cids', epoch1_encoding_dataset, ("human/leffingwell", "human/arctander"))

fig_dist, ax = plt.subplots(3, 1)
for i, (dist, lbl) in enumerate(zip([epoch1_simi_arct_gdsc, epoch1_simi_gdsc_leff, epoch1_simi_leff_arct], ["Arct-Gdsc", "Gdsc-Leff", "Leff-Arct"])):
    simi_bin_edges: array = histogram_bin_edges(diag(dist), bins=20)
    simi_n_samples_in_bins: array = histogram(diag(dist), bins=simi_bin_edges)[0]
    simi_fractions: array = simi_n_samples_in_bins / simi_n_samples_in_bins.sum()
    ax[i].plot(simi_bin_edges[:-1], simi_fractions, color=colormaps['viridis'](256//3 * i), label=lbl)
    ax[i].set(xlim=[-.1, 1.1], ylim=[0, .15])
for i, (shuf, lbl) in enumerate(zip([epoch1_simi_arct_gdsc, epoch1_simi_gdsc_leff, epoch1_simi_leff_arct], ["Permutation", "Permutation", "Permutation"])):
    shuf_bin_edges: array = histogram_bin_edges(diag(shuf), bins=20)
    shuf_n_samples_in_bins: array = histogram(diag(shuf), bins=shuf_bin_edges)[0]
    shuf_fractions: array = shuf_n_samples_in_bins / shuf_n_samples_in_bins.sum()
    ax[i].plot(shuf_bin_edges[:-1], shuf_fractions, color=colormaps['viridis'](256//3 * i), label=lbl, linestyle='dashed')
for i in range(3):
    ax[i].legend()
    ax[i].grid()
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[0].set_yticklabels(['', '.05', '.10', '.15'])
ax[1].set_yticklabels(['', '.05', '.10', '.15'])
ax[0].set(title="Cosine Distance between descriptions from different datasets of same molecule")
ax[1].set(ylabel='Fraction of total molecules')
ax[2].set(xlabel='Cosine Distances')
plt.subplots_adjust(hspace=0)
