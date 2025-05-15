#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:35:37 2025

@author: cyrille
"""

import pandas as pd
import numpy as np
import scipy.io

cids_aligned_descriptions = pd.read_csv("bar_plots/merged_NEW.csv", sep="$")
contenders: pd.DataFrame = cids_aligned_descriptions.loc[np.logical_not(cids_aligned_descriptions.CID.isna()) & (cids_aligned_descriptions.CID != 0)]
contenders.rename(columns={'CID': 'FLOATCID',
                           'Description': 'Human descriptions',
                           'BagOfWords': 'Human labels',
                           'generated_human': 'Beautified human descriptions',
                           'generated_w_labels': 'Beautified human labels',
                           'NEW_deepnose_sentences': 'Beautified DeepNose descriptions',
                           'NEW_deepnose_labels': 'Beautified DeepNose labels',
                           'zero_shot': 'From SMILES - zero shot',
                           'few_shot_human': 'From SMILES - human examples',
                           'few_shot_gen_BoW': 'From SMILES - beautified human examples',
                           'few_shot_gen_human': 'From SMILES - beautified labels examples'},
                  inplace=True)
contenders['CID'] = [str(cid) for cid in np.int64(contenders['FLOATCID'].to_list())]
for dataset in ["arctander", "goodscents", "leffingwell"]:
    contenders[contenders.Dataset == dataset] = contenders[contenders.Dataset == dataset].drop_duplicates(subset=["CID"], keep=False)
contenders.set_index('CID', inplace=True)

description_types = ['Dataset', 'Beautified human descriptions', 'Human descriptions', 'Human labels', 'Beautified human labels', 'Beautified DeepNose descriptions','Beautified DeepNose labels', 'From SMILES - zero shot', 'From SMILES - human examples', 'From SMILES - beautified human examples', 'From SMILES - beautified labels examples']
contenders = contenders[description_types]

pd.DataFrame(description_types, columns=[""]).to_csv("description_types.csv", index=False, header=False)

contenders.to_csv("descriptions.csv", sep="$")
pd.to_pickle(contenders, "descriptions.pkl")
scipy.io.savemat("descriptions.mat", contenders.to_dict())
