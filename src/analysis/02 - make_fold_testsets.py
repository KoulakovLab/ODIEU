#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:35:54 2025

@author: cyrille
"""

import numpy as np
import pandas as pd
import scipy.io


contenders_beautified_first = pd.read_pickle("descriptions.pkl")

testsets_beautified_first: dict[int, pd.DataFrame] = dict()
for fold in range(1, 5+1):
    aggregateset = np.load(f"datasets/cross_validation/fold_{fold}.npz")
    testsets_datasets = np.array([s.replace("human/", "") for s in aggregateset['testset_datasets']])
    select = (contenders_beautified_first.index == 0)
    for icid, cid in enumerate(aggregateset['testset_cids']):
        select |= ((contenders_beautified_first.index == cid) & (contenders_beautified_first.Dataset == testsets_datasets[icid]))
    testsets_beautified_first[fold] = pd.DataFrame.from_dict({'CID': aggregateset['testset_cids'],
                                                              'Dataset': testsets_datasets,
                                                              'Reference Description': contenders_beautified_first.loc[select, 'Beautified human descriptions']})
    testsets_beautified_first[fold].set_index('CID', inplace=True)
    testsets_beautified_first[fold].to_csv(f"testsets_{fold}.csv")
    pd.to_pickle(testsets_beautified_first[fold], f"testsets_{fold}.pkl")
    scipy.io.savemat(f"testsets_{fold}.mat", testsets_beautified_first[fold].to_dict())
