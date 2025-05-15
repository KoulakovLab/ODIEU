#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:28:26 2025

@author: cyrille
"""
import sys

import numpy as np
import pandas as pd
import scipy.io


df = pd.read_pickle(sys.argv[1])
for ind in df.index:
    for col in df.columns:
        df[ind, col] = np.mean(df[ind, col])

df.to_csv("average_" + sys.argv[1].replace(".pkl", ".csv"))
pd.to_pickle(df, "average_" + sys.argv[1])
scipy.io.savemat("average_" + sys.argv[1].replace(".pkl", ".mat"), df.to_dict())
