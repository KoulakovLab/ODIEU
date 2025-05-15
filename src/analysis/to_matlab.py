#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:33:55 2025

@author: cyrille
"""

import sys

import pandas as pd
import scipy.io


in_file = sys.argv[1]
out_file = ""
if in_file.find(".pkl"):
    df = pd.read_pickle(in_file)
    out_file = in_file.replace(".pkl", ".mat")
elif in_file.find(".pkl"):
    df = pd.read_csv(in_file)
    out_file = in_file.replace(".csv", ".mat")
else:
    print("Can only convert files with .csv or .pkl extensions")
    exit(-1)

scipy.io.savemat(out_file, df.to_dict())
