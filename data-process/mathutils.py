# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import torch
import numpy as np
import pandas as pd


def log_transform(col:pd.Series, use_gpu:bool) -> pd.Series:
    if use_gpu:
        lst = col.tolist()
        vector = torch.tensor(lst).cuda()
        log_vector = torch.log10(vector)
        lst = log_vector.tolist()
        col = pd.Series(lst)
    else:
        col = np.log10(col)
    return col

