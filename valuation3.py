# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 18:33:29 2024

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import torch
import re

USE_GPU = True

# %%

# Path to the folder containing the CSV spreadsheets
folder_path = r'C:\Users\marcc\My Drive\Data Extraction\geckoscan-all'

# List to store the "marketcap" column from each spreadsheet
df_list = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    
    date = re.search(r'\d{4}-\d{2}-\d{2}', filename).group()
    df = pd.read_csv(os.path.join(folder_path, filename))
    df = df[['Symbol', 'Volume24h', 'MarketCap']]

    df_list.append((date, df))
    
# %%
data = pd.concat([df for _, df in df_list], keys=[date for date, _ in df_list], names=['Date'])

# %%
import seaborn as sns
# Plot the movement of scatterplots across all dates
sns.lineplot(data=data, x='Volume24h', y='MarketCap', hue='Date')