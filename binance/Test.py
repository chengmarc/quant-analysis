# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 19:45:55 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%

from Connector import get_exchange_info, get_active_pairs

exchange_info = get_exchange_info()
active_pairs = get_active_pairs()
active_pairs = active_pairs[(active_pairs['status'] == 'TRADING') & (active_pairs['quoteAsset'] == 'USDT')]

# %%

TICKER, TIMEFRAME, PERIOD = 'DODOUSDT', '5m', 26000*2

from Connector import get_market_data    
df = get_market_data(TICKER, interval=TIMEFRAME, total_klines=PERIOD)


# %%
import itertools

from collections import defaultdict
def generate_sequences(n):
    """Generate all possible sequences of -1 and 1 up to length n."""
    sequences = []
    for length in range(1, n+1):  # Include lengths from 1 to n
        sequences.extend(itertools.product([-1, 1], repeat=length))
    return sequences

def find_occurrences(df, n):
    """Find occurrences of sequences in 'trend' and collect 'pct_change' values after each occurrence."""
    results = defaultdict(list)
    
    # Use a sliding window to efficiently track sequences
    for i in range(len(df) - 1):  # Ensure there's a next row
        for length in range(1, min(n, len(df) - i) + 1):  # Limit sequence length
            seq = tuple(df['trend'].iloc[i:i+length])
            if i + length < len(df):  # Ensure there's a next row for pct_change
                results[seq].append(df['pct_change'].iloc[i + length])
    
    return dict(results)


results = find_occurrences(df, n=12)

# %%
def create_summary_dataframe(results):
    """Create a DataFrame with mean, variance, sample size, and the list itself for each sequence."""
    data = []
    for seq, values in results.items():
        sample_size = len(values)
        if values:  # Ensure non-empty list
            mean_val = sum(values) / sample_size
            variance_val = sum((x - mean_val) ** 2 for x in values) / sample_size
        else:
            mean_val, variance_val = None, None
        data.append({'sequence': seq, 'mean': mean_val, 'variance': variance_val, 'sample_size': sample_size, 'values': values})
    
    return pd.DataFrame(data)

def plot_histograms(df):
    """Plot histograms for each sequence's pct_change values."""
    for _, row in df.iterrows():
        values = row['values']
        if values:
            plt.figure(figsize=(6, 4))
            plt.hist(values, bins=10, range=[-0.5, 0.5], edgecolor='black', alpha=0.7)
            plt.title(f'Histogram for sequence {row["sequence"]}')
            plt.xlabel('Pct Change')
            plt.ylabel('Frequency')
            plt.show()
            
summary_df = create_summary_dataframe(results)

summary_df = summary_df[(summary_df["sample_size"] > 200)]
plot_histograms(summary_df)