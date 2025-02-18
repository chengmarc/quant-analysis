# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:34:09 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd
import matplotlib.pyplot as plt


def plot_data(df, ticker):
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    axs[0].plot(df.index, df['Close'], label='Weekly Close Price', color='blue')
    axs[0].set_title(f'Weekly Close Price of {ticker} Index')
    axs[0].set_ylabel('Price')

    axs[1].plot(df.index, df['LogClose'], label='Weekly Log Close Price', color='red')
    axs[1].set_title(f'Weekly Log Close Price of {ticker} Index')
    axs[1].set_ylabel('Log Price')

    [ax.grid(True) or ax.legend() for ax in axs]
    _, _ = plt.tight_layout(), plt.show()
    

from datahelper import yf_get, yf_ha
df = yf_get('^GSPC', cutoff='1990-01-01', interval='1d')
df = yf_ha(df)
plot_data(df, 'BTC')


# %%
def count_patterns_up(data, n):
    
    df = data.copy()
        
    df[f"Uptrend_{n}"] = df["Trend"].rolling(window=n).sum() == n
    up_all_consec = df[f"Uptrend_{n}"].sum()
    
    df[f"Up_Down_{n}"] = df[f"Uptrend_{n}"] & (df["Trend"].shift(-1) == -1)
    up_reversal = df[f"Up_Down_{n}"].sum()
   
    return up_all_consec, up_reversal

def count_patterns_dn(data, n):
    
    df = data.copy()
    
    df[f"Downtrend_{n}"] = df["Trend"].rolling(window=n).sum() == -n
    dn_all_consec = df[f"Downtrend_{n}"].sum()
    
    df[f"Down_Up_{n}"] = df[f"Downtrend_{n}"] & (df["Trend"].shift(-1) == 1)
    dn_reversal = df[f"Down_Up_{n}"].sum()

    return dn_all_consec, dn_reversal


result = []
for i in range(30):
    
    a, b = count_patterns_up(df, n=i+1)
    c, d = count_patterns_dn(df, n=i+1)
    
    result.append([i+1, b/a, d/c])
    
result = pd.DataFrame(result)
    
