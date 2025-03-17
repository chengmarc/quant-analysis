# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd
import itertools
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

from tqdm import tqdm
from binance.spot import Spot


# %%
client = Spot()
print(client.time())

exchange_info = client.exchange_info()
exchange_pair = exchange_info["symbols"]
symbols = [s["symbol"] for s in exchange_pair if 'USDT' in s["symbol"]]


# %%
def get_df(ticker, interval="1d", total_klines=5000):
    
    all_klines = []
    end_time = None  # Start from the latest data

    while len(all_klines) < total_klines:
        try:
            # Fetch K-lines with pagination
            klines = client.klines(ticker, "1d", limit=1000, endTime=end_time)
            if not klines: break

            all_klines.extend(klines)
            end_time = klines[0][0] - 1
            
        except Exception as e:            
            print(f"Error fetching {ticker}: {e}")
            break

    # Convert to DataFrame
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_klines[:total_klines], columns=columns)  # Trim excess data
    
    df = df.apply(pd.to_numeric, errors='coerce')
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df.sort_values(by="open_time").reset_index(drop=True)

    return df

market = {}
for s in tqdm(symbols[:5]):
    market[s] = get_df(s)


# %%
for symbol in market:    
    df = market[symbol]
    df['high_change'] = (df["high"] - df["open"]) / df["open"] + 1
    df['low_change'] = (df["low"] - df["open"]) / df["open"] + 1
    df['close_change'] = (df["close"] - df["open"]) / df["open"] + 1
    market[symbol] = df


# %%
def fit_dist(rv_list):

    distributions = [stats.expon, stats.norm, stats.cauchy, stats.t, stats.f,
                     stats.alpha, stats.beta, stats.gamma, 
                     stats.chi, stats.chi2]

    best_fit = None
    best_params = None
    best_ks_stat = np.inf

    for distribution in distributions:

        params = distribution.fit(rv_list)
        ks_stat, _ = stats.kstest(rv_list, distribution.cdf, args=params)
        # Perform the Kolmogorov-Smirnov test

        if ks_stat < best_ks_stat:
            best_fit = distribution
            best_params = params
            best_ks_stat = ks_stat

    print("Best fit distribution:", best_fit.name)
    print("Best fit parameters:", best_params)
    print("Kolmogorov-Smirnov statistic:", best_ks_stat)

    return best_fit, best_params


distribution = {}

for symbol in market:
    df = market[symbol]
    distribution[symbol] = {}

    observ_high = df['high_change'].tolist()
    observ_low = df['low_change'].tolist()
    observ_close = df['close_change'].tolist()
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    axs[0, 0].hist(observ_high, bins=30, range=(1, 1.5), color='lightgreen', edgecolor='black')
    axs[0, 1].hist(observ_low, bins=30, range=(0.5, 1), color='lightcoral', edgecolor='black')
    axs[0, 2].hist(observ_close, bins=30, range=(0.5, 1.5), color='skyblue', edgecolor='black')
    
    dist_a, params_a = fit_dist(observ_high)
    dist_b, params_b = fit_dist(observ_low)
    dist_c, params_c = fit_dist(observ_close)
    
    distribution[symbol]["high"] = dist_a, params_a
    distribution[symbol]["low"] = dist_b, params_b
    distribution[symbol]["close"] = dist_c, params_c
        
    axs[1, 0].hist(dist_a.rvs(*params_a, size=10000), range=(1, 1.5), bins=30, color='lightgreen', edgecolor='black')
    axs[1, 1].hist(dist_b.rvs(*params_b, size=10000), range=(0.5, 1), bins=30, color='lightcoral', edgecolor='black')
    axs[1, 2].hist(dist_c.rvs(*params_c, size=10000), range=(0.5, 1.5), bins=30, color='skyblue', edgecolor='black')
     
    fig.suptitle(f'Observed vs Simulated Distribution of {symbol} High, Low, Close Price Comparing to Open Price', fontsize=14, fontweight='bold')
    
    [ax.grid(True) for ax in list(itertools.chain(*axs))]
    _, _ = plt.tight_layout(), plt.show()
    
    
# %%

