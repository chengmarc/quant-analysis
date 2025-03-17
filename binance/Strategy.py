# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:34:09 2025

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
TICKER, TIMEFRAME, PERIOD = 'PEPEUSDT', '1m', 43200*4

def null_hypothesis():  # H0: Patterns with high win rates are merely statistical coincidences
    
    random_list = np.random.choice([-1, 1], size=PERIOD)
    df = pd.DataFrame({'trend': random_list})
    df_sample, df_test = df[:PERIOD*3], df[PERIOD*3:]    
    return df_sample, df_test
    

def alternate_hypothesis():  # H1: Patterns with high win rates arise due to an unknown underlying cause
    
    from Connector import get_market_data    
    df = get_market_data(TICKER, interval=TIMEFRAME, total_klines=PERIOD)
    df_sample, df_test = df[:PERIOD//4*3], df[PERIOD//4*3:]    
    return df_sample, df_test


# %%
from Algorithm import get_prob_matrix, sort_prob_matrix

df_sample, df_test = null_hypothesis()
prob_matrix_h0 = get_prob_matrix(df_sample, 15)
prob_matrix_h0 = sort_prob_matrix(prob_matrix_h0, sortby='Win Rate')
prob_matrix_h0 = prob_matrix_h0[(prob_matrix_h0["Sample Size"] > 200)]

df_sample, df_test = alternate_hypothesis()
prob_matrix_h1 = get_prob_matrix(df_sample, 15)
prob_matrix_h1 = sort_prob_matrix(prob_matrix_h1, sortby='Win Rate')
prob_matrix_h1 = prob_matrix_h1[(prob_matrix_h1["Sample Size"] > 200)]

observations = df_sample['pct_change'].tolist()[1:]
print("Mean:", np.mean(observations))
print("Standard Deviation:", np.std(observations))
print("Max:", np.max(observations))
print("Min:", np.min(observations))


# %%

def generate_signals(df, pattern, direction):
    
    data = df.copy()
    if type(pattern) is tuple: pattern = list(pattern)
    name = str(pattern).replace(' ', '')
    
    data[f'{name}'] = data['trend'].rolling(window=len(pattern)).apply(lambda x: list(x) == pattern, raw=False)     
    data.loc[data[f"{name}"].shift(1) == 1, "Signal"] = direction
    
    return data


winning_patterns = [list(prob_matrix_h1.iloc[:10].itertuples(index=False, name=None))][0]

df_test = df_test.iloc[:, :14]
for pattern in winning_patterns:
    df_test = generate_signals(df_test, pattern[0], pattern[3])


# %%

def passive_strategy(df, start_index):
    
    data = df.copy()[start_index:]
    
    capital = 10000
    starting_price = data.iloc[0]['open']
    
    amount_held = capital / starting_price  
    final_value = amount_held * data.iloc[-1]['close']  
    
    equity_curve = [amount_held * price for price in data['close']]
    data.loc[:, 'equity'] = equity_curve
    
    return data, final_value


def active_strategy(df, start_index):
    
    data = df.copy()[start_index:]
    
    capital = 10000
    position = 0
    leverage = 1  # 3x leverage
    equity_curve = []  
    trade_history = []
    
    for i in range(len(data)):
        price_open = data.iloc[i]['open']
        price_close = data.iloc[i]['close']
        action = data.iloc[i]['Signal']

        if action == "Long":
            # Increase the capital used for buying by applying leverage
            amount_to_buy = (capital * leverage) / price_open
            position += amount_to_buy * 1  # Apply trading fee
            capital -= amount_to_buy * price_open  # Deduct capital for buying
            trade_history.append({'Type': 'Buy', 'Price': price_open, 'Amount': amount_to_buy, 'Capital': capital})

            # Selling the position (closing the long trade)
            amount_to_sell = position
            position -= amount_to_sell
            capital += amount_to_sell * price_close * 1  # Apply trading fee when selling
            trade_history.append({'Type': 'Sell', 'Price': price_close, 'Amount': amount_to_sell, 'Capital': capital})
        
        if action == "Short":
            # Increase the capital used for short selling by applying leverage
            amount_to_sell = (capital * leverage) / price_open
            position -= amount_to_sell  # Position becomes negative for shorting
            capital += amount_to_sell * price_open * 1  # Collect capital from short selling, apply trading fee
            trade_history.append({'Type': 'Sell', 'Price': price_open, 'Amount': amount_to_sell, 'Capital': capital})

            # Buying back the position to close the short trade
            amount_to_buy = abs(position)
            position += amount_to_buy * 1  # Apply trading fee
            capital -= amount_to_buy * price_close  # Deduct capital for buying back
            trade_history.append({'Type': 'Buy', 'Price': price_close, 'Amount': amount_to_buy, 'Capital': capital})
        
        # Calculate equity
        equity = capital + (position * price_close)
        equity_curve.append(equity)  # Track equity curve
    
    data.loc[:, 'equity'] = equity_curve
    final_value = capital + (position * data.iloc[-1]['close'])
    
    return data, trade_history, final_value


# %%
df_passive, _ = passive_strategy(df_test, start_index=1)
df_active, trade_history, _ = active_strategy(df_test, start_index=1)

plt.figure(figsize=(8, 5))
plt.plot(df_passive.index, df_passive['equity'], label='Returns (Passive)', color='blue', linewidth=2)
plt.plot(df_active.index, df_active['equity'], label='Returns (Active)', color='gray', linestyle='dashed', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Equity Curve Comparison")
plt.legend()
plt.grid(True)

