# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:34:09 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd
import itertools
from binance.spot import Spot

client = Spot()
print(client.time())


# %%
def get_df(ticker, interval="1m", total_klines=43200):
    
    all_klines = []
    end_time = None  # Start from the latest data

    while len(all_klines) < total_klines:
        try:
            # Fetch K-lines with pagination
            klines = client.klines(ticker, interval, limit=1000, endTime=end_time)
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

df = get_df("IOSTUSDT")

df["Pct_Change"] = (df["close"] - df["open"]) / df["open"] * 100
df["Trend"] = df["Pct_Change"].apply(lambda x: 1 if x > 0 else -1)


# %%
def get_pattern(n, direction, reversal=False):
    
    if direction == 'up': k = 1
    elif direction == 'down': k = -1
    
    if not reversal: pattern = [k for _ in range(n)]
    else: pattern = [k for _ in range(n)] + [-k]

    return pattern


def calculate_prob(n, data):
    
    df = data.copy()
    result = []
    
    for i in range(n):
        
        pattern_up = get_pattern(i+1, 'up')
        pattern_up_reverse = get_pattern(i+1, 'up', reversal=True)
        
        series_up = df['Trend'].rolling(window=len(pattern_up)).apply(lambda x: list(x) == pattern_up, raw=False)
        series_up_rev = df['Trend'].rolling(window=len(pattern_up_reverse)).apply(lambda x: list(x) == pattern_up_reverse, raw=False)
        
        pattern_down = get_pattern(i+1, 'down')
        pattern_down_reverse = get_pattern(i+1, 'down', reversal=True)
        
        series_down = df['Trend'].rolling(window=len(pattern_down)).apply(lambda x: list(x) == pattern_down, raw=False)
        series_down_rev = df['Trend'].rolling(window=len(pattern_down_reverse)).apply(lambda x: list(x) == pattern_down_reverse, raw=False)
        
        prob_up = series_up_rev.sum() / series_up.sum()
        prob_down = series_down_rev.sum() / series_down.sum()
        
        result.append([i+1, prob_up, prob_down])
        
    result = pd.DataFrame(result)    
    return result


result = calculate_prob(20, df)


# %%
def generate_sequences(length):
    comb = list(itertools.product([1, -1], repeat=length))
    return [list(pattern) for pattern in comb]


n = 5
comb = generate_sequences(n)

result = []
for pattern in comb:
    
    data = df.copy()
    name = str(pattern).replace(' ', '')
    up = pattern + [1]
    down = pattern + [-1]
    
    data[f'{name}'] = data['Trend'].rolling(window=len(pattern)).apply(lambda x: list(x) == pattern, raw=False)
    data[f'{name}_up'] = data['Trend'].rolling(window=len(up)).apply(lambda x: list(x) == up, raw=False)
    data[f'{name}_down'] = data['Trend'].rolling(window=len(down)).apply(lambda x: list(x) == down, raw=False)
    
    prob_up = data[f'{name}_up'].sum() / data[f'{name}'].sum()
    prob_down = data[f'{name}_down'].sum() / data[f'{name}'].sum()
    
    result.append([name, prob_up, prob_down])
    

result = pd.DataFrame(result)
result = result.sort_values(by=result.columns[1], ascending=True)

# %%
patterns = [[-1,-1,-1,-1,-1],
            [-1,-1,1,-1,-1],
            [-1,-1,-1,1,-1],
            [-1,1,-1,-1,-1],
            [1,-1,-1,-1,-1],]
            

def generate_signals(df, pattern):
    
    name = str(pattern).replace(' ', '')
    
    df[f'{name}'] = df['Trend'].rolling(window=len(pattern)).apply(lambda x: list(x) == pattern, raw=False)
     
    df.loc[df[f"{name}"] == 1, "Signal"] = True # Buy signal
    df.loc[df["Signal"].shift(1) == True, "Action"] = "Short"
    return df

for pattern in patterns:
    df = generate_signals(df, pattern)

# %%


def HODL_strategy(df, start_time, initial_capital):
    
    df = df.copy()[start_time:]
    first_price = df.iloc[0]['close']
    btc_held = initial_capital / first_price  
    final_value = btc_held * df.iloc[-1]['close']  
    
    equity_curve = [btc_held * price for price in df['close']]  
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve

    return output, final_value

df_HODL, final_value_HODL = HODL_strategy(df, start_time=1, initial_capital=10000)


# %%
def PROB_strategy(df, start_time, initial_capital):
    
    df = df.copy()[start_time:]
    capital = initial_capital
    position = 0
    equity_curve = []  
    trade_history = []
    
    for i in range(len(df)):
        price_open = df.iloc[i]['open']
        price_close = df.iloc[i]['close']
        action = df.iloc[i]['Action']

        if action == "Long":
            btc_to_buy = capital / price_open
            position += btc_to_buy
            trade_history.append({'Type': 'Buy', 'Price': price_open, 'BTC': btc_to_buy, 'Capital': capital})
            capital = 0

            btc_to_sell = position
            capital += btc_to_sell * price_close  
            trade_history.append({'Type': 'Sell', 'Price': price_close, 'BTC': btc_to_sell, 'Capital': capital})
            position = 0
        
        if action == "Short":
            btc_to_sell = capital / price_open
            position -= btc_to_sell
            trade_history.append({'Type': 'Sell', 'Price': price_open, 'BTC': btc_to_sell, 'Capital': capital})
            capital += btc_to_sell * price_open  # Collect capital from short selling

            btc_to_buy = abs(position)
            capital -= btc_to_buy * price_close  # Buy back BTC at close price
            trade_history.append({'Type': 'Buy', 'Price': price_close, 'BTC': btc_to_buy, 'Capital': capital})
            position = 0
        
        equity = capital + (position * price_open)

        equity_curve.append(equity)  
        
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve
    final_value = capital + (position * df.iloc[-1]['close'])
    
    return output, trade_history, final_value

df_PROB, trade_history, final_value_PROB = PROB_strategy(df, start_time=1, initial_capital=10000)


# %%

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(12, 10))

axs[0].plot(df_HODL.index, df_HODL['Equity'], label='Returns (Buy & Hold)', color='gray', linestyle='dashed', linewidth=2)
axs[0].plot(df_PROB.index, df_PROB['Equity'], label='Returns (DCA)', color='blue', linewidth=2)
axs[0].set_title('Equity Curve Comparison')
axs[0].set_ylabel('Portfolio Value ($)')
