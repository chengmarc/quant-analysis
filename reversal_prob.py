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
from binance.spot import Spot

import itertools
from tqdm import tqdm
from collections import defaultdict

client = Spot()
exchange_info = client.exchange_info()


# %%
active_pairs = []
for pairs_info in exchange_info['symbols']:
    active_pairs.append([pairs_info['symbol'], pairs_info['status'], pairs_info['baseAsset'], pairs_info['quoteAsset']])
active_pairs = pd.DataFrame(active_pairs)
active_pairs = active_pairs[(active_pairs[1] == 'TRADING') & (active_pairs[3] == 'USDT')]

TICKER, TIMEFRAME, PERIOD = 'PEPEUSDT', '5m', 21600


# %%

def get_df(ticker, interval=TIMEFRAME, total_klines=PERIOD*4):
    
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
    
    df["pct_change"] = (df["close"] - df["open"]) / df["open"] * 100
    df["trend"] = df["pct_change"].apply(lambda x: 1 if x > 0 else -1)

    return df


# %%
check_hypothesis = True

if check_hypothesis: 
    
    random_list = np.random.choice([-1, 1], size=PERIOD*4)
    df = pd.DataFrame({'trend': random_list})
    df_sample, df_test = df[:PERIOD*3], df[PERIOD*3:]

else:
    
    df = get_df(TICKER)
    df_sample, df_test = df[:PERIOD*3], df[PERIOD*3:]


    observations = df['pct_change'].tolist()[1:]

    print("Mean:", np.mean(observations))
    print("Standard Deviation:", np.std(observations))

    plt.figure(figsize=(8, 8))
    plt.hist(observations, bins=60, range=(-5, 5), edgecolor='black')
    plt.title('Histogram of Observed Daily Returns')
    plt.xlabel('Daily Return Percentage %')
    plt.ylabel('Frequency')
    plt.show()


# %% O(2^n) Super slow, wrote it myself, just for reference

def generate_patterns(length):
    
    combinations = list(itertools.product([1, -1], repeat=length))
    return [list(pattern) for pattern in combinations]


def calculate_prob(df, pattern):
    
    data = df.copy()
    name = str(pattern).replace(' ', '')
    up = pattern + [1, 1, 1]
    down = pattern + [-1, -1, -1]
    
    data[f'{name}'] = data['trend'].rolling(window=len(pattern)).apply(lambda x: list(x) == pattern, raw=False)
    data[f'{name}_up'] = data['trend'].rolling(window=len(up)).apply(lambda x: list(x) == up, raw=False)
    data[f'{name}_down'] = data['trend'].rolling(window=len(down)).apply(lambda x: list(x) == down, raw=False)
    
    sample_size = data[f'{name}'].sum()
    prob_up = data[f'{name}_up'].sum() / data[f'{name}'].sum()
    prob_down = data[f'{name}_down'].sum() / data[f'{name}'].sum()
    
    result = [pattern, prob_up, prob_down, sample_size]
    return result


patterns = []
for i in range(3):
    patterns.extend(generate_patterns(i+1))

prob_matrix = []
for pattern in tqdm(patterns):
    prob_matrix.append(calculate_prob(df_sample, pattern))
    
prob_matrix = pd.DataFrame(prob_matrix, columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])


# %% O(n^2) Quadratic, wrote by ChatGPT, a bit faster

def generate_patterns(length):
    """Generate all possible trend patterns of a given length."""
    return np.array(list(itertools.product([1, -1], repeat=length)))


def calculate_prob(df, patterns):
    """Calculate probabilities for all patterns of the same length simultaneously."""
    
    patterns = np.array(patterns)  # Ensure patterns is a NumPy array
    length = patterns.shape[1]  # Length of patterns
    trends = df['trend'].values  # Convert column to NumPy array for speed
    
    if len(trends) < length + 1:
        return []
    
    rolling_matrix = np.lib.stride_tricks.sliding_window_view(trends, length + 1)

    results = []
    for pattern in patterns:
        mask = np.all(rolling_matrix[:, :-1] == pattern, axis=1)
        
        sample_size = mask.sum()
        if sample_size == 0:
            prob_up, prob_down = 0, 0
        else:
            prob_up = np.sum(mask & (rolling_matrix[:, -1] == 1)) / sample_size
            prob_down = np.sum(mask & (rolling_matrix[:, -1] == -1)) / sample_size
        
        results.append([tuple(pattern), prob_up, prob_down, sample_size])
    
    return results


patterns_dict = {i: generate_patterns(i) for i in range(1, 4)}

prob_matrix = []
for length, patterns in tqdm(patterns_dict.items()):
    prob_matrix.extend(calculate_prob(df_sample, patterns))
    
prob_matrix = pd.DataFrame(prob_matrix, columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])


# %% O(n^2) Quadratic, wrote by ChatGPT, use this one

def encode_pattern(pattern):
    """Convert a list pattern into an integer representation (bitwise encoding)."""
    return int("".join(['1' if x == 1 else '0' for x in pattern]), 2)


def generate_patterns(length):
    """Generate all possible trend patterns of a given length."""
    return [list(pattern) for pattern in itertools.product([1, -1], repeat=length)]


def calculate_prob(df, max_pattern_length=5):
    """Efficiently compute probabilities using hash maps with length-specific keys."""
    
    trends = df['trend'].values  # Convert column to NumPy array for speed
    n = len(trends)
    
    if n < max_pattern_length + 1:
        return pd.DataFrame(columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])  
    
    # Hash maps to store counts (keys are tuples of (encoded_pattern, length))
    pattern_counts = defaultdict(int)
    up_counts = defaultdict(int)
    down_counts = defaultdict(int)

    # Scan through the dataset once
    for i in range(n - max_pattern_length):
        for length in range(1, max_pattern_length + 1):
            pattern = tuple(trends[i:i+length])  # Ensure unique tuple per pattern
            key = (encode_pattern(pattern), length)  # Unique key per length
            
            pattern_counts[key] += 1
            if i + length < n:
                if trends[i+length] == 1:
                    up_counts[key] += 1
                else:
                    down_counts[key] += 1

    # Convert results into a DataFrame
    results = []
    for length in range(1, max_pattern_length + 1):
        for pattern_list in generate_patterns(length):
            key = (encode_pattern(pattern_list), length)
            sample_size = pattern_counts[key]
            
            if sample_size == 0:
                prob_up, prob_down = 0, 0
            else:
                prob_up = up_counts[key] / sample_size
                prob_down = down_counts[key] / sample_size
            
            results.append([tuple(pattern_list), prob_up, prob_down, sample_size])
    
    return pd.DataFrame(results, columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])


# Run the optimized function
prob_matrix = calculate_prob(df_sample, max_pattern_length=10)


# %%

def sort_prob(df):
    
    data = df.copy()
    data['Win Rate'] = data[['Up Probability', 'Down Probability']].max(axis=1)
    data['Direction'] = data.apply(lambda row: 'Short' if row['Win Rate'] == row['Down Probability'] else 'Long', axis=1)
    data = data.sort_values(by='Win Rate', ascending=False)
    data = data.drop(columns = ['Up Probability', 'Down Probability'])
    
    return data


prob_matrix = sort_prob(prob_matrix)


# %% Filter patterns
prob_matrix = prob_matrix[(prob_matrix["Direction"] == "Short")]
prob_matrix = prob_matrix[(prob_matrix["Sample Size"] > 400)]


# %%

def generate_signals(df, pattern, direction):
    
    data = df.copy()
    if type(pattern) is tuple: pattern = list(pattern)
    name = str(pattern).replace(' ', '')
    
    data[f'{name}'] = data['trend'].rolling(window=len(pattern)).apply(lambda x: list(x) == pattern, raw=False)     
    data.loc[data[f"{name}"].shift(1) == 1, "Signal"] = direction
    
    return data


winning_patterns = list(prob_matrix.iloc[:1].itertuples(index=False, name=None))

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


df_passive, final_value_passive = passive_strategy(df_test, start_index=1)


# %%

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


df_active, trade_history, final_value_active = active_strategy(df_test, start_index=1)


# %%
plt.figure(figsize=(8, 5))
plt.plot(df_passive.index, df_passive['equity'], label='Returns (Passive)', color='blue', linewidth=2)
plt.plot(df_active.index, df_active['equity'], label='Returns (Active)', color='gray', linestyle='dashed', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Equity Curve Comparison")
plt.legend()
plt.grid(True)

