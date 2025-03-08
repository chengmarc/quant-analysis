# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:34:09 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


df = pd.read_csv('VOO.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()
df = df.rename(columns={"Close/Last": 'Close'})
df["drop_count"] = [sum(df["Close"].iloc[:i] > df["Close"].iloc[i]) for i in range(len(df))]
    
plt.plot(df.index, df['drop_count'], label='Close Price', color='blue')
# %%
def generate_lookback(df, period=20):
    df = df.copy()
    df["drop_percentage_{period}"] = [
        1 - sum(df["Close"].iloc[max(0, i - period):i] > df["Close"].iloc[i]) / min(i, period)
        if i > 0 else 0
        for i in range(len(df))]
    
    plt.plot(df.index, df['drop_percentage_{period}'], label='Close Price', color='blue')
    df["Signal"] = "Empty"
    df.loc[df["drop_percentage_{period}"] <0.1, "Signal"] = "Buy"  # Buy signa
    return df

data = generate_lookback(df, period=20)

# %%

def DCA_strategy(df, cash_per_day=100):
    df = df.copy()
    capital = 0
    position = 0
    df['equity_buy_daily'] = 0  # Column to store the equity curve

    for i in range(len(df)):
        capital += cash_per_day  # Receive $100 at start of day
        if capital > 0:  # Avoid division by zero
            amount_bought = capital / df.loc[df.index[i], 'Close']
            position += amount_bought
            capital= 0  # All money is spent on shares
        
        df.loc[df.index[i], 'capital'] = capital
        df.loc[df.index[i], 'position'] = position
        df.loc[df.index[i], 'equity'] = position * df.loc[df.index[i], 'Close'] + capital
    
    final_value = capital + (position * df.iloc[-1]['Close'])
    return df, final_value

df_daily, _ = DCA_strategy(data)

# %%
def RSI_strategy(df, cash_per_day=100, allocation_per_trade=0.5):
    df = df.copy()
    capital = 0
    position = 0
    df['equity_rsi'] = 0  # Column to store the equity curve

    for i in range(len(df)):
        capital += cash_per_day  # Receive $100 at start of day
        if capital > 0 and df.loc[df.index[i], 'Signal'] == 'Buy':  # Avoid division by zero
            amount_to_invest = capital * allocation_per_trade
            amount_bought = amount_to_invest / df.loc[df.index[i], 'Close']
            position += amount_bought
            capital -= amount_to_invest 
        
        if position > 0 and df.loc[df.index[i], 'Signal'] == 'Sell':  # Avoid division by zero
            position_to_sell = position * allocation_per_trade
            amount_sold = position_to_sell * df.loc[df.index[i], 'Close']
            position -= position_to_sell
            capital += amount_sold 
        
        df.loc[df.index[i], 'capital'] = capital
        df.loc[df.index[i], 'position'] = position
        df.loc[df.index[i], 'equity'] = position * df.loc[df.index[i], 'Close'] + capital
    
    final_value = capital + (position * df.iloc[-1]['Close'])
    return df, final_value

df_rsi, _ = RSI_strategy(data)

# %%
def optimize_allocation(df):
    
    all_final_values = []

    for lookback in tqdm(range(1, 200)):        
        data = generate_lookback(df, period=lookback)
        data = generate_signals(data, period=lookback)
        _, final_value = RSI_strategy(data, cash_per_day=100, allocation_per_trade=1)
        all_final_values.append(final_value)
    return all_final_values

dct = optimize_allocation(df)

# %%

def plot_results(df_daily, df_rsi, log=False):
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    axs[0].plot(df_daily.index, df_daily['equity'], label='Returns (Buy & Hold)', color='gray', linestyle='dashed', linewidth=2)
    axs[0].plot(df_rsi.index, df_rsi['equity'], label='Returns (DCA)', color='blue', linewidth=2)
    axs[0].set_title('Equity Curve Comparison')
    axs[0].set_ylabel('Portfolio Value ($)')

    # Plot 2: BTC Price with Buy/Sell Signals
    buy_signals = df_daily[df_daily['Signal'] == 'Buy']
    sell_signals = df_daily[df_daily['Signal'] == 'Sell']

    axs[1].plot(df_daily.index, df_daily['Close'], label='BTC Price', color='black', linewidth=2)
    axs[1].scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', color='green', marker='s', s=25)
    axs[1].scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', color='red', marker='s', s=25)
    axs[1].set_title('BTC Price with Buy/Sell Signals')
    axs[1].set_ylabel('Price')

    [ax.grid(True) or ax.legend() for ax in axs]
    _, _ = plt.tight_layout(), plt.show()


plot_results(df_daily, df_rsi)

