# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:17:40 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
df = pd.read_csv('btc.csv')
df['Date'] = pd.to_datetime(df['Timestamp'])
df.set_index('Date', inplace=True)
df = df.sort_index()
df = df[['Open', 'High', 'Low', 'Close']]

def ha(df):
    
    ha_df = df.copy()
    ha_df['HA_Close'] = (ha_df['Open'] + ha_df['High'] + ha_df['Low'] + ha_df['Close']) / 4

    ha_open = [ha_df['Open'].iloc[0]]  # Start with the first open value
    for i in range(1, len(ha_df)):
        ha_open.append((ha_open[-1] + ha_df['HA_Close'].iloc[i-1]) / 2)

    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
    
    ha_df['HA_Trend'] = np.where(ha_df['HA_Close'] > ha_df['HA_Open'], 1, -1)
    
    return ha_df

df = ha(df)
# %%
df = pd.read_csv('VOO.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()
df = df.rename(columns={"Close/Last": 'Close'})
# %%
# %%
# Compute Bollinger Bands
df['SMA'] = df['Close'].rolling(window=20).mean()  # 20-period simple moving average
df['std'] = df['Close'].rolling(window=20).std()  # Standard deviation over 20 periods
df['UpperBand'] = df['SMA'] + (2 * df['std'])  # Upper Bollinger Band
df['LowerBand'] = df['SMA'] - (2 * df['std'])  # Lower Bollinger Band

# Compute MACD: Using 12-period and 26-period EMAs
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()  # 12-period EMA
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()  # 26-period EMA
df['MACD'] = df['EMA12'] - df['EMA26']  # MACD Line
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line

# Define initial cash and portfolio values
initial_cash = 10000
cash = initial_cash
positions = 0
strategy_equity_curve = [initial_cash]  # Portfolio value at the start

# Define Buy and Hold equity curve (buy at start and hold till the end)
buy_and_hold_equity_curve = [initial_cash]
buy_and_hold_positions = initial_cash / df['Close'].iloc[0]  # Buy at the first price

# Define risk management parameters
stop_loss_pct = 0.02  # 2% stop loss
take_profit_pct = 0.05  # 5% take profit

# Backtest the strategy
for i in range(1, len(df)):
    # Mean reversion buy signal (Price hits lower Bollinger Band)
    if df['Close'].iloc[i] < df['LowerBand'].iloc[i] and positions == 0:
        positions = cash / df['Close'].iloc[i]
        cash = 0
    
    # Mean reversion sell signal (Price hits upper Bollinger Band)
    elif df['Close'].iloc[i] > df['UpperBand'].iloc[i] and positions > 0:
        cash = positions * df['Close'].iloc[i]
        positions = 0
    
    # Trend-following buy signal (MACD crosses above Signal)
    elif df['MACD'].iloc[i] > df['Signal'].iloc[i] and positions == 0:
        positions = cash / df['Close'].iloc[i]
        cash = 0
    
    # Trend-following sell signal (MACD crosses below Signal)
    elif df['MACD'].iloc[i] < df['Signal'].iloc[i] and positions > 0:
        cash = positions * df['Close'].iloc[i]
        positions = 0
    
    # Risk Management - Stop Loss / Take Profit
    if positions > 0:
        current_price = df['Close'].iloc[i]
        entry_price = df['Close'].iloc[i - 1]
        # Stop Loss Check
        if (current_price / entry_price) < (1 - stop_loss_pct):
            cash = positions * current_price
            positions = 0
        # Take Profit Check
        elif (current_price / entry_price) > (1 + take_profit_pct):
            cash = positions * current_price
            positions = 0
    
    # Update portfolio value for strategy
    strategy_equity_curve.append(cash + positions * df['Close'].iloc[i])
    
    # Update Buy and Hold portfolio value
    buy_and_hold_equity_curve.append(buy_and_hold_positions * df['Close'].iloc[i])

# Convert to DataFrame for easier plotting
strategy_equity_curve = pd.Series(strategy_equity_curve, index=df.index)
buy_and_hold_equity_curve = pd.Series(buy_and_hold_equity_curve, index=df.index)

# Plot the equity curves for both strategies
plt.figure(figsize=(10, 6))
plt.plot(strategy_equity_curve, label='Mean Reversion + MACD Strategy', color='blue')
plt.plot(buy_and_hold_equity_curve, label='Buy and Hold Strategy', color='green')
plt.title('Equity Curve: Mean Reversion + MACD Strategy vs Buy and Hold Strategy')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()