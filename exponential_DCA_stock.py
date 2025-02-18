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
from sklearn.linear_model import LinearRegression


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
def fit_trend(df, window_size=52*2):
    
    output = df.copy()
    output['trend'] = np.nan
    output['residual'] = np.nan

    # Iterate over rolling windows
    for i in range(window_size, len(output)):
        
        window_data = output.iloc[:i]
        
        Y = window_data['LogClose'].values.reshape(-1, 1)
        X = np.arange(len(window_data)).reshape(-1, 1)
        
        model = LinearRegression().fit(X, Y)
        
        predicted = model.predict([[len(window_data)]])[0][0]
        
        output.at[output.index[i], 'trend'] = predicted
        output.at[output.index[i], 'residual'] = output.iloc[i]['LogClose'] - predicted
    
    return output


def plot_residual(df):
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(df.index, df['LogClose'], label='Close Price', color='blue')
    axs[0].plot(df.index, df['trend'], label='Trend (Linear Regression)', color='red', linestyle='--')
    axs[0].set_title('Close Price and Trend')
    axs[0].set_ylabel('Price')

    axs[1].plot(df.index, df['residual'], label='Residuals', color='green', alpha=1)
    axs[1].set_title('Residuals')
    axs[1].set_ylabel('Residual')

    [ax.grid(True) or ax.legend() for ax in axs]
    _, _ = plt.tight_layout(), plt.show()


df = fit_trend(df)
plot_residual(df)


# %%
def generate_signals(df):
    
    df = df.copy()
    df["Signal"] = "Empty"
    df.loc[df["residual"] < -0.1, "Signal"] = "Buy"  # Buy signal
    df.loc[df["residual"] > 0, "Signal"] = "Sell"  # Sell signal
    return df

df = generate_signals(df)


# %%
def HODL_strategy(df, start_time, initial_capital):
    
    df = df.copy()[start_time:]
    first_price = df.iloc[0]['Close']
    btc_held = initial_capital / first_price  
    final_value = btc_held * df.iloc[-1]['Close']  
    
    equity_curve = [btc_held * price for price in df['Close']]  
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve

    return output, final_value

df_HODL, final_value_HODL = HODL_strategy(df, start_time=52*2, initial_capital=10000)


# %%
def DCA_strategy(df, start_time, initial_capital, allocation_per_trade):
    
    df = df.copy()[start_time:]
    capital = initial_capital
    position = 0
    equity_curve = []  
    trade_history = []
    
    for i in range(len(df)):
        price = df.iloc[i]['Close']
        signal = df.iloc[i]['Signal']
        equity = capital + (position * price)

        if signal == 'Buy' and capital > 10:
            amount_to_invest = capital * allocation_per_trade
            btc_to_buy = amount_to_invest / price
            
            if btc_to_buy > 0:
                position += btc_to_buy
                capital -= amount_to_invest  
                trade_history.append({'Type': 'Buy', 'Price': price, 'BTC': btc_to_buy, 'Capital': capital})

        elif signal == 'Sell' and position > 1:
            btc_to_sell = position * allocation_per_trade
            
            if btc_to_sell > 0:
                capital += btc_to_sell * price  
                position -= btc_to_sell  
                trade_history.append({'Type': 'Sell', 'Price': price, 'BTC': btc_to_sell, 'Capital': capital})

        equity_curve.append(equity)  
        
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve
    final_value = capital + (position * df.iloc[-1]['Close'])
    
    return output, trade_history, final_value

df_DCA, trade_history, final_value_DCA = DCA_strategy(df, start_time=52*2, initial_capital=10000, allocation_per_trade=0.1)



# %%
def plot_results(df_HODL, df_DCA, log=False):
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Equity Curves
    if log:
        axs[0].plot(df_HODL.index, np.log(df_HODL['Equity']), label='Returns (Buy & Hold)', color='gray', linestyle='dashed', linewidth=2)
        axs[0].plot(df_DCA.index, np.log(df_DCA['Equity']), label='Returns (DCA)', color='blue', linewidth=2)
        axs[0].set_title('Logarithmic Equity Curve Comparison')
        axs[0].set_ylabel('Logarithmic Portfolio Value')
    else:
        axs[0].plot(df_HODL.index, df_HODL['Equity'], label='Returns (Buy & Hold)', color='gray', linestyle='dashed', linewidth=2)
        axs[0].plot(df_DCA.index, df_DCA['Equity'], label='Returns (DCA)', color='blue', linewidth=2)
        axs[0].set_title('Equity Curve Comparison')
        axs[0].set_ylabel('Portfolio Value ($)')

    # Plot 2: BTC Price with Buy/Sell Signals
    buy_signals = df_DCA[df_DCA['Signal'] == 'Buy']
    sell_signals = df_DCA[df_DCA['Signal'] == 'Sell']
    
    if log:
        axs[1].plot(df_DCA.index, np.log(df_DCA['Close']), label='BTC Price', color='black', linewidth=2)
        axs[1].scatter(buy_signals.index, np.log(buy_signals['Close']), label='Buy Signal', color='green', marker='s', s=25)
        axs[1].scatter(sell_signals.index, np.log(sell_signals['Close']), label='Sell Signal', color='red', marker='s', s=25)
        axs[1].set_title('Logarithmic BTC Price with Buy/Sell Signals')
        axs[1].set_ylabel('Logarithmic BTC Price')
    else:
        axs[1].plot(df_DCA.index, df_DCA['Close'], label='BTC Price', color='black', linewidth=2)
        axs[1].scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', color='green', marker='s', s=25)
        axs[1].scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', color='red', marker='s', s=25)
        axs[1].set_title('BTC Price with Buy/Sell Signals')
        axs[1].set_ylabel('Price')

    [ax.grid(True) or ax.legend() for ax in axs]
    _, _ = plt.tight_layout(), plt.show()


plot_results(df_HODL, df_DCA)
plot_results(df_HODL, df_DCA, log=True)

