# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:25:24 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


# %%
def prepare_data():

    df = pd.read_csv('https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv')
    df = df[365*2:-1]

    df['Date'] = pd.to_datetime(df['time'])
    df.set_index('Date', inplace=True)
    
    df['LogPriceUSD'] = np.log(df['PriceUSD'])
    df = df[['PriceUSD', 'LogPriceUSD']]
    df.fillna(0, inplace=True)
    
    return df

data = prepare_data()


# %%
def log_fit(x, a, b, c):
    
    return a * np.log(x + c) + b


def get_log_trend(X, Y, get_param=False):
    """
    X : array-like, a series of date indices in integer form.
    Y : array-like, a series of float values corresponding to the dependent variable.
        
    get_param : bool, optional
        If `get_param` is False, returns an array of fitted trend values.
        If `get_param` is True, returns a tuple `(a, b, c)`, which are the parameters of the logarithmic trend.
    """
    
    fitted_param, _ = curve_fit(log_fit, X, Y, maxfev=5000)
    a, b, c = fitted_param
    
    trend = log_fit(X, a, b, c)
    if get_param: return a, b, c
    else: return np.array(trend)


def get_vol_trend(X, Y):
    
    rolling_window=60
    volatility = Y.rolling(rolling_window).std().dropna()
    volatility = volatility.values.reshape(-1, 1)
    
    X = np.arange(len(volatility)).reshape(-1, 1)  # Time as feature
    
    regressor = LinearRegression()
    regressor.fit(X, volatility)        
    trend = regressor.predict(np.arange(len(Y)).reshape(-1, 1))
    
    return trend.flatten()


# %%
def transform_col(data, name):
    
    df = data.copy()
    Y = pd.Series(df[f'{name}'])
    X = (Y.index - Y.index[0]).days

    log_trend = get_log_trend(X, Y, get_param=False)
    residual = np.array(df[f'{name}']) - log_trend
    vol_trend = get_vol_trend(X, Y)
    scale_res = residual / vol_trend
    
    df.loc[:, f'{name}_log_trend'] = log_trend
    df.loc[:, f'{name}_residuals'] = residual
    df.loc[:, f'{name}_scaled_residuals'] = scale_res
    
    return df


def plot_col(df, name):
        
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].set_title("Original Data and Logarithmic Trend")
    axs[1].set_title("Residuals of Logarithmic Trend")
    axs[2].set_title('Scaled Residuals of Logarithmic Trend')

    axs[0].plot(df.index, df[f'{name}'], label='Original Data', color='blue')
    axs[0].plot(df.index, df[f'{name}_log_trend'], label='Logarithmic Trend', color='red', linestyle='--')
    axs[1].plot(df.index, df[f'{name}_residuals'], label='Residuals', color='green')
    axs[2].plot(df.index, df[f'{name}_scaled_residuals'], label='Scaled Residuals')
    
    [ax.grid(True) or ax.legend() for ax in axs]
    _, _ = plt.tight_layout(), plt.show()


for i in tqdm(range(8*365, len(data), 50)):
    df = pd.DataFrame()
    df = transform_col(data[:i], 'LogPriceUSD')
    plot_col(df,'LogPriceUSD')


# %%
def transform_col(data, name, warmup):
    
    df = data.copy()
    Y = pd.Series(df[f'{name}']) 
    X = (Y.index - Y.index[0]).days + 1

    initial_log_trend = get_log_trend(X[:warmup], Y[:warmup])
    initial_residual = np.array(df[f'{name}'][:warmup]) - initial_log_trend
    initial_vol_trend = get_vol_trend(X[:warmup], Y[:warmup])
    initial_scale_res = initial_residual / initial_vol_trend

    for i in tqdm(range(warmup, len(Y))):
        today_log_trend = get_log_trend(X[:i], Y[:i])[-1]
        today_residual = (np.array(Y[:i]) - get_log_trend(X[:i], Y[:i]))[-1]
        today_vol_trend = get_vol_trend(X[:i], Y[:i])[-1]
        today_scale_res = today_residual / today_vol_trend
        
        initial_log_trend = np.append(initial_log_trend, today_log_trend)
        initial_residual = np.append(initial_residual, today_residual)
        initial_vol_trend = np.append(initial_vol_trend, today_vol_trend)
        initial_scale_res = np.append(initial_scale_res, today_scale_res)
        
    df.loc[:, f'{name}_log_trend'] = initial_log_trend
    df.loc[:, f'{name}_residuals'] = initial_residual
    df.loc[:, f'{name}_scaled_residuals'] = initial_scale_res
    
    return df


df = transform_col(data, 'LogPriceUSD', warmup=8*365)
plot_col(df,'LogPriceUSD')


# %%
def generate_signals(df):
    
    df = df.copy()
    df["Signal"] = "Empty"
    df.loc[df["LogPriceUSD_scaled_residuals"] < -5, "Signal"] = "Buy"  # Buy signal
    df.loc[df["LogPriceUSD_scaled_residuals"] > 10, "Signal"] = "Sell"  # Sell signal
    return df

df = generate_signals(df)


# %%
def HODL_strategy(df, start_time, initial_capital):
    
    df = df.copy()[start_time:]
    first_price = df.iloc[0]['PriceUSD']
    btc_held = initial_capital / first_price  
    final_value = btc_held * df.iloc[-1]['PriceUSD']  
    
    equity_curve = [btc_held * price for price in df['PriceUSD']]  
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve

    return output, final_value

df_HODL, final_value_HODL = HODL_strategy(df, start_time=6*365, initial_capital=10000)


# %%
def DCA_strategy(df, start_time, initial_capital, allocation_per_trade):
    
    df = df.copy()[start_time:]
    capital = initial_capital
    position = 0
    equity_curve = []  
    trade_history = []
    
    for i in range(len(df)):
        price = df.iloc[i]['PriceUSD']
        signal = df.iloc[i]['Signal']
        equity = capital + (position * price)

        if signal == 'Buy' and capital > 10:
            amount_to_invest = capital * allocation_per_trade
            btc_to_buy = amount_to_invest / price
            
            if btc_to_buy > 0:
                position += btc_to_buy
                capital -= amount_to_invest  
                trade_history.append({'Type': 'Buy', 'Price': price, 'BTC': btc_to_buy, 'Capital': capital})

        elif signal == 'Sell' and position > 0.001:
            btc_to_sell = position * allocation_per_trade
            
            if btc_to_sell > 0:
                capital += btc_to_sell * price  
                position -= btc_to_sell  
                trade_history.append({'Type': 'Sell', 'Price': price, 'BTC': btc_to_sell, 'Capital': capital})

        equity_curve.append(equity)  
        
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve
    final_value = capital + (position * df.iloc[-1]['PriceUSD'])
    
    return output, trade_history, final_value

df_DCA, trade_history, final_value_DCA = DCA_strategy(df, start_time=6*365, initial_capital=10000, allocation_per_trade=0.05)


# %%
def optimize_allocation(df):
    
    dct = {}
    for year in range(4, 14):
        all_final_values = []
        for i in tqdm(range(1, 100)):
            _, _, final_value = DCA_strategy(df, start_time=year*365, initial_capital=10000, allocation_per_trade=i/100)
            all_final_values.append(final_value)
        dct[f'{year} years'] = all_final_values.copy()
    return dct

#dct = optimize_allocation(df)


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
        axs[1].plot(df_DCA.index, np.log(df_DCA['PriceUSD']), label='BTC Price', color='black', linewidth=2)
        axs[1].scatter(buy_signals.index, np.log(buy_signals['PriceUSD']), label='Buy Signal', color='green', marker='s', s=25)
        axs[1].scatter(sell_signals.index, np.log(sell_signals['PriceUSD']), label='Sell Signal', color='red', marker='s', s=25)
        axs[1].set_title('Logarithmic BTC Price with Buy/Sell Signals')
        axs[1].set_ylabel('Logarithmic BTC Price')
    else:
        axs[1].plot(df_DCA.index, df_DCA['PriceUSD'], label='BTC Price', color='black', linewidth=2)
        axs[1].scatter(buy_signals.index, buy_signals['PriceUSD'], label='Buy Signal', color='green', marker='s', s=25)
        axs[1].scatter(sell_signals.index, sell_signals['PriceUSD'], label='Sell Signal', color='red', marker='s', s=25)
        axs[1].set_title('BTC Price with Buy/Sell Signals')
        axs[1].set_ylabel('Price')

    [ax.grid(True) or ax.legend() for ax in axs]
    _, _ = plt.tight_layout(), plt.show()


plot_results(df_HODL, df_DCA)
plot_results(df_HODL, df_DCA, log=True)

