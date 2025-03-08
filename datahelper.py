# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:52:20 2025

@author: Admin
"""
import numpy as np
import pandas as pd
import yfinance as yf


def yf_get(ticker, cutoff, interval):
    
    df = yf.download(ticker, start=cutoff, interval=interval)
    output = pd.DataFrame()
    
    output['Close'] = df['Close'][ticker]
    output['High'] = df['High'][ticker]
    output['Low'] = df['Low'][ticker]
    output['Open'] = df['Open'][ticker]
    output['Volume'] = df['Volume'][ticker]
    
    output["Pct_Change"] = (output["Close"] - output["Open"]) / output["Open"] * 100
    output["Trend"] = output["Pct_Change"].apply(lambda x: 1 if x > 0 else -1)
    
    output['LogClose'] = np.log(df['Close'][ticker])
    output['LogHigh'] = np.log(df['High'][ticker])
    output['LogLow'] = np.log(df['Low'][ticker])
    output['LogOpen'] = np.log(df['Open'][ticker])
    output['Volume'] = df['Volume'][ticker]
    
    return output


def yf_ha(df):
    
    ha_df = df.copy()
    ha_df['HA_Close'] = (ha_df['Open'] + ha_df['High'] + ha_df['Low'] + ha_df['Close']) / 4

    ha_open = [ha_df['Open'].iloc[0]]  # Start with the first open value
    for i in range(1, len(ha_df)):
        ha_open.append((ha_open[-1] + ha_df['HA_Close'].iloc[i-1]) / 2)

    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)

    ha_df["HA_Pct_Change"] = (ha_df["HA_Close"] - ha_df["HA_Open"]) / ha_df["HA_Open"] * 100
    ha_df["HA_Trend"] = ha_df["HA_Pct_Change"].apply(lambda x: 1 if x > 0 else -1)
    
    return ha_df


def cmc_get(ticker, cutoff):

    df = pd.read_csv(f'https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/{ticker}.csv')

    df['Date'] = pd.to_datetime(df['time'])
    df = df[df['Date']>=cutoff]
    df = df[:-1]
    df.set_index('Date', inplace=True)
    
    df['LogPriceUSD'] = np.log(df['PriceUSD'])
    df = df[['PriceUSD', 'LogPriceUSD']]
    df.fillna(0, inplace=True)
    
    df['pct_change'] = df['PriceUSD'].pct_change() + 1

    return df

