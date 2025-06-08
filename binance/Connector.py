# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 22:55:03 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd
from binance.spot import Spot

client = Spot()
exchange_info = client.exchange_info()


# %%
def get_exchange_info():
    
    client = Spot()
    exchange_info = client.exchange_info()    
    return exchange_info

    
def get_active_pairs():
    
    exchange_info = get_exchange_info()
    
    active_pairs = []
    for pairs_info in exchange_info['symbols']:
        active_pairs.append([pairs_info['symbol'], pairs_info['status'], pairs_info['baseAsset'], pairs_info['quoteAsset']])
    active_pairs = pd.DataFrame(active_pairs, columns=['symbol', 'status', 'baseAsset', 'quoteAsset'])
    return active_pairs


def get_market_data(ticker, interval, total_klines):
    
    client = Spot()
    
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
    df["trend"] = df["pct_change"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    return df

