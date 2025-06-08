# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:01:49 2025

@author: Admin
"""

import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd
import time, requests, hmac, hashlib
from tqdm import tqdm


# %%
# Binance API keys
API_KEY = ""
SECRET_KEY = ""

# Binance Futures base URL
v1 = "https://fapi.binance.com/fapi/v1/"
v2 = "https://fapi.binance.com/fapi/v2/"
v3 = "https://fapi.binance.com/fapi/v3/"


def get_signature(params):

    params["timestamp"] = int(time.time() * 1000)
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    
    signature = hmac.new(SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}    
    return params, headers


# %%

def get_account_info(endpoint_version):    
    params, headers = get_signature({})
    return requests.get(endpoint_version + "account", params=params, headers=headers).json()


account_info = get_account_info(v2)
print('Total Balance (USDT):\t\t\t', account_info['totalWalletBalance'])
print('Available Balance (USDT):\t\t', account_info['availableBalance'])
print('Unrealzied Profit/Loss (USDT):\t', account_info['totalCrossUnPnl'])

account_assets = pd.DataFrame(account_info['assets']).sort_values(by="walletBalance", ascending=False)
account_positions = pd.DataFrame(account_info['positions']).sort_values(by="initialMargin", ascending=False)


# %%

def get_exchange_info(endpoint_version):
    params, headers = get_signature({})
    return requests.get(v1 + "exchangeInfo", params=params, headers=headers).json()


exchange_info = get_exchange_info(v1)
exchange_info = pd.DataFrame(exchange_info['symbols'])


# %%

def change_initial_leverage(endpoint_version, symbol, leverage):
    params, headers = get_signature({'symbol': symbol, 'leverage': leverage})
    return requests.post(v1 + "leverage", params=params, headers=headers).json()


for symbol in tqdm(exchange_info['symbol'].tolist()):
    leverage = change_initial_leverage(v1, symbol, 1)


# %%


def get_position_info(endpoint_version):
    params, headers = get_signature({'symbol': 'ETHUSDT'})
    return requests.get(v1 + "apiTradingStatus", params=params, headers=headers).json()


position_info = get_position_info(v1)

