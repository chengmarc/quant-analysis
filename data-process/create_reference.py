# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


file_path = os.path.join(os.path.dirname(script_path), "data-dump")
file_list = [x for x in os.listdir(file_path) if ".csv" in x]


# %% Preprocess data


def get_symbol(file:str) -> str:
    parts = file[:-4].split("-usd-max")
    symbol = parts[0] + parts[1]
    return symbol


def preprocess(df:pd.DataFrame, symbol:str) -> pd.DataFrame:
    df['snapped_at'] = pd.to_datetime(df['snapped_at'])
    df.insert(0, 'symbol', symbol)
    return df


all_data = []

for file in tqdm(file_list):
    symbol = get_symbol(file)
    df = pd.read_csv(os.path.join(file_path, file))
    all_data.append(preprocess(df, symbol))

all_data = pd.concat(all_data)


# %% Select unique dates
unique_dates = pd.read_csv(os.path.join(file_path, "btc-usd-max.csv"))
unique_dates = pd.to_datetime(unique_dates["snapped_at"]).to_list()


# %% Create reference dictionary
mcap_data = all_data[["snapped_at", "market_cap"]]
mcap_data = mcap_data[mcap_data["market_cap"] > 0]
mcap_data = mcap_data.dropna()
mcap_data['market_cap'] = np.log10(mcap_data['market_cap'])

tvol_data = all_data[["snapped_at", "total_volume"]]
tvol_data = tvol_data[tvol_data["total_volume"] > 0]
tvol_data = tvol_data.dropna()
tvol_data['total_volume'] = np.log10(tvol_data['total_volume'])

pair_data = all_data[["snapped_at", "symbol", "market_cap", "total_volume"]]
pair_data = pair_data[pair_data["market_cap"] > 0]
pair_data = pair_data[pair_data["total_volume"] > 0]
tvol_data = tvol_data.dropna()
pair_data['market_cap'] = np.log10(pair_data['market_cap'])
pair_data['total_volume'] = np.log10(pair_data['total_volume'])

dictionary = {}
for single_date in tqdm(unique_dates):
    df1 = mcap_data[mcap_data['snapped_at'] == single_date]
    df2 = tvol_data[tvol_data['snapped_at'] == single_date]
    df3 = pair_data[pair_data['snapped_at'] == single_date]
    dictionary[single_date] = {"df_mcap": df1, 'df_tvol': df2, 'df_pair': df3}


# %% Save reference diticonary
with open(os.path.join(file_path, 'reference.pkl'), 'wb') as file:
    pickle.dump(dictionary, file)

