# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from scipy.stats import percentileofscore as percentile


file_path = os.path.join(os.path.dirname(script_path), "data-dump")
file_list = [x for x in os.listdir(file_path) if ".csv" in x]


# %% Load reference
with open(os.path.join(file_path, 'reference.pkl'), 'rb') as file:
    reference = pickle.load(file)

unique_dates = pd.read_csv(os.path.join(file_path, "btc-usd-max.csv"))
unique_dates = pd.to_datetime(unique_dates["snapped_at"]).to_list()


# %% Define checklist
checklist = ["Uni-usd-max(4).csv", "eth-usd-max.csv", "ltc-usd-max.csv", "cgpt-usd-max.csv"]


# %% Preprocess data


def calculate_percentile(number: float, lst: list) -> float:
    if number == -np.inf:
        return 0    
    count = 0
    for i in lst:
        if i <= number:
            count += 1
    percentile = (count / len(lst)) * 100
    return percentile


def preprocess(df:pd.DataFrame) -> pd.DataFrame:
    df["snapped_at"] = pd.to_datetime(df['snapped_at'])
    df = df[df["market_cap"] > 0]
    df = df[df["total_volume"] > 0]
    df = df.dropna()
    df["log_mcap"] = np.log10(df["market_cap"])
    df["log_tvol"] = np.log10(df["total_volume"])
    return df


# %% Calculate percentile of a given coin at each day


def valuate(df:pd.DataFrame) -> pd.DataFrame:

    output = []
    for date in unique_dates:
        row = df[df['snapped_at'] == date]

        if not row.empty:
            market_mcap = reference[date]["df_mcap"].iloc[:, 1].to_list()
            market_tvol = reference[date]["df_tvol"].iloc[:, 1].to_list()

            coin_mcap = row['log_mcap'].to_list()[0]
            coin_tvol = row['log_tvol'].to_list()[0]

            pcnt_mcap = percentile(market_mcap, coin_mcap)
            pcnt_tvol = percentile(market_tvol, coin_tvol)

            output.append([date, coin_mcap, coin_tvol, pcnt_mcap, pcnt_tvol])

    output = pd.DataFrame(output, columns=["snapped_at", "mcap", "tvol", "pcnt_mcap", "pcnt_tvol"])
    return output


valuation = {}
for file_name in tqdm(checklist):
    df = pd.read_csv(os.path.join(file_path, file_name))
    df = preprocess(df)
    if not df.empty:
        valuation[file_name] = valuate(df)

  
# %% Plot results


def plot_results(dataframe: pd.DataFrame, title: str):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    ax1.plot(dataframe['snapped_at'], dataframe['mcap'], label='Market Cap')
    ax1.plot(dataframe['snapped_at'], dataframe['tvol'], label='Total Volume')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Market Data of {title}')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--')

    ax2.plot(dataframe['snapped_at'], dataframe['pcnt_mcap'], label='Market Cap %')
    ax2.plot(dataframe['snapped_at'], dataframe['pcnt_tvol'], label='Total Volume %')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Percentage')
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(script_path), "plots", f"prctl-{title[:-4]}.png"))


for file_name in tqdm(valuation.keys()):
    plot_results(valuation[file_name], file_name)

