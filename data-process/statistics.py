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
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


file_path = os.path.join(os.path.dirname(script_path), "data-dump")
file_list = [x for x in os.listdir(file_path) if ".csv" in x]


# %% Load reference
with open(os.path.join(file_path, 'reference.pkl'), 'rb') as file:
    reference = pickle.load(file)

unique_dates = pd.read_csv(os.path.join(file_path, "btc-usd-max.csv"))
unique_dates = pd.to_datetime(unique_dates["snapped_at"]).to_list()


# %% Calcualte statistics


def calculate_stats(lst: list) -> list:
    mean = np.mean(lst)
    std_dev = np.std(lst)
    quantiles = np.percentile(lst, [10, 25, 50, 75, 90])

    stats = [mean, std_dev, quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]]
    return stats


def calculate_stats_sequence(key: str) -> pd.DataFrame:
    stats_list = []
    for single_date in tqdm(unique_dates):
        df = reference[single_date][key]
        if not df.empty:
            stats = calculate_stats(df.iloc[:, 1].to_list())
            stats_list.append([single_date] + stats)

    df = pd.DataFrame(stats_list, columns = ['date', 'mean', 'std', '10th', '25th', 'median', '75th', '90th'])
    return df


stats_mcap = calculate_stats_sequence("df_mcap")
stats_tvol = calculate_stats_sequence("df_tvol")


# %% Plot results


def plot_mean(dataframe: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dataframe['date'], dataframe['mean'], label='Mean')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean (Log 10)')
    ax.set_title(f'Mean of {title} over Time')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(axis='y', linestyle='--')
    plt.show()


def plot_std(dataframe: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dataframe['date'], dataframe['std'], label='Standard Deviation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Standard Deviation (Log 10)')
    ax.set_title(f'Standard Deviation of {title} over Time')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(axis='y', linestyle='--')
    plt.show()


def plot_prctl(dataframe: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dataframe['date'], dataframe['10th'], label='10th Percentile')
    ax.plot(dataframe['date'], dataframe['25th'], label='25th Percentile')
    ax.plot(dataframe['date'], dataframe['median'], label='Median')
    ax.plot(dataframe['date'], dataframe['75th'], label='75th Percentile')
    ax.plot(dataframe['date'], dataframe['90th'], label='90th Percentile')
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentiles (Log 10)')
    ax.set_title(f'Percentiles of {title} over Time')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(axis='y', linestyle='--')
    plt.show()


plot_mean(stats_mcap, "Market Capitalization")
plot_mean(stats_tvol, "Trading Volume")

plot_std(stats_mcap, "Market Capitalization")
plot_std(stats_tvol, "Trading Volume")

plot_prctl(stats_mcap, "Market Capitalization")
plot_prctl(stats_tvol, "Trading Volume")

