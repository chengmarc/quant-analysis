# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os, re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

USE_GPU = True

data_path = r'C:\Users\marcc\My Drive\Data Extraction\geckoscan-all'


# %% Load and clean dataset


def clean_list(lst: list) -> list:
    lst = [str(item) for item in lst]
    lst = [item.replace(',', '') for item in lst]
    lst = [item for item in lst if item != "-"]
    lst = [item for item in lst if item != "nan"]
    lst = [float(item) for item in lst]
    lst = [item for item in lst if item != 0]
    return lst


def log_transform(lst: list) -> list:
    if USE_GPU == True:
        vector = torch.tensor(lst).cuda()
        log_vector = torch.log10(vector)
        lst = log_vector.tolist()
    else:
        lst = np.log(lst)
    return lst


mcap_sequence, tvol_sequence = [], []
for filename in os.listdir(data_path):

    date = re.search(r'\d{4}-\d{2}-\d{2}', filename).group()
    df = pd.read_csv(os.path.join(data_path, filename))

    mcap = df['MarketCap'].tolist()
    mcap = clean_list(mcap)
    mcap = log_transform(mcap)
    mcap_sequence.append((date, mcap))

    tvol = df['Volume24h'].tolist()
    tvol = clean_list(tvol)
    tvol = log_transform(tvol)
    tvol_sequence.append((date, tvol))


# %% Plot distribution


def plot_distribution(sequence, x: int, y: int):
    num_plots = len(sequence)
    num_cols = 10
    num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 30))

    for i, (date, lst) in enumerate(sequence):
        if i < num_plots:  # Check if there are still plots to be displayed
            row = i // num_cols
            col = i % num_cols
            axs[row, col].hist(lst, bins=30)
            axs[row, col].set_xlim(x)
            axs[row, col].set_ylim(y)
            axs[row, col].set_title(date)

    for i in range(num_plots, num_rows * num_cols):
        axs.flatten()[i].axis('off')

    plt.tight_layout()
    plt.show()


plot_distribution(mcap_sequence, (-2, 14), (0, 900))
plot_distribution(tvol_sequence, (-4, 12), (0, 1400))


# %% Calculate statistics


def calculate_stats(lst: list) -> list:
    mean = np.mean(lst)
    std_dev = np.std(lst)
    quantiles = np.percentile(lst, [10, 25, 50, 75, 90])

    stats = [mean, std_dev, quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]]
    return stats


def calculate_stats_sequence(sequence) -> pd.DataFrame:
    stats_list = []
    for date, lst in sequence:
        stats = calculate_stats(lst)
        stats_list.append([date] + stats)

    df = pd.DataFrame(stats_list, columns = ['date', 'mean', 'std', '10th', '25th', 'median', '75th', '90th'])
    return df


stats_mcap = calculate_stats_sequence(mcap_sequence)
stats_tvol = calculate_stats_sequence(tvol_sequence)


# %% Plot statistics


def plot_stats(dataframe: pd.DataFrame, title: str):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5)) # set size

    # Mean over time
    axs[0].plot(dataframe['date'], dataframe['mean'], label='Mean')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Value')
    axs[0].set_title(f'Mean of {title} over Time')
    axs[0].xaxis.set_major_locator(mdates.MonthLocator())
    axs[0].grid(axis='y', linestyle='--')

    # SD over time
    axs[1].plot(dataframe['date'], dataframe['std'], label='Standard Deviation')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Value')
    axs[1].set_title(f'Standard Deviation of {title} over Time')
    axs[1].xaxis.set_major_locator(mdates.MonthLocator())
    axs[1].grid(axis='y', linestyle='--')

    # Quantiles over time
    axs[2].plot(dataframe['date'], dataframe['10th'], label='10th Percentile')
    axs[2].plot(dataframe['date'], dataframe['25th'], label='25th Percentile')
    axs[2].plot(dataframe['date'], dataframe['median'], label='Median')
    axs[2].plot(dataframe['date'], dataframe['75th'], label='75th Percentile')
    axs[2].plot(dataframe['date'], dataframe['90th'], label='90th Percentile')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Value')
    axs[2].set_title(f'Quantiles of {title} over Time')
    axs[2].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    axs[2].xaxis.set_major_locator(mdates.MonthLocator())
    axs[2].grid(axis='y', linestyle='--')

    plt.show()


plot_stats(stats_mcap, "Market Capitalization")
plot_stats(stats_tvol, "Trading Volume")


# %% Percentile calculation


def calculate_percentile(number: float, lst: list) -> float:
    count = 0
    for i in lst:
        if i <= number:
            count += 1
    percentile = (count / len(lst)) * 100
    return percentile


# %% Create percentile dataframe

def valuation(coin_of_interest: str):

    output = []
    for filename in os.listdir(data_path):

        date = re.search(r'\d{4}-\d{2}-\d{2}', filename).group()
        df = pd.read_csv(os.path.join(data_path, filename))
        row = df[df['Symbol'].str.lower() == coin_of_interest.lower()]

        coin_mcap = row['MarketCap'].to_list()
        coin_mcap = clean_list(coin_mcap)
        coin_mcap = log_transform(coin_mcap)

        if coin_mcap: 
            mcap = df['MarketCap'].tolist()
            mcap = clean_list(mcap)
            mcap = log_transform(mcap)
            percentile_mcap = calculate_percentile(coin_mcap[0], mcap)
        else: 
            percentile_mcap = 0

        coin_tvol = row['Volume24h'].to_list()
        coin_tvol = clean_list(coin_tvol)
        coin_tvol = log_transform(coin_tvol)
        coin_tvol = coin_tvol

        if coin_tvol:
            tvol = df['Volume24h'].tolist()
            tvol = clean_list(tvol)
            tvol = log_transform(tvol)
            percentile_tvol = calculate_percentile(coin_tvol[0], tvol)
        else:
            percentile_tvol = 0

        output.append([date, percentile_mcap, percentile_tvol])

    output = pd.DataFrame(output, columns=['Date', 'MarketCap Percentile', 'Volume24h Percentile'])
    return output


# %% Plot result


def plot_result(df, name):
    plt.subplots(figsize=(20, 5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.plot(df['Date'], df['MarketCap Percentile'], label='MarketCap Percentile')
    plt.plot(df['Date'], df['Volume24h Percentile'], label='Volume24h Percentile')

    plt.xlabel('Date')
    plt.ylabel('Percentile')
    plt.title(f'Percentile Progression of {name} vs Date')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()


# %% Demonstration on BTC and ETH


result = valuation("BTC")
plot_result(result, 'BTC')

result = valuation("PEPE")
plot_result(result, 'PEPE')

