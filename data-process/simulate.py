# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats


file_path = os.path.join(os.path.dirname(script_path), "data-dump")
file_list = [x for x in os.listdir(file_path) if ".csv" in x]


# %% Define checklist
checklist = ["Uni-usd-max(4).csv", "eth-usd-max.csv", "ltc-usd-max.csv", "cgpt-usd-max.csv"]


# %% Calculate percentage change


def calculate_dpct(df: pd.DataFrame):
    observations = (df['price'].pct_change() * 100).tolist()
    observations = [x for x in observations if not np.isnan(x)]

    print("Mean:", np.mean(observations))
    print("Standard Deviation:", np.std(observations))

    return observations


# %% Fit distribution for observations


def fit_dist(rv_list):

    distributions = [stats.norm, stats.cauchy, stats.t, stats.f,
                     stats.alpha, stats.beta, stats.gamma, 
                     stats.chi, stats.chi2]

    best_fit = None
    best_params = None
    best_ks_stat = np.inf

    for distribution in distributions:

        params = distribution.fit(rv_list)
        ks_stat, _ = stats.kstest(rv_list, distribution.cdf, args=params)
        # Perform the Kolmogorov-Smirnov test

        if ks_stat < best_ks_stat:
            best_fit = distribution
            best_params = params
            best_ks_stat = ks_stat

    print("Best fit distribution:", best_fit.name)
    print("Best fit parameters:", best_params)
    print("Kolmogorov-Smirnov statistic:", best_ks_stat)

    return best_fit, best_params


# %% Histogram for observed and fitted distribution


def plot_distribution(observations, savename):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.hist(observations, bins=60, range=(-30, 30), edgecolor='black')
    ax1.grid(color='gray', alpha=0.5)
    ax1.set_title('Histogram of Observed Daily Returns')
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Frequency')

    ax2.hist(dist.rvs(*params, size=10000), bins=60, range=(-30, 30), edgecolor='black')
    ax2.grid(color='gray', alpha=0.5)
    ax2.set_title('Histogram of Simulated Daily Returns')
    ax2.set_xlabel('Daily Return')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(script_path), "plots", f"dist-{savename[:-4]}.png"))


# %% Function for simulation


def simulate_once(dist, params, starting_price, depth):

    price = starting_price
    simulations = [starting_price]
    for i in range(depth):
        change = -100
        while change < -30 or change > 30:
            change = dist.rvs(*params)
        price = price * (1 + change / 100)
        simulations.append(price)

    return simulations


def simulate_100(dist, params, df, savename):
    simulations = []
    plt.figure(figsize=(15, 5))
    for i in tqdm(range(100)):
        simulations.append(simulate_once(dist, params, df['price'].to_list()[-1], 365))

    for series in simulations:
        plt.plot(series, color='grey', alpha=0.1)

    plt.title('One Year Price Simulation')
    plt.xlabel('Days Past')
    plt.ylabel('Price (USD)')
    plt.savefig(os.path.join(os.path.dirname(script_path), "plots", f"sim-{savename[:-4]}.png"))


# %% Execution for a given check list:
for item in checklist:
    df = pd.read_csv(os.path.join(file_path, item))
    observations = calculate_dpct(df)
    dist, params = fit_dist(observations)
    plot_distribution(observations, item)
    simulate_100(dist, params, df, item)


# %% Extra utilities


def simulate_10000(dist, params, df):
    simulations = []
    plt.figure(figsize=(15, 5))
    for i in tqdm(range(10000)):
        simulations.append(simulate_once(dist, params, df['price'].to_list()[-1], 365))
    return simulations


def find_max(simulations: list):
    maxes = []
    for simulation in simulations:
        maxes.append(max(simulation))
    return maxes


# %% Plot distribution for maxes
checklist = ["btc-usd-max.csv"]

for item in checklist:
    df = pd.read_csv(os.path.join(file_path, item))
    observations = calculate_dpct(df)
    dist, params = fit_dist(observations)

    simulations = simulate_10000(dist, params, df)
    maxes = find_max(simulations)
    maxes = np.log10(maxes)

    plt.hist(maxes, bins=50, range=(4.5, 7), color='skyblue', edgecolor='black')
    plt.xlabel('Max Price in a Year')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated Maxes')
    plt.show()

