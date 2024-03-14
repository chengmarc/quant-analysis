# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# %% Price chart preview
df = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1400000000&period2=2000000000&interval=1d&events=history&includeAdjustedClose=true")

plt.figure(figsize=(15, 5))
plt.plot(df['Date'].to_list(), df['Close'].to_list())
plt.grid(color='gray', alpha=0.5)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Bitcoin Price since September 17th 2014')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()


# %% Histogram for observations
observations = (df['Close'].pct_change() * 100).tolist()
observations = [x for x in observations if not np.isnan(x)]

print("Mean:", np.mean(observations))
print("Standard Deviation:", np.std(observations))

plt.hist(observations, bins=60, range=(-30, 30), edgecolor='black')
plt.grid(color='gray', alpha=0.5)
plt.title('Histogram of Observed Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()


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


dist, params = fit_dist(observations)


# %% Histogram for fitted distribution
plt.hist(dist.rvs(*params, size=10000), bins=60, range=(-30, 30), edgecolor='black')
plt.grid(color='gray', alpha=0.5)
plt.title('Histogram of Simulated Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()


# %% Function for simulation


def simulate_market(dist, params, starting_price, depth):

    price = starting_price
    simulations = [starting_price]
    for i in range(depth):
        change = -100
        while change < -30 or change > 30:
            change = dist.rvs(*params)
        price = price * (1 + change / 100)
        simulations.append(price)

    return simulations


# %% Run simulation
simulation = df['Close'].to_list()[-365*4:-1] + simulate_market(dist, params, df['Close'].to_list()[-1], 365*2)
plt.figure(figsize=(15, 5))
plt.plot(simulation)
plt.grid(color='gray', alpha=0.5)
plt.title('Simulated Bitcoin Price from 2020 to 2026')
plt.xlabel('Days Past')
plt.ylabel('Price (USD)')
plt.show()


# %% Simulate 100 times
simulations = []
plt.figure(figsize=(15, 5))
for i in tqdm(range(100)):
    simulations.append(simulate_market(dist, params, df['Close'].to_list()[-1], 365))

for series in simulations:
    plt.plot(series, color='grey', alpha=0.1)

plt.title('Multiple Series Plot of Bitcoin Price Simulation')
plt.xlabel('Days Past')
plt.ylabel('Price (USD)')
plt.ylim(0, 600000)
plt.show()

