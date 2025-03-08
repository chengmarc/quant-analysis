# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# %%
from datahelper import cmc_get
df = cmc_get('btc', cutoff='2014-09-22')


# %% Histogram for observations
observations = df['pct_change'].tolist()[1:]

print("Mean:", np.mean(observations))
print("Standard Deviation:", np.std(observations))

plt.figure(figsize=(8, 8))
plt.hist(observations, bins=60, range=(0.7, 1.3), edgecolor='black')
plt.title('Histogram of Observed Daily Returns')
plt.xlabel('Daily Return Percentage %')
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
plt.figure(figsize=(8, 8))
plt.hist(dist.rvs(*params, size=10000), bins=60, range=(0.7, 1.3), edgecolor='black')
plt.title('Histogram of Simulated Daily Returns')
plt.xlabel('Daily Return Percentage %')
plt.ylabel('Frequency')
plt.show()


# %% Function for simulation


def simulate_market(dist, params, starting_price, depth):

    price = starting_price
    simulations = [starting_price]
    for i in range(depth):
        change = -1
        while change < 0 or change > 1.3:
            change = dist.rvs(*params)
        price = price * change
        simulations.append(price)

    return simulations


# %% Run simulation

historical_data = df['PriceUSD'].to_list()[-365*4:-1]  # Last 4 years of data
simulated_data = simulate_market(dist, params, df['PriceUSD'].to_list()[-1], 365*2)


plt.figure(figsize=(12, 8))

plt.plot(historical_data, label='Historical Data', color='blue')
plt.plot(range(len(historical_data), len(historical_data) + len(simulated_data)), 
         simulated_data, label='Simulated Data', color='red')
plt.title('Simulated XRP Price Using Markov Process')
plt.xlabel('Days Past')
plt.ylabel('Price (USD)')
plt.legend() or plt.show()


# %% Simulate 100 times
simulations = []
plt.figure(figsize=(15, 5))
for i in tqdm(range(100)):
    simulations.append(simulate_market(dist, params, df['PriceUSD'].to_list()[-1], 365*2))

for series in simulations:
    plt.plot(series, color='black', alpha=0.1)

plt.title('Multiple Series Plot of XRP Price Simulation')
plt.xlabel('Days Past')
plt.ylabel('Price (USD)')
plt.ylim(0, 600000)
plt.show()

