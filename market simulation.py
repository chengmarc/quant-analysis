# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:55:49 2024

@author: uzcheng
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# %% Price chart preview

df = pd.read_csv(r"C:\Users\marcc\Desktop\btc-usd-max.csv")

plt.plot(range(1, len(df['price']) + 1), df['price'].to_list(), marker = "o")
plt.show()

# %% Histogram for observations
observations = (df['price'].pct_change() * 100).tolist()
observations = [x for x in observations if not np.isnan(x)]

bull = observations[1400: 1750] + observations[2500: 3200]

print("Mean:", np.mean(observations))
print("Standard Deviation:", np.std(observations))

plt.hist(observations, bins=50, range=(-30, 30), edgecolor='black')
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
dist, params = fit_dist(bull)

# %% Histogram for fitted distribution
   
plt.hist(dist.rvs(*params, size=10000), bins=50, range=(-30, 30), edgecolor='black')
plt.show()

# %% 
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

# %%
simulation = simulate_market(dist, params, 70000, 365)
plt.plot(simulation, color='grey', alpha=1)


# %%
simulations = []
for i in tqdm(range(100)):
    simulations.append(simulate_market(dist, params, 70000, 365))
    
# %%
for series in simulations:
    plt.plot(series, color='grey', alpha=0.1)

plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Multiple Series Plot with Transparency')
plt.ylim(0, 1000000)
plt.legend()
plt.show()
