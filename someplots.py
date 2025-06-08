# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:39:26 2024

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)


import pandas as pd
from yfinance_helper import get_ticker
df_CM = pd.read_csv("https://raw.githubusercontent.com/coinmetrics-io/data/master/csv/btc.csv")
df_YF = get_ticker('BTC-USD')

# %%
df_CM['time'] = pd.to_datetime(df_CM['time'])
df_CM.set_index('time', drop=True, inplace=True) 

df_CM = df_CM[df_CM.index>='2014-09-17']

df = pd.merge(df_CM, df_YF, how='outer', left_index=True, right_index=True)

# %%
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 


df.index=pd.to_datetime(df.index) 
print(plt.style.available)

### Plot The number of block produced daily in the Bitcoin system. 

ax = df['BlkCnt'].plot(color='blue', figsize=(14, 10), linewidth=2, fontsize=16) 
ax.set_xlabel('Date',fontsize=14) 
ax.set_title('The number of blocks produced daily in the Bitcoin system from 2014 to 2024', fontsize=16)

ax.axhline(y=df['BlkCnt'].mean(), color='red')

# %%
#### Plot The number and accumulated number of BTC minted daily in scatter plot and the total BTC minted in line plot
###Check all color maps here https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

ax = df[['IssTotNtv','SplyCur']].plot(figsize=(14, 10), linewidth=2, fontsize=16, subplots=True, legend=False, sharex=True, layout=(2,1))
ax[0][0].set_title('The number of BTC minted daily from 2014 to 2024 (Linear)',fontsize=16)
ax[1][0].set_title('The accumulated number of BTC minted daily from 2014 to 2024 (Linear)',fontsize=16)
ax[0][0].set_ylim([0,15000])

# Add a red vertical line for the second halving date
ax[0][0].axvline('2016-07-09', color='red', linestyle='--')
ax[1][0].axvline('2016-07-09', color='red', linestyle='--')

# Add a red vertical line for the third halving date
ax[0][0].axvline('2020-05-11', color='red', linestyle='--')
ax[1][0].axvline('2020-05-11', color='red', linestyle='--')

# %%

####Calculate the annualized Bitcoin dilution rate 
####More about Rolling average here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
####annualized dilution rate in percentage=100*365*the moving average of the number of generated btc in the past 30 days/total generated btc
df['Annualized_DilutionRate']=100*365*df['IssTotNtv'].rolling(window=30).mean()/df['SplyCur']

### Plot The annualized Bitcoin dilution rate in percentage.
ax = df['Annualized_DilutionRate'].plot(color='blue', figsize=(14, 10), linewidth=2, fontsize=16)

# Specify the x-axis label in your plot
ax.set_xlabel('Date',fontsize=16)

# Specify the title in your plot
ax.set_title('The annualized Bitcoin dilution rate in percentage from 2014 to 2024 (Log)', fontsize=16)
ax.set_yscale('log')

# %%
###calculate the total transaction volume
####calculate daily Revenue 
df['Total daily transaction volume']=df['TxTfrValAdjUSD']+df['Volume']
df['Total daily transaction volume']['2010-07-18':'2013-4-28']=df['TxTfrValAdjUSD']['2010-07-18':'2013-4-28']
df[['Total daily transaction volume','TxTfrValAdjUSD','Volume']]['2010-07-18':'2013-4-28']


###Plot total daily transaction volume
ax = df['Total daily transaction volume']['2010-07-18':].plot(color='blue', figsize=(14, 10), linewidth=2, fontsize=16)

# Specify the x-axis label in your plot
ax.set_xlabel('Date',fontsize=16)

# Specify the title in your plot
ax.set_title('The total daily transaction volume from 2014 to 2024 (Log)', fontsize=16)

# Show plot
ax.set_yscale('log')

# %%
####Calculate the velocity
####More about Rolling average here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
####The velocity in percentage=100*the moving average of the volumn in the past 90 days/total market cap
df['Velocity']=100*df['Total daily transaction volume'].rolling(window=90).mean()/df['CapMrktCurUSD']

### Plot The Bitcoin velocity in percentage
ax = df['Velocity']['2010-07-18':].plot(color='blue', figsize=(14, 10), linewidth=2, fontsize=16)

# Specify the x-axis label in your plot
ax.set_xlabel('Date',fontsize=16)

# Specify the title in your plot
ax.set_title('The Bitcoin velocity from 2014 to 2024 (Linear)', fontsize=16)

# %%
#####plot Miners' revenue in USD: transaction fees and block rewards
ax = df[['FeeTotUSD','IssTotUSD','PriceUSD']]['2010-07-08':].plot(figsize=(14, 10), linewidth=2, fontsize=16,subplots=True,legend=False, sharex=True, layout=(3,1))
ax[0][0].set_title('The transaction fees in USD from 2014 to 2024 (Linear)',fontsize=16)
ax[1][0].set_title('The block rewards in USD from 2014 to 2024 (Linear)',fontsize=16)
ax[2][0].set_title('The bitcoin price in USD from 2014 to 2024 (Linear)',fontsize=16)

# %%
####calculate daily Revenue 
df['The daily revenue of bitcoin miners']=df['IssTotUSD']+df['FeeTotUSD']
###calcualte the accumulated revenue 
df['The accumulated revenue of bitcoin miners']=df['The daily revenue of bitcoin miners'].cumsum()

ax = df[['The daily revenue of bitcoin miners','The accumulated revenue of bitcoin miners']]['2010-07-08': ].plot(figsize=(14, 10), linewidth=2, fontsize=16,subplots=True,legend=False, sharex=True, layout=(2,1))
ax[0][0].set_title('The daily revenue of miners in USD from 2014 to 2024 (Linear)',fontsize=16)
ax[1][0].set_title('The accumulated revenue of miners in USD from 2014 to 2024 (Linear)',fontsize=16)

# %%
###calculate P/E Ratio=MarketCap/365*the moving average of miner's revenue in the past 365 days
df['P/E Ratio']=df['CapMrktCurUSD']/(365*df['The daily revenue of bitcoin miners'])

#####plot the bitcoin price and the PE Ratio
ax = df[['PriceUSD','P/E Ratio']]['2010-07-08': ].plot(figsize=(14, 10), linewidth=2, fontsize=16,subplots=True,legend=False, sharex=True, layout=(2,1))
ax[0][0].set_title('The BTC price in USD from 2010-7-18 to 2020-12-31',fontsize=16)
ax[1][0].set_title('The P/E Ratio from 2010-7-18 to 2020-12-31',fontsize=16)
ax[0][0].set_yscale('log')

# %%
#####plot the bitcion price and the NVT 90 Adjusted Ration
ax = df[['PriceUSD','NVTAdj90']]['2010-07-08': ].plot(figsize=(14, 10), linewidth=2, fontsize=16,subplots=True,legend=False, sharex=True, layout=(2,1))
ax[0][0].set_title('The BTC price in USD from 2010-7-18 to 2020-12-31',fontsize=16)
ax[1][0].set_title('The NVT Ratio from 2010-7-18 to 2020-12-31',fontsize=16)
ax[0][0].set_yscale('log')

# %%
df['Metcalfe']=df['AdrActCnt'].pow(2)
df['PM']=df['PriceUSD']/df['Metcalfe']

fig, ax1 =plt.subplots()
ax1 = df['PriceUSD']['2010-7-18': ].plot(figsize=(14, 10), linewidth=2, fontsize=16, color='blue')
plt.legend(loc='upper left')
ax2=ax1.twinx()
ax2=df['PM']['2010-7-18': ].plot(figsize=(14, 10), linewidth=2, fontsize=16,color='green')
ax1.set_yscale('log')
ax2.set_yscale('log')
plt.title('The BTC price in USD and the PM Ratio from 2010-7-18 to 2020-12-31')
plt.legend(loc='best')