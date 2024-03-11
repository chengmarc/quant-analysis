# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:05:15 2024

@author: Admin
"""
import os, sys, time, datetime, json, getpass

try:
    import pandas as pd
    from requests import Session
    print("SYSTEM: Core modules imported.")
    print("")

except ImportError as e:
    print(f"SYSTEM: The module '{e.name}' is not found, please install it using either pip or conda.")
    getpass.getpass("SYSTEM: Press Enter to quit in a few seconds...")
    sys.exit()

session = Session()


# %% Library


def coin_info(coin: dict) -> list:
    """
    Given a json dictionary represent the information for a specific coin,
    This function will convert it to a list, which represents a row in a dataframe.
    """
    info = []
    for key in coin.keys():
        info.append(coin[key])

    return info


def coin_headers(coin: dict) -> list:
    """
    Given a json dictionary represent the information for a specific coin,
    This function will convert it to a list, which represents the headers in a dataframe.
    """
    headers = list(coin.keys())

    return headers


def get_datetime() -> str:
    """
    This function returns a string that represents the current datetime.

    Return:         a string of the format: %Y-%m-%d_%H-%M-%S
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime


def notice_extracting(page: int) -> None:
    print(f'Extracting page {page}...')


def notice_df_extracted() -> None:
    print("")
    print('Dataframe extracted')


def notice_save_success() -> None:
    print("Dataframe saved")
    getpass.getpass("Press Enter to quit...")
    sys.exit()
    

# %% Execution


"""API Request"""
listing = []
try: 
    for i in range(50):
        time.sleep(2.100)
        url = 'https://api.coingecko.com/api/v3/coins/markets'  
        parameters = {'x_cg_demo_api_key': 'CG-PkV3NeNr54HvgZjoyN8yNjwC', 
                      'vs_currency': 'usd', 'per_page': 250, 'page': i+1}
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        listing.extend(data)
        notice_extracting(i+1)
except:
    pass


"""Extract dataframe"""
df = []
for coin in listing:
    df.append(coin_info(coin))
df = pd.DataFrame(df)

df.columns = coin_headers(coin)
df.rename(columns={'symbol': 'Symbol', 'name': 'Name', 'current_price': 'Price', 
                   'market_cap': 'MarketCap', 'total_volume': 'Volume24h'}, inplace=True)
notice_df_extracted()


"""Save dataframe"""
output_path = r"C:\Users\marcc\My Drive\Data Extraction\geckoscan-all"
output_name = f"all-crypto-{get_datetime()}.csv"
df.to_csv(os.path.join(output_path, output_name))
notice_save_success()

