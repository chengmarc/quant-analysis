# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:19:33 2025

@author: Admin
"""

url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
from requests import Request, Session
from bs4 import BeautifulSoup
import pandas as pd

def get_html(url):
    session = Session()
    parameters = {}
    response = session.get(url, params=parameters)

    html = response.text
    return html

def get_table(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    rows = table.find_all("tr")
    data = []

    for row in rows:
        cells = row.find_all(["th", "td"])
        data.append([cell.text.strip() for cell in cells])

    df = pd.DataFrame(data[1:], columns=data[0])  
    return df

def fill_empty(df):
    df["x-coordinate"] = df["x-coordinate"].astype(int)
    df["y-coordinate"] = df["y-coordinate"].astype(int)

    max_x = df["x-coordinate"].max()
    max_y = df["y-coordinate"].max()
    
    all_coords = pd.DataFrame([(x, y) for x in range(max_x+1) for y in range(max_y+1)], columns=["x-coordinate", "y-coordinate"])
    merged_df = pd.merge(all_coords, df, how="left", on=["x-coordinate", "y-coordinate"])
    
    merged_df["Character"] = merged_df["Character"].fillna(" ")
    merged_df = merged_df.sort_values(by=["y-coordinate", "x-coordinate"]).reset_index(drop=True)
    
    return merged_df

def print_df(df):
    pivoted_df = df.pivot(index="y-coordinate", columns="x-coordinate", values="Character")
    pivoted_df = pivoted_df[::-1]

    for _, row in pivoted_df.iterrows():
        print("".join(row))

def main(url):
    html = get_html(url)
    df = get_table(html)
    df = fill_empty(df)
    print_df(df)
    
main(url)
