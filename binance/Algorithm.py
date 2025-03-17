# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 22:44:30 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd
import numpy as np

import itertools
from tqdm import tqdm
from collections import defaultdict


# %% O(2^n) Super slow, wrote it myself, just for reference

def generate_patterns_1(length):
    
    combinations = list(itertools.product([1, -1], repeat=length))
    return [list(pattern) for pattern in combinations]


def calculate_prob_1(df, pattern):
    
    data = df.copy()
    name = str(pattern).replace(' ', '')
    up = pattern + [1]
    down = pattern + [-1]
    
    data[f'{name}'] = data['trend'].rolling(window=len(pattern)).apply(lambda x: list(x) == pattern, raw=False)
    data[f'{name}_up'] = data['trend'].rolling(window=len(up)).apply(lambda x: list(x) == up, raw=False)
    data[f'{name}_down'] = data['trend'].rolling(window=len(down)).apply(lambda x: list(x) == down, raw=False)
    
    sample_size = data[f'{name}'].sum()
    prob_up = data[f'{name}_up'].sum() / data[f'{name}'].sum()
    prob_down = data[f'{name}_down'].sum() / data[f'{name}'].sum()
    
    result = [pattern, prob_up, prob_down, sample_size]
    return result


def get_prob_matrix_1(df, length=3):
    
    patterns = []
    for i in range(length):
        patterns.extend(generate_patterns_1(i+1))

    prob_matrix = []
    for pattern in tqdm(patterns):
        prob_matrix.append(calculate_prob_1(df, pattern))
        
    prob_matrix = pd.DataFrame(prob_matrix, columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])
    return prob_matrix


# %% O(n^2) Quadratic, wrote by ChatGPT, a bit faster

def generate_patterns_2(length):
    """Generate all possible trend patterns of a given length."""
    return np.array(list(itertools.product([1, -1], repeat=length)))


def calculate_prob_2(df, patterns):
    """Calculate probabilities for all patterns of the same length simultaneously."""
    
    patterns = np.array(patterns)  # Ensure patterns is a NumPy array
    length = patterns.shape[1]  # Length of patterns
    trends = df['trend'].values  # Convert column to NumPy array for speed
    
    if len(trends) < length + 1:
        return []
    
    rolling_matrix = np.lib.stride_tricks.sliding_window_view(trends, length + 1)

    results = []
    for pattern in patterns:
        mask = np.all(rolling_matrix[:, :-1] == pattern, axis=1)
        
        sample_size = mask.sum()
        if sample_size == 0:
            prob_up, prob_down = 0, 0
        else:
            prob_up = np.sum(mask & (rolling_matrix[:, -1] == 1)) / sample_size
            prob_down = np.sum(mask & (rolling_matrix[:, -1] == -1)) / sample_size
        
        results.append([tuple(pattern), prob_up, prob_down, sample_size])
    
    return results


def get_prob_matrix_2(df, length=3):
    
    patterns_dict = {i: generate_patterns_2(i) for i in range(1, length+1)}

    prob_matrix = []
    for length, patterns in tqdm(patterns_dict.items()):
        prob_matrix.extend(calculate_prob_2(df, patterns))
        
    prob_matrix = pd.DataFrame(prob_matrix, columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])
    return prob_matrix


# %% O(n^2) Quadratic, wrote by ChatGPT, use this one

def encode_pattern(pattern):
    """Convert a list pattern into an integer representation (bitwise encoding)."""
    return int("".join(['1' if x == 1 else '0' for x in pattern]), 2)


def generate_patterns_3(length):
    """Generate all possible trend patterns of a given length."""
    return [list(pattern) for pattern in itertools.product([1, -1], repeat=length)]


def calculate_prob_3(df, max_pattern_length=5):
    """Efficiently compute probabilities using hash maps with length-specific keys."""
    
    trends = df['trend'].values  # Convert column to NumPy array for speed
    n = len(trends)
    
    if n < max_pattern_length + 1:
        return pd.DataFrame(columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])  
    
    # Hash maps to store counts (keys are tuples of (encoded_pattern, length))
    pattern_counts = defaultdict(int)
    up_counts = defaultdict(int)
    down_counts = defaultdict(int)

    # Scan through the dataset once
    for i in range(n - max_pattern_length):
        for length in range(1, max_pattern_length + 1):
            pattern = tuple(trends[i:i+length])  # Ensure unique tuple per pattern
            key = (encode_pattern(pattern), length)  # Unique key per length
            
            pattern_counts[key] += 1
            if i + length < n:
                if trends[i+length] == 1:
                    up_counts[key] += 1
                else:
                    down_counts[key] += 1

    # Convert results into a DataFrame
    results = []
    for length in range(1, max_pattern_length + 1):
        for pattern_list in generate_patterns_3(length):
            key = (encode_pattern(pattern_list), length)
            sample_size = pattern_counts[key]
            
            if sample_size == 0:
                prob_up, prob_down = 0, 0
            else:
                prob_up = up_counts[key] / sample_size
                prob_down = down_counts[key] / sample_size
            
            results.append([tuple(pattern_list), prob_up, prob_down, sample_size])
    
    return pd.DataFrame(results, columns=["Pattern", "Up Probability", "Down Probability", "Sample Size"])


def get_prob_matrix_3(df, length=3):
    
    prob_matrix = calculate_prob_3(df, max_pattern_length=length)
    return prob_matrix


# %%

def get_prob_matrix(df, length, complexity='logarithmic'):
    
    if complexity == 'exponential':
        prob_matrix = get_prob_matrix_1(df, length)

    elif complexity == 'quadratic':
        prob_matrix = get_prob_matrix_2(df, length)
        
    elif complexity == 'logarithmic':
        prob_matrix = get_prob_matrix_3(df, length)

    return prob_matrix


def sort_prob_matrix(df, sortby='Win Rate'):
    
    data = df.copy()
    data['Win Rate'] = data[['Up Probability', 'Down Probability']].max(axis=1)
    data['Direction'] = data.apply(lambda row: 'Short' if row['Win Rate'] == row['Down Probability'] else 'Long', axis=1)
    data['Score'] = np.log(data['Sample Size']) * (data['Win Rate'] - 0.5)
    data = data.sort_values(by=sortby, ascending=False)
    data = data.drop(columns = ['Up Probability', 'Down Probability'])
    
    return data

