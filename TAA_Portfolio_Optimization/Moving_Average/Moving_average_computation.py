# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 20:14:56 2025

@author: hp
"""

import pandas as pd
from tabulate import tabulate
import datetime
import matplotlib.pyplot as plt

import numpy as np
from Import_data import data,list_tickers,ticker

# Step 5: Calculate SMA15 (15-month moving average) and shift to avoid lookahead bias
for ticker in list_tickers:
    data[f"SMA15_{ticker}"] = data[ticker].rolling(window=15).mean().shift(1)

# Step 6: Combine price and SMA data
combined = pd.concat([data[list_tickers], data[[f"SMA15_{ticker}" for ticker in list_tickers]]], axis=1)

# Step 7: Convert index for better readability
combined.index = pd.to_datetime(combined.index).strftime('%y/%b/%d')


# Step 8: Display last 10 rows in tabular form
print(tabulate(combined.tail(10), headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))



# Plot each stock with its SMA15
for ticker in list_tickers:
    plt.figure(figsize=(10, 5))
    plt.plot(data[ticker], label=f'{ticker} Price', color='blue', linewidth=2)
    plt.plot(data[f'SMA15_{ticker}'], label=f'{ticker} SMA15', color='orange', linestyle='--', linewidth=2)
    
    plt.title(f'{ticker} Price vs SMA15 (15-month Moving Avg)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Price (in ₹ or $)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
