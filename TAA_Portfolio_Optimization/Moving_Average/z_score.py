import pandas as pd
import yfinance as yf
from tabulate import tabulate
import yfinance as yf
import pandas as pd
import datetime
from tabulate import tabulate


    
list_tickers=[]

n=int(input("How many stocks do you want to enter:?\n "))


for i in range(n):
    ticker = input(f"Enter Stocks {i+1}: ").upper()
    list_tickers.append(ticker)


# 2. Download data
data = pd.DataFrame()
for ticker in list_tickers:
    df = yf.download(ticker, period="5y", interval="1mo")["Close"]
    data[ticker] = df

# 3. Calculate pct_change, SMA3, SMA12, Momentum
for col in list_tickers:
    data[f"pct_{col}"] = data[col].pct_change(1)
    data[f"SMA3_{col}"] = data[col].rolling(window=3).mean().shift(1)
    data[f"SMA12_{col}"] = data[col].rolling(window=12).mean().shift(1)
    data[f"Momentum_factor_{col}"] = data[f"SMA3_{col}"] - data[f"SMA12_{col}"]

# 4. Split data
split = int(0.7 * len(data))
train_set = data.iloc[:split, :].copy()
test_set = data.iloc[split:, :].copy()

# 5. Normalize with z-score
momentum_cols = [f"Momentum_factor_{col}" for col in list_tickers]
train_mean = train_set[momentum_cols].mean()
train_std = train_set[momentum_cols].std()

train_set[momentum_cols] = (train_set[momentum_cols] - train_mean) / train_std
test_set[momentum_cols] = (test_set[momentum_cols] - train_mean) / train_std

# 6. Compute medians
median = train_set[momentum_cols].median()

# 7. Compute signals and profits
for ticker in list_tickers:
    factor_col = f"Momentum_factor_{ticker}"
    signal_col = f"signal_{ticker}"
    profit_col = f"profit_{ticker}"
    pct_col = f"pct_{ticker}"
    
    test_set[signal_col] = 0  # Default neutral
    test_set.loc[test_set[factor_col] < median[factor_col], signal_col] = -1  # Sell
    test_set.loc[test_set[factor_col] > median[factor_col], signal_col] = 1   # Buy
    # Neutral stays as 0
    test_set[profit_col] = test_set[signal_col].shift(1) * test_set[pct_col]

# 8. Combine Z-score + signals + profits for display
display_cols = []
for ticker in list_tickers:
    display_cols.append(f"Momentum_factor_{ticker}")
    display_cols.append(f"signal_{ticker}")
    display_cols.append(f"profit_{ticker}")

display_df = test_set[display_cols].dropna().copy()
display_df.index = display_df.index.strftime('%d-%m-%y')
display_df.reset_index(inplace=True)
display_df.rename(columns={"index": "Date"}, inplace=True)

# Rename for better headers
for ticker in list_tickers:
    display_df.rename(columns={
        f"Momentum_factor_{ticker}": f"Momentum {ticker}",
        f"signal_{ticker}": f"Signal {ticker}",
        f"profit_{ticker}": f"Profit {ticker}"
    }, inplace=True)

# 9. Final table
print("\n📊 Combined Table: Momentum Z-score, Signal (1=Buy, -1=Sell, 0=Neutral), and Profit\n")
print(tabulate(display_df, headers="keys", tablefmt="fancy_grid", showindex=False))