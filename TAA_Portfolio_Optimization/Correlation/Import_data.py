import pandas as pd
import yfinance as yf
from tabulate import tabulate

# Step 1: User input tickers
list_tickers = []
n = int(input("How many stocks do you want to enter?: "))
for i in range(n):
    ticker = input(f"Enter Stock {i+1}: ").upper()
    list_tickers.append(ticker)

# Step 2: Download monthly close data (5 years)
data_raw = yf.download(list_tickers, period="5y", interval="1mo")["Close"]
data_raw.dropna(inplace=True)

# Step 3: Add momentum and forward returns
data = data_raw.copy()
for col in list_tickers:
    data[f"pct+1_{col}"] = data[col].pct_change(-1)  # forward 1-month
    data[f"pct-12_{col}"] = data[col].pct_change(12) # 12-month momentum

# Drop NaNs introduced by pct_change
data.dropna(inplace=True)

# Convert index to readable date
data.index = data.index.strftime('%y/%b/%d')

# Step 4: Split train/test safely with .copy()
split = int(0.7 * len(data))
train_set = data.iloc[:split, :].copy()
test_set = data.iloc[split:, :].copy()

# Step 5: Compute correlation of pct-12 vs pct+1
corr = []
valid_tickers = []

for col in list_tickers:
    pct12 = f"pct-12_{col}"
    pct1 = f"pct+1_{col}"
    if pct12 in train_set.columns and pct1 in train_set.columns:
        try:
            val = train_set[[pct12, pct1]].corr().iloc[0, 1]
            if pd.notna(val):
                corr.append(val)
                valid_tickers.append(col)
        except:
            continue

correlation_df = pd.DataFrame({'Correlation': corr}, index=valid_tickers)
correlation_df.sort_values(by="Correlation", ascending=False, inplace=True)

# Step 6: Z-score normalization on pct-12
zscore_cols = [f"pct-12_{col}" for col in valid_tickers]
train_mean = train_set[zscore_cols].mean()
train_std = train_set[zscore_cols].std()

# FIX: No warning — use .loc
train_set.loc[:, zscore_cols] = (train_set[zscore_cols] - train_mean) / train_std
test_set.loc[:, zscore_cols] = (test_set[zscore_cols] - train_mean) / train_std

# Step 7: Signal generation based on Z-score vs median
median_z = train_set[zscore_cols].median()
for col in valid_tickers:
    zcol = f"pct-12_{col}"
    signal_col = f"signal_{col}"
    ret_col = f"strategy_ret_{col}"
    
    # FIX: No warning — use .loc
    test_set.loc[:, signal_col] = 0
    test_set.loc[test_set[zcol] > median_z[zcol], signal_col] = 1
    test_set.loc[test_set[zcol] < median_z[zcol], signal_col] = -1
    
    test_set.loc[:, ret_col] = test_set[signal_col] * test_set[f"pct+1_{col}"]

# Step 8: Tabulate outputs

# 1. Monthly returns (last 10)
print("\n📊 Recent Monthly Returns (last 10 rows):")
print(tabulate(data_raw[valid_tickers].tail(10), headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

# 2. Correlation table
print("\n📈 Correlation between 12M Momentum and 1M Forward Return:")
print(tabulate(correlation_df, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

# 3. Median Z-scores
median_z_df = median_z.to_frame(name="Median Z-Score")
print("\n🧮 Median Z-scores (standardized 12M momentum):")
print(tabulate(median_z_df, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

# 4. Strategy Returns (last 10 rows)
strategy_cols = [f"strategy_ret_{col}" for col in valid_tickers]
print("\n💼 Strategy Returns (last 10 rows):")
print(tabulate(test_set[strategy_cols].tail(10), headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
