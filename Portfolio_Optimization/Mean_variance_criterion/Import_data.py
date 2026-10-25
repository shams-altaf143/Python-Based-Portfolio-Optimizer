import yfinance as yf
import pandas as pd
import datetime
from tabulate import tabulate


    
list_tickers=[]

n=int(input("How many stocks do you want to enter:?\n "))


for i in range(n):
    ticker = input(f"Enter Stocks {i+1}: ").upper()
    list_tickers.append(ticker)


# Importation of data
#list_tickers = [ "META" , "NFLX" , "TSLA" ]
database = yf.download(list_tickers,period="5y",interval="1mo")

 
# Take only the adjusted stock price
# Resample to get last closing price of each year
database= database["Close"]
 
 #'Y' is deprecated and will be removed in a future version, please use 'YE' instead.
 
 
# Drop missing values
data = database.dropna().pct_change().dropna()
 
 #The dropna method is used to remove missing values (NaN) from a DataFrame or Series. It helps clean datasets by dropping rows or columns with null values.
 
# Convert index to year only for readability
data.index = data.index.strftime('%y/%b/%d')


 
print(tabulate(data.tail(10), headers="keys", tablefmt="fancy_grid",floatfmt=".4f"))


