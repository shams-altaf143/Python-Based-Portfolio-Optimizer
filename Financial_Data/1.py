
import yfinance as yf

nsei = yf.download("TATASTEEL.NS", period="1y")
print(nsei.head())