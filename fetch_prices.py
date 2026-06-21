import yfinance as yf
import pandas as pd

ticker = "AAPL"

df = yf.download(
    ticker,
    period="1mo",
    interval="1d",
    auto_adjust=True,
    multi_level_index=False,
)

print("Shape:", df.shape)
print()
print("First 5 rows:")
print(df.head())
print()
print("Columns:", df.columns.tolist())
print()
print("Closing prices:")
print(df["Close"])