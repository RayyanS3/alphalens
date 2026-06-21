# src/data.py — fetching market data

import yfinance as yf


def get_prices(ticker, period="3mo", interval="1d"):
    """Fetch historical price data for a ticker as a clean DataFrame."""
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        multi_level_index=False,
    )
    return df