# src/sources/prices.py — market price data via yfinance

import re
import yfinance as yf


def valid_ticker(ticker):
    """Check the ticker is 1-5 uppercase letters."""
    return bool(re.fullmatch(r"[A-Z]{1,5}", ticker))


def get_prices(ticker, period="3mo", interval="1d"):
    """Fetch historical price data for a ticker as a clean DataFrame."""
    if not valid_ticker(ticker):
        raise ValueError("Invalid ticker symbol")

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )

    if df is None or df.empty or len(df.dropna()) == 0:
        raise ValueError(f"No data found for '{ticker}'. It may not be a real ticker.")

    return df


def get_company_name(ticker):
    """Resolve a ticker to its company name."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker