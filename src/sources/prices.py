import re
import pandas as pd
import yfinance as yf


def valid_ticker(ticker: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}", ticker))


def get_prices(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    if not valid_ticker(ticker):
        raise ValueError("Invalid ticker symbol")

    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, multi_level_index=False, progress=False,
    )
    if df is None or df.empty or len(df.dropna()) == 0:
        raise ValueError(f"No data found for '{ticker}'. It may not be a real ticker.")
    return df


def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker