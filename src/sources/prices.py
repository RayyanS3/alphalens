from __future__ import annotations

import logging
import re

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_TICKER_PATTERN = re.compile(r"[A-Z]{1,5}")

def valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_PATTERN.fullmatch(ticker))


def get_prices(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    if not valid_ticker(ticker):
        raise ValueError(f"Invalid ticker format: '{ticker}' (expected 1-5 letters).")

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )

    if df is None or df.empty or len(df.dropna()) == 0:
        raise ValueError(f"No price data found for '{ticker}'. It may not be a real ticker.")

    return df