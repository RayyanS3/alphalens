"""Market price data via yfinance.

Provides ticker validation, historical price retrieval, and company-name
lookup. All functions are pure data accessors that raise ValueError on
invalid input or missing data rather than returning empty results silently.
"""
from __future__ import annotations

import logging
import re

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_TICKER_PATTERN = re.compile(r"[A-Z]{1,5}")


def valid_ticker(ticker: str) -> bool:
    """Check that a ticker is 1-5 uppercase letters.

    Args:
        ticker: The ticker symbol to validate.

    Returns:
        True if the ticker matches the expected format, else False.
    """
    return bool(_TICKER_PATTERN.fullmatch(ticker))


def get_prices(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical price data for a ticker as a clean DataFrame.

    Args:
        ticker: The stock ticker symbol (e.g. "AAPL").
        period: How far back to fetch (e.g. "1mo", "3mo", "1y").
        interval: Bar size (e.g. "1d", "1wk").

    Returns:
        A DataFrame of OHLCV price data indexed by date.

    Raises:
        ValueError: If the ticker format is invalid or no data is returned.
    """
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


def get_company_name(ticker: str) -> str:
    """Resolve a ticker to its company name, falling back to the ticker itself.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        The company's short or long name, or the ticker if lookup fails.
    """
    try:
        info = yf.Ticker(ticker).info
    except (KeyError, ValueError, ConnectionError) as e:
        logger.warning("Could not resolve company name for %s: %s", ticker, e)
        return ticker

    return info.get("shortName") or info.get("longName") or ticker