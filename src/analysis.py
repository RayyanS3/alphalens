"""Technical indicators computed on price data.

Pure functions that take a price DataFrame and return it enriched with
computed indicator columns. Window lengths are parameterized so callers
can tune them without editing this module.
"""
from __future__ import annotations

import pandas as pd

SHORT_WINDOW = 5
LONG_WINDOW = 20


def add_moving_averages(
    df: pd.DataFrame,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
) -> pd.DataFrame:
    """Add short- and long-window moving averages of the closing price.

    Args:
        df: A price DataFrame containing a "Close" column.
        short_window: Window length for the short moving average.
        long_window: Window length for the long moving average.

    Returns:
        A copy of the input DataFrame with added "MA{short}" and "MA{long}"
        columns. The original DataFrame is not modified.

    Raises:
        KeyError: If the DataFrame has no "Close" column.
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column.")

    df = df.copy()
    df[f"MA{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"MA{long_window}"] = df["Close"].rolling(window=long_window).mean()
    return df