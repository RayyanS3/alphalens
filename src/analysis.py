from __future__ import annotations
import pandas as pd

SHORT_WINDOW = 5
LONG_WINDOW = 20


def add_moving_averages(df: pd.DataFrame, short_window: int = SHORT_WINDOW, long_window: int = LONG_WINDOW) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column.")

    df = df.copy()
    df[f"MA{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"MA{long_window}"] = df["Close"].rolling(window=long_window).mean()
    return df