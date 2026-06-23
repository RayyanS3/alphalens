import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    return df