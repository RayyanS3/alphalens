# src/analysis.py — computing indicators on price data

def add_moving_averages(df):
    """Add 5-day and 20-day moving averages of the closing price."""
    df = df.copy()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    return df