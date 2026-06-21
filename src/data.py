# fetching market data
import re
import yfinance as yf

def valid_ticker(ticker):
    return bool(re.fullmatch(r"[A-Z]{1,5}", ticker))


def get_prices(ticker, period="3mo", interval="1d"):
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