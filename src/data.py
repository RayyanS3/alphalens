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

def get_news(ticker, limit=5):
    raw_news = yf.Ticker(ticker).news
    headlines = []
    for item in raw_news[:limit]:
        content = item.get("content", {})
        headlines.append({
            "title": content.get("title", ""),
            "summary": content.get("summary", ""),
            "publisher": content.get("provider", {}).get("displayName", ""),
            "url": content.get("canonicalUrl", {}).get("url", ""),
        })

    return headlines