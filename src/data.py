# fetching market data
import re
import yfinance as yf
import os
import requests
from dotenv import load_dotenv
from yfinance import data
from datetime import datetime, timedelta

load_dotenv()

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

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker

def get_news(ticker, limit=10, days_back=21):
    """Fetch recent company news for a ticker via Finnhub."""
    api_key = os.getenv("FINNHUB_KEY")
    if not api_key:
        raise ValueError("FINNHUB_KEY not found in environment.")

    # Finnhub needs a date range in YYYY-MM-DD
    today = datetime.now()
    start = today - timedelta(days=days_back)
    date_from = start.strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")

    response = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": ticker,
            "from": date_from,
            "to": date_to,
            "token": api_key,
        },
        timeout=10,
    )

    if response.status_code != 200:
        raise ValueError(f"Finnhub error: status {response.status_code}")

    raw_articles = response.json()

    headlines = []
    seen_titles = set()

    for article in raw_articles:
        title = article.get("headline", "") or ""
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)

        # Convert Unix timestamp to a readable date
        ts = article.get("datetime", 0)
        published = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""

        headlines.append({
            "title": title,
            "summary": article.get("summary", ""),
            "content": article.get("summary", ""),   # Finnhub gives summary, not full body
            "publisher": article.get("source", ""),
            "url": article.get("url", ""),
            "published": published,
            "category": article.get("category", ""),
            "related": article.get("related", ""),
        })

        if len(headlines) >= limit:
            break

    return headlines