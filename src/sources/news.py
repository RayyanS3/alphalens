import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.cache import cached


load_dotenv()

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"


@cached(ttl=1800)
def get_news(ticker: str, limit: int = 10, days_back: int = 21) -> list[dict]:
    api_key = os.getenv("FINNHUB_KEY")
    if not api_key:
        raise ValueError("FINNHUB_KEY not found in environment.")

    today = datetime.now()
    start = today - timedelta(days=days_back)
    date_from = start.strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")

    response = requests.get(
        FINNHUB_NEWS_URL,
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

        ts = article.get("datetime", 0)
        published = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""

        headlines.append({
            "title": title,
            "summary": article.get("summary", ""),
            "content": article.get("summary", ""),
            "publisher": article.get("source", ""),
            "url": article.get("url", ""),
            "published": published,
            "category": article.get("category", ""),
            "related": article.get("related", ""),
        })

        if len(headlines) >= limit:
            break

    return headlines