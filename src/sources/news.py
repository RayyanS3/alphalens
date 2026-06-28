from __future__ import annotations
from datetime import datetime, timedelta
import logging

import requests
from src.cache import cached
from src.config import (
    FINNHUB_KEY,
    FINNHUB_NEWS_URL,
    NEWS_DAYS_BACK,
    NEWS_LIMIT,
    CACHE_TTL_NEWS,
)

logger = logging.getLogger(__name__)
_HTTP_TIMEOUT = 10

@cached(ttl=CACHE_TTL_NEWS)
def get_news(ticker: str, limit: int = NEWS_LIMIT, days_back: int = NEWS_DAYS_BACK,) -> list[dict]:
    if not FINNHUB_KEY:
        raise ValueError("FINNHUB_KEY not found in environment.")

    today = datetime.now()
    date_from = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")

    try:
        response = requests.get(
            FINNHUB_NEWS_URL,
            params={
                "symbol": ticker,
                "from": date_from,
                "to": date_to,
                "token": FINNHUB_KEY,
            },
            timeout=_HTTP_TIMEOUT,
        )
    except requests.RequestException as e:
        raise ValueError(f"Finnhub request failed for {ticker}: {e}") from e

    if response.status_code != 200:
        raise ValueError(f"Finnhub error for {ticker}: status {response.status_code}")

    raw_articles = response.json()

    headlines: list[dict] = []
    seen_titles: set[str] = set()

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

    logger.info("Fetched %d news articles for %s", len(headlines), ticker)
    return headlines