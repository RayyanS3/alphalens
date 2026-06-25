"""Company news retrieval via the Finnhub API.

Fetches recent ticker-tagged news, deduplicates by headline, and normalizes
each article into a consistent dict shape used throughout the pipeline.
Results are cached on disk (see src.cache) to avoid redundant API calls.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

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
def get_news(
    ticker: str,
    limit: int = NEWS_LIMIT,
    days_back: int = NEWS_DAYS_BACK,
) -> list[dict]:
    """Fetch recent, deduplicated company news for a ticker via Finnhub.

    Args:
        ticker: The stock ticker symbol (e.g. "AAPL").
        limit: Maximum number of articles to return.
        days_back: How many days of history to request.

    Returns:
        A list of normalized article dicts, each with keys: title, summary,
        content, publisher, url, published, category, related.

    Raises:
        ValueError: If the Finnhub key is missing or the API returns an error.
    """
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