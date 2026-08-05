"""Company news retrieval via the Finnhub API."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from src.cache import cached
from src.config import (
    CACHE_TTL_NEWS,
    FINNHUB_KEY,
    FINNHUB_NEWS_URL,
    NEWS_DAYS_BACK,
    NEWS_LIMIT,
)
from src.models import SourceDocument, make_document_id

logger = logging.getLogger(__name__)
_HTTP_TIMEOUT = 10


@cached(ttl=CACHE_TTL_NEWS)
def _fetch_raw_news(ticker: str, days_back: int = NEWS_DAYS_BACK) -> list[dict]:
    """Fetch raw article dicts from Finnhub. Cached; returns JSON-safe data."""
    if not FINNHUB_KEY:
        raise ValueError("FINNHUB_KEY not found in environment.")

    today = datetime.now(timezone.utc)
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

    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError( # noqa: TRY004
            f"Finnhub returned unexpected response shape for {ticker}: {type(payload).__name__}"
        )
    return payload



def get_news_documents(
    ticker: str,
    limit: int = NEWS_LIMIT,
    days_back: int = NEWS_DAYS_BACK,
) -> list[SourceDocument]:
    """Fetch recent news as structured SourceDocuments, newest first.

    Deduplicates on canonical URL and normalized title, sorts by publication
    time, then applies the limit — so the newest articles are kept, not
    whichever the API happened to return first.
    """
    raw_articles = _fetch_raw_news(ticker, days_back=days_back)

    documents: list[SourceDocument] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for article in raw_articles:
        title = (article.get("headline") or "").strip()
        if not title:
            continue

        url = (article.get("url") or "").strip()
        title_key = title.lower()

        if (url and url in seen_urls) or title_key in seen_titles:
            continue
        if url:
            seen_urls.add(url)
        seen_titles.add(title_key)

        ts = article.get("datetime", 0)
        published_at = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None

        summary = (article.get("summary") or "").strip()

        documents.append(
            SourceDocument(
                document_id=make_document_id(ticker, "news", url or title),
                ticker=ticker,
                source_type="news",
                title=title,
                text=f"{title}. {summary}".strip(),
                url=url or None,
                published_at=published_at,
                publisher=(article.get("source") or "").strip() or None,
            )
        )

    documents.sort(key=lambda d: d.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    documents = documents[:limit]

    logger.info("Fetched %d news documents for %s", len(documents), ticker)
    return documents