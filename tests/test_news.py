"""Tests for news normalization, deduplication, and ordering."""
from __future__ import annotations

from unittest.mock import patch

from src.sources.news import get_news_documents


def _article(headline, url, ts, summary="Summary text.", source="Reuters"):
    """Build a raw Finnhub-shaped article dict."""
    return {
        "headline": headline,
        "url": url,
        "datetime": ts,
        "summary": summary,
        "source": source,
    }


def _patch_raw(articles):
    """Patch the network call so tests use controlled data."""
    return patch("src.sources.news._fetch_raw_news", return_value=articles)


def test_articles_become_source_documents():
    with _patch_raw([_article("Apple beats earnings", "https://a.com/1", 1785000000)]):
        docs = get_news_documents("AAPL")

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_type == "news"
    assert doc.ticker == "AAPL"
    assert doc.title == "Apple beats earnings"
    assert doc.publisher == "Reuters"
    assert doc.url == "https://a.com/1"


def test_summary_is_not_duplicated_in_text():
    """Regression: summary was previously written to two fields and concatenated."""
    with _patch_raw([_article("Headline", "https://a.com/1", 1785000000, summary="Body.")]):
        text = get_news_documents("AAPL")[0].text

    assert text.count("Body.") == 1


def test_duplicate_urls_are_collapsed():
    with _patch_raw([
        _article("Version one", "https://a.com/same", 1785000000),
        _article("Version two", "https://a.com/same", 1785000001),
    ]):
        assert len(get_news_documents("AAPL")) == 1


def test_duplicate_titles_are_collapsed():
    with _patch_raw([
        _article("Same headline", "https://a.com/1", 1785000000),
        _article("same HEADLINE", "https://a.com/2", 1785000001),
    ]):
        assert len(get_news_documents("AAPL")) == 1


def test_articles_without_headlines_are_skipped():
    with _patch_raw([
        _article("", "https://a.com/1", 1785000000),
        _article("Real headline", "https://a.com/2", 1785000001),
    ]):
        docs = get_news_documents("AAPL")

    assert len(docs) == 1
    assert docs[0].title == "Real headline"


def test_articles_sorted_newest_first():
    with _patch_raw([
        _article("Older", "https://a.com/1", 1785000000),
        _article("Newest", "https://a.com/2", 1785999999),
        _article("Middle", "https://a.com/3", 1785500000),
    ]):
        assert [d.title for d in get_news_documents("AAPL")] == ["Newest", "Middle", "Older"]


def test_limit_keeps_newest_not_first_returned():
    """Sorting must happen before truncation, or the limit drops recent news."""
    with _patch_raw([
        _article("Old A", "https://a.com/1", 1785000000),
        _article("Old B", "https://a.com/2", 1785000001),
        _article("Newest", "https://a.com/3", 1785999999),
    ]):
        docs = get_news_documents("AAPL", limit=1)

    assert len(docs) == 1
    assert docs[0].title == "Newest"


def test_missing_timestamp_leaves_published_at_none():
    with _patch_raw([_article("No date", "https://a.com/1", 0)]):
        assert get_news_documents("AAPL")[0].published_at is None


def test_timestamps_are_utc_aware():
    with _patch_raw([_article("Dated", "https://a.com/1", 1785000000)]):
        published = get_news_documents("AAPL")[0].published_at

    assert published is not None
    assert published.tzinfo is not None


def test_empty_response_returns_no_documents():
    with _patch_raw([]):
        assert get_news_documents("AAPL") == []