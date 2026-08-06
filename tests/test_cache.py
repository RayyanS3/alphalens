"""Tests for cache key stability."""
from __future__ import annotations

from src.cache import _make_key


def _sample(ticker: str, days_back: int = 21) -> list:
    return []


def test_positional_and_keyword_calls_share_a_key():
    """The same call written two ways must hit the same cache entry."""
    assert _make_key(_sample, ("AAPL", 21), {}) == _make_key(_sample, ("AAPL",), {"days_back": 21})


def test_omitted_default_matches_explicit_default():
    assert _make_key(_sample, ("AAPL",), {}) == _make_key(_sample, ("AAPL", 21), {})


def test_different_arguments_differ():
    assert _make_key(_sample, ("AAPL",), {}) != _make_key(_sample, ("MSFT",), {})
    assert _make_key(_sample, ("AAPL", 7), {}) != _make_key(_sample, ("AAPL", 21), {})