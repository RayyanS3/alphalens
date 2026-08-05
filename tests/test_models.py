"""Tests for domain models and deterministic ID generation."""
from __future__ import annotations

from datetime import datetime, timezone

from src.models import SourceDocument, make_chunk_id, make_document_id


def _filing_doc(**overrides) -> SourceDocument:
    defaults = {
        "document_id": "doc1",
        "ticker": "AAPL",
        "source_type": "filing",
        "title": "Apple Inc. 10-Q — Risk Factors",
        "text": "Risk text.",
        "url": "https://sec.gov/example",
        "published_at": datetime(2026, 7, 31, tzinfo=timezone.utc),
        "filing_form": "10-Q",
        "accession_number": "0000320193-26-000020",
        "section": "Risk Factors",
    }
    return SourceDocument(**{**defaults, **overrides})


def test_metadata_contains_filing_identity():
    meta = _filing_doc().to_metadata()
    assert meta["source_type"] == "filing"
    assert meta["filing_form"] == "10-Q"
    assert meta["section"] == "Risk Factors"
    assert meta["accession_number"] == "0000320193-26-000020"


def test_metadata_excludes_none_values():
    """ChromaDB rejects None in metadata, so it must be stripped."""
    meta = _filing_doc(url=None, section=None, publisher=None).to_metadata()
    assert "url" not in meta
    assert "section" not in meta
    assert all(value is not None for value in meta.values())


def test_metadata_values_are_chroma_safe_scalars():
    meta = _filing_doc().to_metadata()
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


def test_published_at_serialized_as_iso_string():
    assert _filing_doc().to_metadata()["published_at"].startswith("2026-07-31")


def test_document_id_is_deterministic():
    a = make_document_id("AAPL", "filing", "0000320193-26-000020")
    b = make_document_id("AAPL", "filing", "0000320193-26-000020")
    assert a == b


def test_document_id_differs_by_natural_key():
    assert make_document_id("AAPL", "filing", "acc-1") != make_document_id("AAPL", "filing", "acc-2")


def test_chunk_id_differs_by_index_and_content():
    base = make_chunk_id("doc1", 0, "same text")
    assert make_chunk_id("doc1", 1, "same text") != base
    assert make_chunk_id("doc1", 0, "other text") != base
    assert make_chunk_id("doc1", 0, "same text") == base