"""Tests for text chunking and document-to-chunk conversion."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models import SourceDocument
from src.rag import chunk_document, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_becomes_single_chunk():
    chunks = chunk_text("short text", chunk_size=800, overlap=100)
    assert chunks == ["short text"]


def test_consecutive_chunks_share_overlap():
    """The tail of one chunk must reappear at the head of the next."""
    chunks = chunk_text("A" * 1000, chunk_size=400, overlap=100)
    assert len(chunks) > 1
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunks_cover_entire_text():
    text = "".join(str(i % 10) for i in range(2000))
    joined = "".join(chunk_text(text, chunk_size=300, overlap=50))
    assert text[:100] in joined
    assert text[-100:] in joined


def test_zero_chunk_size_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("some text", chunk_size=0, overlap=0)


def test_negative_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=100, overlap=-1)


def test_overlap_equal_to_chunk_size_rejected():
    """Would advance zero characters per iteration and loop forever."""
    with pytest.raises(ValueError, match="less than"):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_overlap_greater_than_chunk_size_rejected():
    with pytest.raises(ValueError, match="less than"):
        chunk_text("some text", chunk_size=100, overlap=150)


def _document(text: str) -> SourceDocument:
    return SourceDocument(
        document_id="doc-abc",
        ticker="AAPL",
        source_type="filing",
        title="Apple Inc. 10-Q — Risk Factors",
        text=text,
        url="https://sec.gov/example",
        published_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
        filing_form="10-Q",
        accession_number="0000320193-26-000020",
        section="Risk Factors",
    )


def test_chunks_inherit_document_identity():
    """Every chunk must carry its parent document's metadata for citation."""
    chunks = chunk_document(_document("B" * 2000), chunk_size=400, overlap=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == "doc-abc"
        assert chunk.metadata["section"] == "Risk Factors"
        assert chunk.metadata["filing_form"] == "10-Q"
        assert chunk.metadata["accession_number"] == "0000320193-26-000020"


def test_chunk_indices_are_sequential():
    chunks = chunk_document(_document("C" * 2000), chunk_size=400, overlap=100)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique():
    chunks = chunk_document(_document("D" * 3000), chunk_size=400, overlap=100)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_empty_document_produces_no_chunks():
    assert chunk_document(_document("")) == []