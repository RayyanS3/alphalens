"""Domain models for AlphaLens.

Defines the structured shapes that flow through the pipeline: fetched source
documents, the chunks derived from them, and the evidence returned by
retrieval. Every chunk and piece of evidence carries the identity of the
document it came from, so any generated claim can be traced back to a source.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

SourceType = Literal["filing", "news"]


@dataclass(frozen=True)
class SourceDocument:
    """A single fetched source: one SEC filing or one news article."""

    document_id: str
    ticker: str
    source_type: SourceType
    title: str
    text: str
    url: str | None = None
    published_at: datetime | None = None

    # Filing-specific
    filing_form: str | None = None
    accession_number: str | None = None
    section: str | None = None

    # News-specific
    publisher: str | None = None

    def to_metadata(self) -> dict:
        """Flatten to a ChromaDB-safe metadata dict (scalars only, no None)."""
        raw = {
            "document_id": self.document_id,
            "ticker": self.ticker,
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "filing_form": self.filing_form,
            "accession_number": self.accession_number,
            "section": self.section,
            "publisher": self.publisher,
        }
        return {k: v for k, v in raw.items() if v is not None}


@dataclass(frozen=True)
class DocumentChunk:
    """A slice of a SourceDocument, retaining that document's identity."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedEvidence:
    """A chunk returned by retrieval, labelled for citation."""

    evidence_id: str          # "E1", "E2", ... assigned per query
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float | None = None

    @property
    def citation_label(self) -> str:
        """A short human-readable source label, e.g. '10-Q Risk Factors 2026-07-31'."""
        meta = self.metadata
        if meta.get("source_type") == "filing":
            parts = [meta.get("filing_form"), meta.get("section")]
        else:
            parts = [meta.get("publisher")]

        date = (meta.get("published_at") or "")[:10]
        parts.append(date or None)
        return " ".join(p for p in parts if p)


def make_document_id(ticker: str, source_type: str, natural_key: str) -> str:
    """Deterministic document ID from a stable natural key.

    Using a content-independent natural key (accession number, article URL)
    means re-fetching the same document produces the same ID, so re-ingestion
    updates rather than duplicates.
    """
    raw = f"{ticker}:{source_type}:{natural_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID from document identity, position, and content."""
    raw = f"{document_id}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]