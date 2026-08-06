"""Ingestion manifests recording how and when each ticker's store was built.

A manifest lets the application decide whether a stored knowledge base is
still valid: whether it is recent enough, and whether it was built with the
same pipeline settings currently in use. Without this, a store can be reused
indefinitely and silently serve stale or incompatible evidence.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    MANIFEST_DIR,
    PIPELINE_VERSION,
    STORE_MAX_AGE_HOURS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionManifest:
    """A record of one ticker's ingestion run."""

    ticker: str
    built_at: str                  # ISO 8601, UTC
    pipeline_version: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    document_ids: list[str]
    accession_numbers: list[str]
    latest_news_at: str | None
    sections: list[str] = field(default_factory=list)

    def settings_fingerprint(self) -> tuple:
        """The settings that must match for a store to be reusable."""
        return (
            self.pipeline_version,
            self.embedding_model,
            self.chunk_size,
            self.chunk_overlap,
        )

    def age_hours(self) -> float:
        """Hours since this store was built."""
        built = datetime.fromisoformat(self.built_at)
        return (datetime.now(timezone.utc) - built).total_seconds() / 3600


def current_settings_fingerprint() -> tuple:
    """The fingerprint of the settings currently configured."""
    return (PIPELINE_VERSION, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)


def _manifest_path(ticker: str) -> str:
    return os.path.join(MANIFEST_DIR, f"{ticker}.json")


def load_manifest(ticker: str) -> IngestionManifest | None:
    """Load a ticker's manifest, or None if absent or unreadable."""
    path = _manifest_path(ticker)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return IngestionManifest(**json.load(f))
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.warning("Could not read manifest for %s (%s); treating as absent.", ticker, e)
        return None


def save_manifest(manifest: IngestionManifest) -> None:
    """Persist a manifest. Failures are logged but never fatal."""
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    try:
        with open(_manifest_path(manifest.ticker), "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)
    except OSError as e:
        logger.warning("Could not write manifest for %s: %s", manifest.ticker, e)


def build_manifest(
    ticker: str,
    chunk_count: int,
    document_ids: list[str],
    accession_numbers: list[str],
    latest_news_at: datetime | None,
    sections: list[str] | None = None,
) -> IngestionManifest:
    """Create a manifest describing an ingestion that just completed."""
    return IngestionManifest(
        ticker=ticker,
        built_at=datetime.now(timezone.utc).isoformat(),
        pipeline_version=PIPELINE_VERSION,
        embedding_model=EMBED_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunk_count=chunk_count,
        document_ids=document_ids,
        accession_numbers=accession_numbers,
        latest_news_at=latest_news_at.isoformat() if latest_news_at else None,
        sections=sorted(set(sections or [])),
    )


def rebuild_reason(manifest: IngestionManifest | None, max_age_hours: int = STORE_MAX_AGE_HOURS) -> str | None:
    """Return why the store must be rebuilt, or None if it can be reused."""
    if manifest is None:
        return "no manifest found"

    if manifest.settings_fingerprint() != current_settings_fingerprint():
        return (
            f"pipeline settings changed "
            f"(built with {manifest.settings_fingerprint()}, now {current_settings_fingerprint()})"
        )

    age = manifest.age_hours()
    if age > max_age_hours:
        return f"store is {age:.1f}h old (max {max_age_hours}h)"

    return None