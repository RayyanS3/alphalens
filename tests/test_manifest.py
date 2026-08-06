"""Tests for ingestion manifests and rebuild decision logic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.manifest import (
    IngestionManifest,
    build_manifest,
    load_manifest,
    rebuild_reason,
    save_manifest,
)


def _manifest(**overrides) -> IngestionManifest:
    defaults = {
        "ticker": "AAPL",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2",
        "embedding_model": "voyage-finance-2",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "chunk_count": 91,
        "document_ids": ["doc1", "doc2"],
        "accession_numbers": ["0000320193-26-000020"],
        "latest_news_at": "2026-08-04T12:00:00+00:00",
    }
    return IngestionManifest(**{**defaults, **overrides})


def _aged(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# --- rebuild decisions ---

def test_missing_manifest_forces_rebuild():
    assert rebuild_reason(None) is not None


def test_fresh_matching_manifest_is_reused():
    with patch("src.manifest.current_settings_fingerprint",
               return_value=("2", "voyage-finance-2", 800, 100)):
        assert rebuild_reason(_manifest(), max_age_hours=24) is None


def test_stale_manifest_forces_rebuild():
    with patch("src.manifest.current_settings_fingerprint",
               return_value=("2", "voyage-finance-2", 800, 100)):
        reason = rebuild_reason(_manifest(built_at=_aged(48)), max_age_hours=24)
    assert reason is not None and "old" in reason


def test_changed_chunk_size_forces_rebuild():
    """A store built with different chunk settings is incompatible."""
    with patch("src.manifest.current_settings_fingerprint",
               return_value=("2", "voyage-finance-2", 700, 100)):
        reason = rebuild_reason(_manifest(chunk_size=800), max_age_hours=24)
    assert reason is not None and "settings changed" in reason


def test_changed_embedding_model_forces_rebuild():
    """Vectors from a different model are not comparable."""
    with patch("src.manifest.current_settings_fingerprint",
               return_value=("2", "different-model", 800, 100)):
        reason = rebuild_reason(_manifest(), max_age_hours=24)
    assert reason is not None and "settings changed" in reason


def test_changed_pipeline_version_forces_rebuild():
    with patch("src.manifest.current_settings_fingerprint",
               return_value=("3", "voyage-finance-2", 800, 100)):
        reason = rebuild_reason(_manifest(pipeline_version="2"), max_age_hours=24)
    assert reason is not None and "settings changed" in reason


# --- persistence ---

def test_manifest_round_trips_through_disk(tmp_path):
    with patch("src.manifest.MANIFEST_DIR", str(tmp_path)):
        save_manifest(_manifest())
        loaded = load_manifest("AAPL")

    assert loaded is not None
    assert loaded.ticker == "AAPL"
    assert loaded.chunk_count == 91
    assert loaded.accession_numbers == ["0000320193-26-000020"]


def test_absent_manifest_loads_as_none(tmp_path):
    with patch("src.manifest.MANIFEST_DIR", str(tmp_path)):
        assert load_manifest("NOSUCH") is None


def test_corrupt_manifest_loads_as_none(tmp_path):
    """A damaged manifest must degrade to a rebuild, not crash."""
    (tmp_path / "AAPL.json").write_text("{ not valid json", encoding="utf-8")
    with patch("src.manifest.MANIFEST_DIR", str(tmp_path)):
        assert load_manifest("AAPL") is None


# --- construction ---

def test_build_manifest_records_current_settings():
    m = build_manifest(
        ticker="AAPL",
        chunk_count=91,
        document_ids=["d1"],
        accession_numbers=["acc-1"],
        latest_news_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
    )
    assert m.ticker == "AAPL"
    assert m.chunk_count == 91
    assert m.latest_news_at.startswith("2026-08-04")
    assert m.age_hours() < 0.1


def test_build_manifest_handles_missing_news_date():
    m = build_manifest("AAPL", 5, ["d1"], [], None)
    assert m.latest_news_at is None


def test_naive_built_at_treated_as_utc():
    """A hand-edited manifest must not crash the freshness check."""
    m = _manifest(built_at="2026-08-01T12:00:00")
    assert m.age_hours() > 0