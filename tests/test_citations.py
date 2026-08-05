"""Tests for citation extraction and verification."""
from __future__ import annotations

from src.models import RetrievedEvidence
from src.rag import extract_cited_ids, verify_citations


def _evidence(*ids) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(evidence_id=i, chunk_id=f"c{i}", text="text", metadata={})
        for i in ids
    ]


def test_extracts_cited_ids():
    assert extract_cited_ids("A claim [E1] and another [E3].") == {"E1", "E3"}


def test_repeated_citations_counted_once():
    assert extract_cited_ids("[E1] and [E1] again") == {"E1"}


def test_adjacent_citations_both_found():
    assert extract_cited_ids("Supported [E1][E2].") == {"E1", "E2"}


def test_no_citations_returns_empty():
    assert extract_cited_ids("An uncited claim.") == set()


def test_valid_citations_pass_verification():
    valid, invalid = verify_citations("Claim [E1] and [E2].", _evidence("E1", "E2", "E3"))
    assert valid == {"E1", "E2"}
    assert invalid == set()


def test_hallucinated_citation_detected():
    """The model must not be able to cite evidence it was never given."""
    valid, invalid = verify_citations("Claim [E7].", _evidence("E1", "E2"))
    assert valid == set()
    assert invalid == {"E7"}


def test_mixed_valid_and_hallucinated():
    valid, invalid = verify_citations("[E1] and [E9].", _evidence("E1", "E2"))
    assert valid == {"E1"}
    assert invalid == {"E9"}