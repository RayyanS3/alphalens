from unittest.mock import patch

import pytest

from src.rag import StoreNotFoundError, answer_question, retrieve
from src.sources.prices import valid_ticker


def test_missing_store_raises():
    with patch("src.rag._chroma_client") as client:
        client.get_collection.side_effect = Exception("not found")
        with pytest.raises(StoreNotFoundError):
            retrieve("NOSUCH", "any question")


def test_missing_store_returns_no_store_status():
    with patch("src.rag.retrieve", side_effect=StoreNotFoundError("none")):
        result = answer_question("NOSUCH", "any question")
    assert result.status == "no_store"
    assert result.evidence == []


def test_empty_retrieval_returns_no_evidence_status():
    with patch("src.rag.retrieve", return_value=[]):
        result = answer_question("AAPL", "any question")
    assert result.status == "no_evidence"

def test_valid_ticker_accepts_class_shares():
    assert valid_ticker("BRK.B")
    assert valid_ticker("BF-B")