"""Retrieval-augmented generation over a ticker's documents.

Handles the full RAG lifecycle: chunking source text, embedding and storing
chunks in a per-ticker ChromaDB collection, retrieving the most relevant
chunks for a query, and generating a grounded answer with the LLM. Each
ticker gets its own collection so retrieval is scoped to one company.
"""
from __future__ import annotations

import logging

import chromadb
import requests

from src.config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_RESULTS,
    LLM_MODEL,
    LLM_MAX_TOKENS,
)
from src.embeddings import embed_texts, embed_query
from src.llm import client
from src.sources.filings import get_filing_text
from src.sources.news import get_news

logger = logging.getLogger(__name__)

_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def _collection_name(ticker: str) -> str:
    """Return the ChromaDB collection name for a ticker."""
    return f"docs_{ticker}"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks suitable for embedding.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks, so ideas
            spanning a boundary survive in both chunks.

    Returns:
        A list of non-empty text chunks. Returns an empty list for empty input.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap

    return [c for c in chunks if c]


def build_store(ticker: str, documents: list[str]) -> None:
    """Embed documents and upsert them into the ticker's collection.

    Args:
        ticker: The stock ticker symbol.
        documents: The text chunks to store.
    """
    if not documents:
        return

    collection = _chroma_client.get_or_create_collection(name=_collection_name(ticker))
    vectors = embed_texts(documents)
    ids = [f"{ticker}_{i}" for i in range(len(documents))]

    collection.upsert(ids=ids, embeddings=vectors, documents=documents)
    logger.info("Stored %d chunks for %s.", len(documents), ticker)


def retrieve(ticker: str, query: str, n_results: int = RETRIEVAL_RESULTS) -> list[str]:
    """Retrieve the most semantically relevant stored chunks for a query.

    Args:
        ticker: The stock ticker symbol.
        query: The search query.
        n_results: Maximum number of chunks to return.

    Returns:
        A list of the most relevant chunk texts, possibly empty.
    """
    collection = _chroma_client.get_or_create_collection(name=_collection_name(ticker))
    query_vector = embed_query(query)

    results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    documents = results.get("documents")

    # ChromaDB returns a list-of-lists (one inner list per query); we sent one.
    if not documents or not documents[0]:
        return []
    return documents[0]


def answer_question(ticker: str, question: str, n_results: int = RETRIEVAL_RESULTS) -> str:
    """Answer a question grounded in the ticker's retrieved documents.

    Args:
        ticker: The stock ticker symbol.
        question: The question to answer.
        n_results: How many chunks to retrieve as context.

    Returns:
        The LLM's grounded answer, or a message if no context is found.
    """
    chunks = retrieve(ticker, question, n_results=n_results)
    if not chunks:
        return "No relevant information found."

    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a financial analyst. Answer the question using ONLY the context below, which comes from {ticker}'s SEC filings and recent news. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def build_full_store(ticker: str) -> int:
    """Build the store for a ticker from both its filings and news.

    Filing and news retrieval each fail independently: if one source is
    unavailable, the other is still stored.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        The number of document chunks stored.
    """
    documents: list[str] = []

    try:
        documents.extend(chunk_text(get_filing_text(ticker)))
    except ValueError as e:
        logger.warning("No filing text for %s: %s", ticker, e)

    try:
        for item in get_news(ticker):
            combined = f"{item['title']}. {item['summary']} {item['content']}".strip()
            documents.extend(chunk_text(combined))
    except (ValueError, requests.RequestException) as e:
        logger.warning("No news for %s: %s", ticker, e)

    if documents:
        build_store(ticker, documents)
    else:
        logger.warning("No documents could be built for %s.", ticker)

    return len(documents)


def store_exists(ticker: str) -> bool:
    """Return True if the ticker already has stored documents."""
    collection = _chroma_client.get_or_create_collection(name=_collection_name(ticker))
    return collection.count() > 0


def ensure_store(ticker: str, rebuild: bool = False) -> int:
    """Build the ticker's store if needed, reusing it otherwise.

    Args:
        ticker: The stock ticker symbol.
        rebuild: If True, delete and rebuild even if a store exists.

    Returns:
        The number of chunks in the store.
    """
    name = _collection_name(ticker)
    collection = _chroma_client.get_or_create_collection(name=name)

    if rebuild:
        logger.info("Rebuilding store for %s.", ticker)
        _chroma_client.delete_collection(name=name)
        return build_full_store(ticker)

    count = collection.count()
    if count == 0:
        return build_full_store(ticker)

    logger.info("Reusing existing store for %s (%d chunks).", ticker, count)
    return count