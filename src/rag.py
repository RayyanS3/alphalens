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
    return f"docs_{ticker}"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap

    return [c for c in chunks if c]


def build_store(ticker: str, documents: list[str]) -> None:
    if not documents:
        return

    collection = _chroma_client.get_or_create_collection(name=_collection_name(ticker))
    vectors = embed_texts(documents)
    ids = [f"{ticker}_{i}" for i in range(len(documents))]

    collection.upsert(ids=ids, embeddings=vectors, documents=documents)
    logger.info("Stored %d chunks for %s.", len(documents), ticker)


def retrieve(ticker: str, query: str, n_results: int = RETRIEVAL_RESULTS) -> list[str]:
    collection = _chroma_client.get_or_create_collection(name=_collection_name(ticker))
    query_vector = embed_query(query)

    results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    documents = results.get("documents")

    if not documents or not documents[0]:
        return []
    return documents[0]


def answer_question(ticker: str, question: str, n_results: int = RETRIEVAL_RESULTS) -> str:
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
    collection = _chroma_client.get_or_create_collection(name=_collection_name(ticker))
    return collection.count() > 0


def ensure_store(ticker: str, rebuild: bool = False) -> int:
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