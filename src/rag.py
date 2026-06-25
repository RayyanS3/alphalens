from __future__ import annotations
import chromadb
import requests
from src.embeddings import embed_texts, embed_query
from src.sources.filings import get_filing_text
from src.sources.news import get_news
from src.llm import client   
import logging


_chroma_client = chromadb.PersistentClient(path="data/chroma")
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return [c for c in chunks if c]


def build_store(ticker: str, documents: list[str]) -> None:
    if not documents:
        return

    collection = _chroma_client.get_or_create_collection(name=f"docs_{ticker}")
    vectors = embed_texts(documents)
    ids = [f"{ticker}_{i}" for i in range(len(documents))]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=documents,
    )


def retrieve(ticker: str, query: str, n_results: int = 5) -> list[str]:
    collection = _chroma_client.get_or_create_collection(name=f"docs_{ticker}")
    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
    )

    return results["documents"][0]


def answer_question(ticker: str, question: str, n_results: int = 5) -> str:
    chunks = retrieve(ticker, question, n_results=n_results)
    if not chunks:
        return "No relevant information found."

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a financial analyst. Answer the question using ONLY the context below, which comes from {ticker}'s SEC filings. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text

def build_full_store(ticker: str) -> int:
    documents: list[str] = []

    try:
        filing_text = get_filing_text(ticker)
        documents.extend(chunk_text(filing_text))
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

    return len(documents)

def store_exists(ticker: str) -> bool:
    collection = _chroma_client.get_or_create_collection(name=f"docs_{ticker}")
    return collection.count() > 0


def ensure_store(ticker: str, rebuild: bool = False) -> int:
    collection = _chroma_client.get_or_create_collection(name=f"docs_{ticker}")

    if rebuild:
        _chroma_client.delete_collection(name=f"docs_{ticker}")
        return build_full_store(ticker)

    if collection.count() == 0:
        return build_full_store(ticker)

    return collection.count()