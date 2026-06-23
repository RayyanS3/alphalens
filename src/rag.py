from __future__ import annotations
import chromadb
from src.embeddings import embed_texts, embed_query

_chroma_client = chromadb.PersistentClient(path="data/chroma")

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