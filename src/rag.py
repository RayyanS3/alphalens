from __future__ import annotations
import chromadb
from src.embeddings import embed_texts, embed_query
from src.llm import client   

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