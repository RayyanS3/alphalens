from __future__ import annotations
import os
import voyageai
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "voyage-finance-2"

_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


def embed_texts(texts: list[str], input_type: str = "document", batch_size: int = 50) -> list[list[float]]:
    if not texts:
        return []

    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = _client.embed(batch, model=EMBED_MODEL, input_type=input_type)
        all_vectors.extend(result.embeddings)

    return all_vectors

def embed_query(text: str) -> list[float]:
    result = _client.embed([text], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]