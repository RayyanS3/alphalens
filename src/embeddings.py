from __future__ import annotations
import os
import voyageai
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "voyage-finance-2"

_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    if not texts:
        return []
    result = _client.embed(texts, model=EMBED_MODEL, input_type=input_type)
    return result.embeddings


def embed_query(text: str) -> list[float]:
    result = _client.embed([text], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]