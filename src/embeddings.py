import logging
import voyageai
from __future__ import annotations
from src.config import VOYAGE_API_KEY, EMBED_MODEL, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)
_client = voyageai.Client(api_key=VOYAGE_API_KEY)

def embed_texts(texts: list[str], input_type: str = "document", batch_size: int = EMBED_BATCH_SIZE,) -> list[list[float]]:
    if not texts:
        return []

    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = _client.embed(batch, model=EMBED_MODEL, input_type=input_type)
        all_vectors.extend(result.embeddings)

    logger.info("Embedded %d texts in %d batch(es).", len(texts), -(-len(texts) // batch_size))
    return all_vectors


def embed_query(text: str) -> list[float]:
    result = _client.embed([text], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]