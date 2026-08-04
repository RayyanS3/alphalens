from __future__ import annotations
import logging
import functools
import voyageai
from src.config import VOYAGE_API_KEY, EMBED_MODEL, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def get_client() -> voyageai.Client:
    """Return the shared Voyage client, created on first use."""
    return voyageai.Client(api_key=VOYAGE_API_KEY)

def embed_texts(texts: list[str], input_type: str = "document", batch_size: int = EMBED_BATCH_SIZE,) -> list[list[float]]:
    if not texts:
        return []

    all_vectors: list[list[float]] = []
    total_batches = -(-len(texts) // batch_size)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info("Embedding batch %d/%d (%d texts)...", batch_num, total_batches, len(batch))
        result = get_client().embed(batch, model=EMBED_MODEL, input_type=input_type)
        all_vectors.extend(result.embeddings)

    logger.info("Embedded %d texts in %d batch(es).", len(texts), total_batches)
    return all_vectors


def embed_query(text: str) -> list[float]:
    result = get_client().embed([text], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]