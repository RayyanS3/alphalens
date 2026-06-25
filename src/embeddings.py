"""Text embedding via the Voyage AI finance model.

Wraps the Voyage client to turn text into vectors for semantic retrieval.
Documents and queries are embedded with different input types, as Voyage
optimizes each differently, which improves retrieval quality. Document
embedding is batched to keep individual requests within rate limits.
"""
from __future__ import annotations

import logging

import voyageai

from src.config import VOYAGE_API_KEY, EMBED_MODEL, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)

_client = voyageai.Client(api_key=VOYAGE_API_KEY)


def embed_texts(
    texts: list[str],
    input_type: str = "document",
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of texts into vectors, batched to respect rate limits.

    Args:
        texts: The texts to embed.
        input_type: Voyage input type, "document" for stored content.
        batch_size: Maximum number of texts per API request.

    Returns:
        A list of embedding vectors, one per input text, in order. Returns
        an empty list if given no texts.
    """
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
    """Embed a single search query into one vector.

    Args:
        text: The query text to embed.

    Returns:
        A single embedding vector for the query.
    """
    result = _client.embed([text], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]