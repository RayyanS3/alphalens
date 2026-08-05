"""LLM client construction for AlphaLens."""
from __future__ import annotations

import functools
import logging

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_client() -> Anthropic:
    """Return the shared Anthropic client, created on first use."""
    return Anthropic(api_key=ANTHROPIC_API_KEY)