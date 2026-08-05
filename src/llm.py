from __future__ import annotations

import functools
import json
import logging

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, LLM_MAX_TOKENS, LLM_MODEL

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def get_client() -> Anthropic:
    """Return the shared Anthropic client, created on first use."""
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def analyze_sentiment(headline: str) -> dict:
    prompt = f"""Analyze the sentiment of this financial headline for a stock investor.

    Headline: "{headline}"

    Respond with ONLY a JSON object, no other text, in exactly this format:
    {{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": <number between 0 and 1>,
    "reasoning": "<one short sentence>"
    }}"""

    response = get_client().messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text
    try:
        return json.loads(extract_json(raw_text))
    except json.JSONDecodeError as e:
        logger.warning("Could not parse sentiment JSON for headline '%s': %s", headline, e)
        return {"sentiment": "neutral", "confidence": 0.0, "reasoning": "Could not parse model output."}