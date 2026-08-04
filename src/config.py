from __future__ import annotations
from dotenv import load_dotenv
import os

load_dotenv()

# --- API keys (read from environment, never hardcoded) ---
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
FINNHUB_KEY: str | None = os.getenv("FINNHUB_KEY")
VOYAGE_API_KEY: str | None = os.getenv("VOYAGE_API_KEY")

# --- Identity (SEC requires a contact email in request headers) ---
SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "AlphaLens rayyan.suhail2001@gmail.com")

# --- Models ---
LLM_MODEL: str = "claude-sonnet-4-6"
EMBED_MODEL: str = "voyage-finance-2"

# --- LLM settings ---
LLM_MAX_TOKENS: int = 500

# --- Data source settings ---
NEWS_DAYS_BACK: int = 21
NEWS_LIMIT: int = 10
FILING_MAX_CHARS: int = 400_000

# --- RAG settings ---
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100
RETRIEVAL_RESULTS: int = 5
EMBED_BATCH_SIZE: int = 50

# --- Cache settings (seconds) ---
CACHE_DIR: str = "data/cache"
CACHE_TTL_NEWS: int = 1_800       # 30 minutes

# --- Pipeline versioning ---
PIPELINE_VERSION: str = "2"

# --- Storage ---
CHROMA_DIR: str = "data/chroma"
MANIFEST_DIR: str = "data/manifests"
STORE_MAX_AGE_HOURS: int = 24

# --- External endpoints ---
FINNHUB_NEWS_URL: str = "https://finnhub.io/api/v1/company-news"

# --- Validation ---

REQUIRED_KEYS: dict[str, str | None] = {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "FINNHUB_KEY": FINNHUB_KEY,
    "VOYAGE_API_KEY": VOYAGE_API_KEY,
}


def validate_config() -> None:
    """Verify configuration is usable, raising early with a clear message.

    Called once at startup so misconfiguration fails immediately rather than
    surfacing as a confusing error deep inside an API call.

    Raises:
        RuntimeError: If a required key is missing or a setting is invalid.
    """
    missing = [name for name, value in REQUIRED_KEYS.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your API keys."
        )

    if CHUNK_SIZE <= 0:
        raise RuntimeError(f"CHUNK_SIZE must be positive, got {CHUNK_SIZE}.")
    if CHUNK_OVERLAP < 0:
        raise RuntimeError(f"CHUNK_OVERLAP must be non-negative, got {CHUNK_OVERLAP}.")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise RuntimeError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})."
        )
    if RETRIEVAL_RESULTS <= 0:
        raise RuntimeError(f"RETRIEVAL_RESULTS must be positive, got {RETRIEVAL_RESULTS}.")
    if STORE_MAX_AGE_HOURS <= 0:
        raise RuntimeError(f"STORE_MAX_AGE_HOURS must be positive, got {STORE_MAX_AGE_HOURS}.")