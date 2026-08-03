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

# --- Storage ---
CHROMA_DIR: str = "data/chroma"

# --- External endpoints ---
FINNHUB_NEWS_URL: str = "https://finnhub.io/api/v1/company-news"