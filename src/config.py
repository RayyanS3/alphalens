"""Centralized configuration for AlphaLens.

All tunable settings, model names, and external endpoints live here so they
can be found and changed in one place rather than scattered across modules.
Secrets (API keys) are NOT stored here — they are read from the environment.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

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
FILINGS_LIMIT: int = 5
FILING_MAX_CHARS: int = 50_000

# --- RAG settings ---
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100
RETRIEVAL_RESULTS: int = 5
EMBED_BATCH_SIZE: int = 50

# --- Cache settings (seconds) ---
CACHE_DIR: str = "data/cache"
CACHE_TTL_NEWS: int = 1_800       # 30 minutes
CACHE_TTL_FILINGS: int = 86_400   # 24 hours

# --- Storage ---
CHROMA_DIR: str = "data/chroma"

# --- External endpoints ---
FINNHUB_NEWS_URL: str = "https://finnhub.io/api/v1/company-news"
SEC_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL: str = "https://data.sec.gov/submissions/CIK{cik}.json"

USEFUL_FORMS: list[str] = ["10-K", "10-Q", "8-K"]