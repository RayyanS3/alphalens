# AlphaLens

**An LLM-powered equity research tool that produces grounded, source-backed analysis of any public stock.**

AlphaLens gathers market data, news, and SEC filings for a ticker, then uses a Retrieval-Augmented Generation (RAG) pipeline to answer research questions grounded in that real data — not the model's memory. It retrieves the most relevant passages from a company's actual filings and news, then has Claude reason over *those*.

It combines three complementary sources — prices (what the market is doing), news (the current narrative), and SEC filings (authoritative fundamentals) — because no single one is enough. It does **not** predict prices; it synthesizes evidence into a grounded, reasoned view.

> **Status:** In active development. Data pipeline and RAG system are complete. A fine-tuned sentiment model and web frontend are on the roadmap.

---

## How it works

```
python main.py AAPL
        │
        ├─ 1. PRICE SNAPSHOT
        │     yfinance → prices + moving averages
        │
        ├─ 2. BUILD KNOWLEDGE BASE  (cached on disk, reused after)
        │     SEC filings + news → chunk → embed (Voyage) → store in ChromaDB
        │
        └─ 3. ANSWER QUESTIONS
              embed question → retrieve most relevant chunks → Claude answers
              grounded only in the retrieved sources
```

Retrieval is **semantic search** — it finds passages by meaning, not keywords. Ask "what are the risks?" and it surfaces the actual risk disclosures while ignoring boilerplate.

---

## Stack

Python · Anthropic Claude (reasoning) · Voyage AI `voyage-finance-2` (embeddings) · ChromaDB (vector store) · yfinance (prices) · Finnhub (news) · SEC EDGAR via `edgartools` (filings) · pandas

---

## Architecture

```
alphalens/
├── main.py              # CLI entry point
├── src/
│   ├── config.py        # Centralized settings
│   ├── cache.py         # Disk-based caching decorator
│   ├── embeddings.py    # Voyage wrapper
│   ├── llm.py           # Anthropic client
│   ├── analysis.py      # Technical indicators
│   ├── rag.py           # Chunking, retrieval, generation
│   └── sources/         # prices.py · news.py · filings.py
```

Built with centralized config, structured logging, type hints, specific exception handling, disk caching, and swappable data sources behind stable interfaces.

---

## Roadmap

- [x] Data pipeline — prices, news, SEC filings, with caching
- [x] RAG pipeline — embeddings, ChromaDB, semantic retrieval, grounded generation
- [x] Hardening — config, logging, type hints, error handling
- [ ] Fine-tuned sentiment model (QLoRA)
- [ ] Evaluation harness
- [ ] Web frontend

---

*Portfolio/learning project built to a professional standard. Not financial advice.*
