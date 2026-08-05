from __future__ import annotations

import argparse
import sys

from src.analysis import add_moving_averages
from src.config import validate_config
from src.logging_config import setup_logging
from src.rag import answer_question, ensure_store
from src.sources.prices import get_prices

RESEARCH_QUESTIONS = [
    "What are the key risks and challenges facing the company?",
    "What is the recent news and sentiment around the company?",
]


def print_price_snapshot(ticker: str) -> None:
    try:
        df = add_moving_averages(get_prices(ticker))
    except (ValueError, KeyError) as e:
        print(f"Price data unavailable: {e}")
        return

    latest = df.iloc[-1]
    print(f"--- {ticker} ---")
    print(f"Latest close: {latest['Close']:.2f}")
    print(f"5-day MA:     {latest['MA5']:.2f}")
    print(f"20-day MA:    {latest['MA20']:.2f}")


def run(ticker: str, rebuild: bool = False) -> None:
    print_price_snapshot(ticker)

    print(f"\nPreparing research data for {ticker}...")
    chunk_count, manifest = ensure_store(ticker, rebuild=rebuild)

    if manifest:
        print(f"Knowledge base: {chunk_count} chunks, updated {manifest.age_hours():.1f}h ago.")
        if manifest.latest_news_at:
            print(f"News through: {manifest.latest_news_at[:10]}")
    else:
        print(f"Knowledge base ready ({chunk_count} chunks).")

    for question in RESEARCH_QUESTIONS:
        print(f"\n=== {question} ===") 
        answer, evidence = answer_question(ticker, question)
        print(answer)

        if evidence:
            print("\nSources:")
            for ev in evidence:
                metadata = ev.metadata
                label = metadata.get("section") or metadata.get("publisher") or metadata.get("source_type", "source")
                form = metadata.get("filing_form", "")
                date = (metadata.get("published_at") or "")[:10]
                url = metadata.get("url", "")
                parts = [p for p in (f"[{ev.evidence_id}]", form, label, date, url) if p]
                print("  " + " ".join(parts))

def main() -> None:
    setup_logging()

    try:
        validate_config()
    except RuntimeError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="AlphaLens equity research.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--rebuild", action="store_true", help="Force a full rebuild of the knowledge base")
    args = parser.parse_args()

    run(args.ticker.upper(), rebuild=args.rebuild)


if __name__ == "__main__":
    main()