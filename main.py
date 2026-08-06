from __future__ import annotations

import argparse
import sys

from src.analysis import add_moving_averages
from src.config import validate_config
from src.logging_config import setup_logging
from src.rag import answer_question, ensure_store
from src.sources.prices import get_prices, valid_ticker

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
    """Run the full AlphaLens analysis pipeline for a ticker."""    

    print_price_snapshot(ticker)

    print(f"\nPreparing research data for {ticker}...")
    chunk_count, manifest = ensure_store(ticker, rebuild=rebuild)

    if chunk_count == 0:
        print(f"\nNo research data could be retrieved for {ticker}.")
        print("No SEC filings were found, so this may not be a US-listed company.")
        return

    if manifest:
        print(f"Knowledge base: {chunk_count} chunks, updated {manifest.age_hours():.1f}h ago.")
        if manifest.latest_news_at:
            print(f"News through: {manifest.latest_news_at[:10]}")
    else:
        print(f"Knowledge base ready ({chunk_count} chunks).")

    for question in RESEARCH_QUESTIONS:
        print(f"\n=== {question} ===")
        result = answer_question(ticker, question)

        if result.status == "no_store":
            print(f"No knowledge base available for {ticker}. Run with --rebuild to ingest.")
            continue

        if result.status == "no_evidence":
            print("No relevant evidence found for this question in the ingested sources.")
            continue

        print(result.text)

        if result.evidence:
            print("\nSources:")
            for ev in result.evidence:
                url = ev.metadata.get("url", "")
                print(f"  [{ev.evidence_id}] {ev.citation_label} {url}".rstrip())
        else:
            print("\n(The answer cited no sources.)")

def main() -> None:
    """Parse command-line arguments and run the pipeline."""
    setup_logging()

    try:
        validate_config()
    except RuntimeError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="AlphaLens equity research.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild of the knowledge base",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if not valid_ticker(ticker):
        print(f"Invalid ticker format: {ticker!r}")
        sys.exit(1)

    run(ticker, rebuild=args.rebuild)


if __name__ == "__main__":
    main()