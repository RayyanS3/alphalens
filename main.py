"""AlphaLens command-line entry point.

Runs the full analysis pipeline for a ticker: a price snapshot, building or
reusing the RAG knowledge base, and generating grounded answers to a set of
research questions. Run with: python main.py <TICKER>
"""
from __future__ import annotations

import sys

from src.logging_config import setup_logging
from src.sources.prices import get_prices
from src.analysis import add_moving_averages
from src.rag import ensure_store, answer_question

RESEARCH_QUESTIONS = [
    "What are the key risks and challenges facing the company?",
    "What is the recent news and sentiment around the company?",
]


def print_price_snapshot(ticker: str) -> None:
    """Print a short price and moving-average snapshot for a ticker.

    Args:
        ticker: The stock ticker symbol.
    """
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


def run(ticker: str) -> None:
    """Run the full AlphaLens analysis pipeline for a ticker.

    Args:
        ticker: The stock ticker symbol (e.g. "AAPL").
    """
    print_price_snapshot(ticker)

    print(f"\nPreparing research data for {ticker}...")
    chunk_count = ensure_store(ticker)
    print(f"Knowledge base ready ({chunk_count} chunks).")

    for question in RESEARCH_QUESTIONS:
        print(f"\n=== {question} ===")
        print(answer_question(ticker, question))


def main() -> None:
    """Parse command-line arguments and run the pipeline."""
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py AAPL")
        sys.exit(1)

    run(sys.argv[1].upper())


if __name__ == "__main__":
    main()