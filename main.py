from __future__ import annotations
from src.sources.prices import get_prices
from src.analysis import add_moving_averages
from src.rag import ensure_store, answer_question

from src.logging_config import setup_logging
import sys

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


def run(ticker: str) -> None:
    print_price_snapshot(ticker)

    print(f"\nPreparing research data for {ticker}...")
    chunk_count = ensure_store(ticker)
    print(f"Knowledge base ready ({chunk_count} chunks).")

    for question in RESEARCH_QUESTIONS:
        print(f"\n=== {question} ===")
        answer, evidence = answer_question(ticker, question)
        print(answer)

        if evidence:
            print("\nSources:")
            for item in evidence:
                meta = item.metadata
                label = meta.get("filing_form") or meta.get("publisher") or meta.get("source_type", "source")
                date = (meta.get("published_at") or "")[:10]
                url = meta.get("url", "")
                print(f"  [{item.evidence_id}] {label} {date} {url}")


def main() -> None:
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py AAPL")
        sys.exit(1)

    run(sys.argv[1].upper())


if __name__ == "__main__":
    main()