import sys
from src.sources.prices import get_prices
from src.analysis import add_moving_averages
from src.rag import ensure_store, answer_question


def run(ticker: str) -> None:
    try:
        df = add_moving_averages(get_prices(ticker))
        latest = df.iloc[-1]
        print(f"--- {ticker} ---")
        print(f"Latest close: {latest['Close']:.2f}")
        print(f"5-day MA:     {latest['MA5']:.2f}")
        print(f"20-day MA:    {latest['MA20']:.2f}")
    except ValueError as e:
        print(f"Price data unavailable: {e}")

    print(f"\nPreparing research data for {ticker}...")
    chunk_count = ensure_store(ticker)
    print(f"Knowledge base ready ({chunk_count} chunks).")

    questions = [
        "What are the key risks and challenges facing the company?",
        "What is the recent news and sentiment around the company?",
    ]

    for question in questions:
        print(f"\n=== {question} ===")
        answer = answer_question(ticker, question)
        print(answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    run(ticker)