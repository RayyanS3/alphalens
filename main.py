import sys
from src.data import get_prices
from src.analysis import add_moving_averages
from src.llm import analyze_sentiment


def run(ticker):
    try:
        df = get_prices(ticker)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    df = get_prices(ticker)
    df = add_moving_averages(df)

    latest = df.iloc[-1]
    print(f"--- {ticker} ---")
    print(f"Latest close: {latest['Close']:.2f}")
    print(f"5-day MA:     {latest['MA5']:.2f}")
    print(f"20-day MA:    {latest['MA20']:.2f}")

    headline = f"{ticker} reports record quarterly revenue, beating expectations."
    sentiment = analyze_sentiment(headline)
    print(f"\nHeadline: {headline}")
    print(f"Sentiment:  {sentiment['sentiment']} ({sentiment['confidence']})")
    print(f"Reasoning:  {sentiment['reasoning']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    run(ticker)