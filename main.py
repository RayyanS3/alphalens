# main.py — AlphaLens entry point: ties the modules together

from src.data import get_prices
from src.analysis import add_moving_averages
from src.llm import analyze_sentiment


def run(ticker):
    # 1. Fetch price data
    df = get_prices(ticker)

    # 2. Compute indicators
    df = add_moving_averages(df)

    # 3. Show the latest price picture
    latest = df.iloc[-1]
    print(f"--- {ticker} ---")
    print(f"Latest close: {latest['Close']:.2f}")
    print(f"5-day MA:     {latest['MA5']:.2f}")
    print(f"20-day MA:    {latest['MA20']:.2f}")

    # 4. Run a sample sentiment analysis (placeholder headline for now)
    headline = f"{ticker} reports record quarterly revenue, beating expectations."
    sentiment = analyze_sentiment(headline)
    print(f"\nHeadline: {headline}")
    print(f"Sentiment:  {sentiment['sentiment']} ({sentiment['confidence']})")
    print(f"Reasoning:  {sentiment['reasoning']}")


if __name__ == "__main__":
    run("AAPL")