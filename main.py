import sys
from src.data import get_prices, get_news
from src.analysis import add_moving_averages
from src.llm import analyze_sentiment


def run(ticker):
    try:
        df = get_prices(ticker)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    df = add_moving_averages(df)

    latest = df.iloc[-1]
    print(f"--- {ticker} ---")
    print(f"Latest close: {latest['Close']:.2f}")
    print(f"5-day MA:     {latest['MA5']:.2f}")
    print(f"20-day MA:    {latest['MA20']:.2f}")

    news = get_news(ticker)
    if not news:
        print("No news found.")
        return
    for item in news:
        title = item.get("title")
        if not title:
            continue
    
    result = analyze_sentiment(title)
    print(f"\n[{result['sentiment'].upper()}] {title}")
    print(f"   Source: {item['publisher']}")
    print(f"   Confidence: {result['confidence']} — {result['reasoning']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    run(ticker)