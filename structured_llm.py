import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

headline = "Apple reports record Q4 revenue, beating analyst expectations on strong iPhone sales."

prompt = f"""Analyze the sentiment of this financial headline for a stock investor.

Headline: "{headline}"

Respond with ONLY a JSON object, no other text, in exactly this format:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <number between 0 and 1>,
  "reasoning": "<one short sentence>"
}}"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=300,
    messages=[
        {"role": "user", "content": prompt}
    ],
)

raw_text = response.content[0].text

def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]                      
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]                 
        text = "\n".join(lines).strip()
    return text

clean_text = extract_json(raw_text)
data = json.loads(clean_text)

print("Sentiment: ", data["sentiment"])
print("Confidence:", data["confidence"])
print("Reasoning: ", data["reasoning"])