# structured_llm.py — getting structured JSON out of an LLM

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
    # Remove ```json ... ``` or ``` ... ``` wrappers if the model added them
    if text.startswith("```"):
        # drop the first line (``` or ```json) and the last line (```)
        lines = text.split("\n")
        lines = lines[1:]                      # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]                 # remove closing fence
        text = "\n".join(lines).strip()
    return text

clean_text = extract_json(raw_text)
data = json.loads(clean_text)   # now parses reliably

print("Sentiment: ", data["sentiment"])
print("Confidence:", data["confidence"])
print("Reasoning: ", data["reasoning"])