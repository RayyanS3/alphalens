# hello_llm.py — first LLM API call for AlphaLens

import os
from dotenv import load_dotenv
from anthropic import Anthropic

# 1. Load the variables from your .env file into the environment
load_dotenv()

# 2. Read your secret key from the environment (NOT hardcoded)
api_key = os.getenv("ANTHROPIC_API_KEY")

# 3. Create the client — this is your connection to Anthropic's API
client = Anthropic(api_key=api_key)

# 4. Send a message and get a response
response = client.messages.create(
    model="claude-sonnet-4-6",      # which model to use
    max_tokens=200,                  # cap on how long the reply can be
    messages=[
        {"role": "user", "content": "In two sentences, what is a stock?"}
    ],
)

# 5. Print just the text of the reply
print(response.content[0].text)