import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

print("--- ENVIRONMENT VARIABLE CHECK ---")
print(f"OPENAI_API_KEY loaded: {'YES' if openai_key else 'NO (Missing or misspelled in .env)'}")
print(f"ANTHROPIC_API_KEY loaded: {'YES' if anthropic_key else 'NO (Missing or misspelled in .env)'}\n")

if openai_key:
    print("Testing OpenAI API...")
    try:
        client = OpenAI(api_key=openai_key)
        response = client.models.list()
        print("OpenAI API Key is VALID and working!\n")
    except Exception as e:
        print(f"OpenAI API Key test FAILED: {e}\n")

if anthropic_key:
    print("Testing Anthropic API...")
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
        print("Anthropic API Key is VALID and working!\n")
    except Exception as e:
        print(f"Anthropic API Key test FAILED: {e}\n")
