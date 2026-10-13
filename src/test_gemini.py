
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Hello! Test my Gemini API connection."
        )
        print(response.text)
        break

    except errors.APIError as e:
        print(f"Attempt {attempt + 1} failed: {e}")

        if attempt == 2:
            print("Please try again later or check model availability.")
            raise

        time.sleep(2 ** (attempt + 1))
