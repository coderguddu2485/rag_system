import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm():
    """
    Configure and return the Groq client.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    
    return Groq(api_key=GROQ_API_KEY)

def generate_answer(prompt):
    """
    Generate an answer using Groq's high-speed open-weight models.
    """
    client = get_llm()
    
    # Using a fast, highly capable free model on Groq
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided RAG context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    
    return completion.choices[0].message.content