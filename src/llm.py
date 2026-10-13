import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load local environment variables if running locally
load_dotenv()

def get_groq_api_key():
    """
    Retrieve Groq API key from environment variables or Streamlit secrets.
    """
    # 1. Try checking Streamlit secrets (for cloud deployment)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    # 2. Try checking standard environment variables (for local .env)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key
        
    return None


def get_llm():
    """
    Configure and return the Groq client.
    """
    api_key = get_groq_api_key()
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables or Streamlit secrets.")
    
    return Groq(api_key=api_key)


def generate_answer(prompt):
    """
    Generate an answer using Groq's active high-performance models.
    """
    client = get_llm()
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided RAG context, blending it smoothly with general knowledge when necessary."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    
    return completion.choices[0].message.content