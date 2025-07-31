# api/gpt_generator.py

import os
import ollama
from dotenv import load_dotenv

# For OpenAI API
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Read config from .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# OpenAI client (only initialized if needed)
openai_client = None
if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using either Ollama (offline) or OpenAI (online)
    based on .env configuration.
    """
    prompt = f"""
    You are an intelligent assistant for document understanding.
    Answer the question using ONLY the provided context.
    If the answer is missing in the context, reply:
    "The information is not available in the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    try:
        # ✅ If using Ollama (offline & free)
        if LLM_PROVIDER == "ollama":
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()

        # ✅ If using OpenAI API (online)
        elif LLM_PROVIDER == "openai":
            if not openai_client:
                raise ValueError("OpenAI API key missing in .env file")

            response = await openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()

        else:
            raise ValueError(f"Invalid LLM_PROVIDER '{LLM_PROVIDER}' in .env. Use 'ollama' or 'openai'.")

    except Exception as e:
        raise RuntimeError(f"Error generating answer using {LLM_PROVIDER}: {e}")
