# retriever/embedder.py

"""
Embedding utility using SentenceTransformers.
Replaces OpenAI embeddings with a local, open-source model to avoid quota/billing issues.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union

# ✅ Lazy loading — model will be loaded once and reused
_model = None


def get_model() -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    global _model
    if _model is None:
        print("📥 Loading embedding model: all-MiniLM-L6-v2 ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded successfully.")
    return _model


def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a single text.
    
    Args:
        text (str): Input text string.
    
    Returns:
        List[float]: Embedding vector.
    """
    text = text.strip().replace("\n", " ")
    model = get_model()
    return model.encode(text).tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts (batch processing).
    
    Args:
        texts (List[str]): List of input text strings.
    
    Returns:
        List[List[float]]: List of embedding vectors.
    """
    cleaned_texts = [t.strip().replace("\n", " ") for t in texts]
    model = get_model()
    return model.encode(cleaned_texts).tolist()


if __name__ == "__main__":
    # ✅ Quick test
    sample_text = "Hello, this is a test embedding."
    print(f"Embedding length: {len(get_embedding(sample_text))}")
