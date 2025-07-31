# retriever/indexer.py

import json
from retriever.pinecone_store import store_chunks  # Use store_chunks directly


def load_chunks(filepath="document_chunks.json"):
    """Load pre-processed chunks from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def index_chunks(namespace="default", chunks_file="document_chunks.json"):
    """Read chunks from file and upload to Pinecone."""
    chunks = load_chunks(chunks_file)
    store_chunks(chunks, namespace=namespace)
    print(f"✅ {len(chunks)} chunks indexed to Pinecone in namespace '{namespace}'.")


if __name__ == "__main__":
    index_chunks()
