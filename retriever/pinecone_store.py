import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from retriever.embedder import get_embedding

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "document-index")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")

def get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
    return pc.Index(PINECONE_INDEX_NAME)

def store_chunks(chunks, namespace="default"):
    vectors = []
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        vectors.append({
            "id": chunk["chunk_id"],
            "values": embedding,
            "metadata": {
                "source": chunk.get("source", ""),
                "page": chunk.get("page_number", 0),
                "text": chunk["text"]
            }
        })
    get_index().upsert(vectors=vectors, namespace=namespace)

def upload_chunks(chunks_file="document_chunks.json", namespace="default"):
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    store_chunks(chunks, namespace)

def query_chunks(query, top_k=5, namespace="default"):
    query_vector = get_embedding(query)
    results = get_index().query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return results["matches"]
