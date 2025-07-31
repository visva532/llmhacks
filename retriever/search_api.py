from retriever.pinecone_store import query_pinecone

# ✅ Wrapper for Pinecone search — used by API layer (Dev 3)
def get_top_chunks(query: str, top_k: int = 5, namespace: str = "default"):
    """
    Retrieve top matching document chunks from Pinecone.
    Accepts a namespace so different datasets can be queried separately.
    """
    matches = query_pinecone(query, top_k=top_k, namespace=namespace)

    # Format and return clean results
    results = []
    for match in matches:
        metadata = match["metadata"]
        results.append({
            "score": match["score"],
            "chunk_id": metadata.get("chunk_id"),
            "text": metadata.get("text"),
            "source": metadata.get("source"),
            "page": metadata.get("page")
        })
    return results
