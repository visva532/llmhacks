from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits a list of documents into semantic chunks.

    Args:
        documents (list[dict]): Each dict must have 'text', 'source', and 'page_number'.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list[dict]: Chunks with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc['text'])
        for i, chunk in enumerate(splits):
            chunks.append({
                "chunk_id": f"{doc['source']}_p{doc['page_number']}_c{i}",
                "text": chunk,
                "page_number": doc["page_number"],
                "source": doc["source"]
            })
    return chunks
