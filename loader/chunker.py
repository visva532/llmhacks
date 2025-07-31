# loader/chunker.py

import fitz  # PyMuPDF
import json
from pathlib import Path

# ✅ Import chunk_text from loader/chunks.py
from loader.chunks import chunk_text  

# ✅ Import store_chunks from retriever/pinecone_store.py
from retriever.pinecone_store import upload_chunks  


def chunk_document(file_path, namespace="default", output_file="document_chunks.json"):
    """
    Loads a PDF, extracts text per page, splits into chunks,
    saves them to JSON, and stores embeddings in Pinecone.
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    # 1. Extract text from PDF
    doc = fitz.open(pdf_path)
    documents = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            documents.append({
                "text": text,
                "page_number": page_num + 1,
                "source": pdf_path.name
            })
    doc.close()

    if not documents:
        raise ValueError("No text extracted from PDF.")

    # 2. Chunk text
    chunks = chunk_text(documents, chunk_size=500, chunk_overlap=50)

    # 3. Add chunk IDs
    for idx, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{pdf_path.stem}_chunk_{idx+1}"

    # 4. Save chunks to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(chunks)} chunks to {output_file}")

    # 5. Store embeddings in Pinecone
    upload_chunks(output_file, namespace=namespace)


if __name__ == "__main__":
    # Example usage: Process `sample.pdf` and upload
    chunk_document("sample.pdf", namespace="default")
