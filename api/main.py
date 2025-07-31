import os
import requests
import ollama
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from loader.chunker import chunk_document
from retriever.pinecone_store import query_chunks

# Load environment variables
TEAM_TOKEN = os.getenv("TEAM_TOKEN", "hackrx2025securetoken")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
DEFAULT_POLICY_URL = os.getenv("DEFAULT_POLICY_URL")

app = FastAPI()

# Health check for Render
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# Request body format
class HackRxRequest(BaseModel):
    documents: list[str]
    questions: list[str]

# Load default policy at startup
@app.on_event("startup")
def preload_default():
    if DEFAULT_POLICY_URL:
        try:
            r = requests.get(DEFAULT_POLICY_URL)
            if r.status_code == 200:
                with open("default.pdf", "wb") as f:
                    f.write(r.content)
                chunk_document("default.pdf", namespace="default_policy")
                print("[INFO] Default policy loaded and chunked.")
            else:
                print(f"[WARN] Failed to load default policy: {r.status_code}")
        except Exception as e:
            print(f"[ERROR] During default preload: {e}")

# Main endpoint for question answering
@app.post("/hackrx/run")
async def hackrx_run(req: Request, payload: HackRxRequest):
    # Check auth token
    auth_header = req.headers.get("Authorization")
    if auth_header != f"Bearer {TEAM_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Process uploaded documents
    for doc_url in payload.documents:
        try:
            r = requests.get(doc_url)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download {doc_url}")
            with open("temp.pdf", "wb") as f:
                f.write(r.content)
            chunk_document("temp.pdf", namespace=doc_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Document error: {str(e)}")

    # Generate answers
    answers = []
    for q in payload.questions:
        top_chunks = []
        for doc_url in payload.documents:
            top_chunks.extend(query_chunks(q, top_k=3, namespace=doc_url))
        context = "\n".join([m["metadata"]["text"] for m in top_chunks])

        prompt = (
            "Answer the question based only on the following policy content. "
            "Quote exact sentences and include page numbers if available.\n\n"
            f"{context}\n\nQuestion: {q}\nAnswer:"
        )

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise insurance assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response["message"]["content"].strip()
        except Exception as e:
            answer = f"[Error from LLM]: {str(e)}"

        answers.append(answer)

    return {"answers": answers}

# Only for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
