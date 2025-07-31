# api/schemas.py
from pydantic import BaseModel
from typing import List

class RunRequest(BaseModel):
    question: str
    document_chunks: List[str]

class RunResponse(BaseModel):
    answer: str
