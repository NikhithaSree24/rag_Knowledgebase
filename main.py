import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from .rag_utils import synthesize_answer
from .ingest import build_faiss_index_from_folder


app = FastAPI(title="RAG Knowledge Base API")

class IngestRequest(BaseModel):
    folder: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

@app.post("/ingest")
def ingest(req: IngestRequest):
    folder = req.folder
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail="folder not found")
    try:
        build_faiss_index_from_folder(folder)
        return {"status": "ok", "message": "ingested"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="empty query")
    try:
        resp = synthesize_answer(req.query, top_k=req.top_k)
        return {"status": "ok", "data": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
