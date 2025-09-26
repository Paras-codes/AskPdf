import shutil
import os
from fastapi import FastAPI, UploadFile, File
from typing import List
from ingest import ingest_pdfs
from rag_chain import get_rag_chain, get_vectordb
from models import QueryRequest, AnswerResponse, ListDocsResponse, DeleteDocsResponse, DeleteAllResponse, UploadResponse, DeleteRequest
from config import UPLOAD_DIR

app = FastAPI(title="RAG System with Gemini")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Upload PDFs
# -----------------------------
@app.post("/upload_pdfs", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)

    result = ingest_pdfs(file_paths)
    return UploadResponse(message="PDFs processed and added to DB", details=result)

# -----------------------------
# Ask Question
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QueryRequest):
    qa_chain = get_rag_chain()
    result = qa_chain({"query": req.question})
    return AnswerResponse(answer=result["result"], sources=[doc.metadata for doc in result["source_documents"]])

# -----------------------------
# List Documents
# -----------------------------
@app.get("/list_docs", response_model=ListDocsResponse)
def list_docs():
    vectordb = get_vectordb()
    all_docs = vectordb.get()
    return ListDocsResponse(ids=all_docs["ids"], count=len(all_docs["ids"]))

# -----------------------------
# Delete Specific Documents
# -----------------------------
@app.post("/delete_docs", response_model=DeleteDocsResponse)
def delete_docs(req: DeleteRequest):
    vectordb = get_vectordb()
    vectordb.delete(ids=req.ids)
    return DeleteDocsResponse(message="Documents deleted", deleted_ids=req.ids)

# -----------------------------
# Delete All Documents
# -----------------------------
@app.delete("/delete_all", response_model=DeleteAllResponse)
def delete_all_docs():
    vectordb = get_vectordb()
    all_ids = vectordb.get()["ids"]
    if all_ids:
        vectordb.delete(ids=all_ids)
    return DeleteAllResponse(message="All documents deleted", count=len(all_ids))
