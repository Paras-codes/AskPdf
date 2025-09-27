import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from ingest import ingest_pdfs
from rag_chain import get_rag_chain, get_vectordb
from models import QueryRequest, AnswerResponse, ListDocsResponse, DeleteDocsResponse, DeleteAllResponse, UploadResponse, DeleteRequest
from config import UPLOAD_DIR
from error_handler import (
    AskPdfBaseException, FileError, DatabaseError, ModelError, QueryError,
    ValidationError, ErrorCode, validate_file_type, validate_file_size,
    setup_logging
)

# Initialize logging
setup_logging()

app = FastAPI(title="RAG System with Gemini")

# Custom exception handler
@app.exception_handler(AskPdfBaseException)
async def askpdf_exception_handler(request, exc: AskPdfBaseException):
    return JSONResponse(
        status_code=400 if exc.error_code != ErrorCode.INTERNAL_SERVER_ERROR else 500,
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    # Convert unexpected errors to our custom exception format
    error = AskPdfBaseException(
        message=f"Internal server error: {str(exc)}",
        error_code=ErrorCode.INTERNAL_SERVER_ERROR,
        original_error=exc
    )
    return JSONResponse(
        status_code=500,
        content=error.to_dict()
    )

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Upload PDFs
# -----------------------------
@app.post("/upload_pdfs", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        # Validate input
        if not files:
            raise ValidationError(
                message="No files provided",
                error_code=ErrorCode.VALIDATION_ERROR
            )
        
        file_paths = []
        for file in files:
            # Validate file type
            validate_file_type(file.filename)
            
            # Validate file size (convert bytes to MB for validation)
            if hasattr(file, 'size') and file.size:
                validate_file_size(file.size)
            
            try:
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                
                # Ensure upload directory exists
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(file_path)
                
            except IOError as e:
                raise FileError(
                    message=f"Failed to save file {file.filename}: {str(e)}",
                    error_code=ErrorCode.FILE_UPLOAD_ERROR,
                    details={"filename": file.filename},
                    original_error=e
                )

        # Process the PDFs
        result = await ingest_pdfs(file_paths)
        return UploadResponse(message="PDFs processed and added to DB", details=result)
        
    except AskPdfBaseException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise FileError(
            message=f"Unexpected error during file upload: {str(e)}",
            error_code=ErrorCode.FILE_UPLOAD_ERROR,
            original_error=e
        )

# -----------------------------
# Ask Question
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QueryRequest):
    try:
        # Validate input
        if not req.question or req.question.strip() == "":
            raise ValidationError(
                message="Question cannot be empty",
                error_code=ErrorCode.VALIDATION_ERROR
            )
        
        # Get QA chain and process query
        qa_chain = get_rag_chain()
        if qa_chain is None:
            raise ModelError(
                message="Failed to initialize RAG chain",
                error_code=ErrorCode.CHAIN_ERROR
            )
            
        result = qa_chain({"query": req.question})
        
        if not result or "result" not in result:
            raise QueryError(
                message="Failed to get response from RAG chain",
                error_code=ErrorCode.QUERY_ERROR,
                details={"question": req.question}
            )
            
        return AnswerResponse(
            answer=result["result"], 
            sources=[doc.metadata for doc in result.get("source_documents", [])]
        )
        
    except AskPdfBaseException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise QueryError(
            message=f"Unexpected error during question processing: {str(e)}",
            error_code=ErrorCode.QUERY_ERROR,
            details={"question": req.question},
            original_error=e
        )

# -----------------------------
# List Documents
# -----------------------------
@app.get("/list_docs", response_model=ListDocsResponse)
async def list_docs():
    try:
        vectordb = get_vectordb()
        if vectordb is None:
            raise DatabaseError(
                message="Failed to connect to vector database",
                error_code=ErrorCode.DB_CONNECTION_ERROR
            )
            
        all_docs = vectordb.get()
        
        if all_docs is None:
            raise DatabaseError(
                message="Failed to retrieve documents from database",
                error_code=ErrorCode.DB_OPERATION_ERROR
            )
            
        doc_ids = all_docs.get("ids", [])
        return ListDocsResponse(ids=doc_ids, count=len(doc_ids))
        
    except AskPdfBaseException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise DatabaseError(
            message=f"Unexpected error while listing documents: {str(e)}",
            error_code=ErrorCode.DB_OPERATION_ERROR,
            original_error=e
        )

# -----------------------------
# Delete Specific Documents
# -----------------------------
@app.post("/delete_docs", response_model=DeleteDocsResponse)
async def delete_docs(req: DeleteRequest):
    try:
        # Validate input
        if not req.ids:
            raise ValidationError(
                message="No document IDs provided for deletion",
                error_code=ErrorCode.VALIDATION_ERROR
            )
            
        vectordb = get_vectordb()
        if vectordb is None:
            raise DatabaseError(
                message="Failed to connect to vector database",
                error_code=ErrorCode.DB_CONNECTION_ERROR
            )
            
        # Check if documents exist before deletion
        existing_docs =vectordb.get()
        existing_ids = existing_docs.get("ids", [])
        
        invalid_ids = [doc_id for doc_id in req.ids if doc_id not in existing_ids]
        if invalid_ids:
            raise ValidationError(
                message=f"Some document IDs not found: {invalid_ids}",
                error_code=ErrorCode.DOCUMENT_NOT_FOUND,
                details={"invalid_ids": invalid_ids}
            )
            
        vectordb.delete(ids=req.ids)
        return DeleteDocsResponse(message="Documents deleted", deleted_ids=req.ids)
        
    except AskPdfBaseException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise DatabaseError(
            message=f"Unexpected error during document deletion: {str(e)}",
            error_code=ErrorCode.DB_OPERATION_ERROR,
            details={"requested_ids": req.ids},
            original_error=e
        )

# -----------------------------
# Delete All Documents
# -----------------------------
@app.delete("/delete_all", response_model=DeleteAllResponse)
async def delete_all_docs():
    try:
        vectordb = get_vectordb()
        if vectordb is None:
            raise DatabaseError(
                message="Failed to connect to vector database",
                error_code=ErrorCode.DB_CONNECTION_ERROR
            )
            
        all_docs = vectordb.get()
        all_ids = all_docs.get("ids", [])
        
        deleted_count = len(all_ids)
        
        if all_ids:
            vectordb.delete(ids=all_ids)
            
        return DeleteAllResponse(
            message=f"All documents deleted ({deleted_count} documents)", 
            count=deleted_count
        )
        
    except AskPdfBaseException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise DatabaseError(
            message=f"Unexpected error during bulk deletion: {str(e)}",
            error_code=ErrorCode.DB_OPERATION_ERROR,
            original_error=e
        )
