from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from config import PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from error_handler import (
    FileError, DatabaseError, ModelError, ErrorCode, 
    AskPdfBaseException, validate_file_type
)

load_dotenv()

async def ingest_pdfs(file_paths: list[str], persist_dir=PERSIST_DIR):
    """
    Ingest PDF files into the vector database with comprehensive error handling.
    
    Args:
        file_paths: List of paths to PDF files
        persist_dir: Directory to persist the vector database
        
    Returns:
        Dict containing ingestion results
        
    Raises:
        FileError: If files cannot be read or processed
        ModelError: If embeddings model fails
        DatabaseError: If vector database operations fail
    """
    try:
        # Validate inputs
        if not file_paths:
            raise FileError(
                message="No file paths provided for ingestion",
                error_code=ErrorCode.VALIDATION_ERROR
            )
        
        # Validate all files exist and are accessible
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileError(
                    message=f"File not found: {file_path}",
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    details={"file_path": file_path}
                )
            
            # Validate file type
            try:
                validate_file_type(os.path.basename(file_path))
            except Exception as e:
                raise FileError(
                    message=f"Invalid file type for {file_path}: {str(e)}",
                    error_code=ErrorCode.INVALID_FILE_FORMAT,
                    details={"file_path": file_path},
                    original_error=e
                )
        
        # Initialize embeddings model
        try:
            embeddings = HuggingFaceEndpointEmbeddings(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                task="feature-extraction"
            )
        except Exception as e:
            raise ModelError(
                message=f"Failed to initialize embeddings model: {str(e)}",
                error_code=ErrorCode.EMBEDDING_ERROR,
                original_error=e
            )
        
        # Initialize vector database
        try:
            # Ensure persist directory exists
            os.makedirs(persist_dir, exist_ok=True)
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        except Exception as e:
            raise DatabaseError(
                message=f"Failed to initialize vector database: {str(e)}",
                error_code=ErrorCode.VECTORDB_ERROR,
                details={"persist_dir": persist_dir},
                original_error=e
            )

        total_chunks = 0
        processed_files = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                
                # Load PDF
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                except Exception as e:
                    failed_files.append({
                        "file": filename,
                        "error": f"PDF loading failed: {str(e)}"
                    })
                    continue
                
                # Check if PDF has content
                if not docs:
                    failed_files.append({
                        "file": filename,
                        "error": "PDF appears to be empty or corrupted"
                    })
                    continue
                
                # Split documents into chunks
                try:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE, 
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    chunks = splitter.split_documents(docs)
                except Exception as e:
                    failed_files.append({
                        "file": filename,
                        "error": f"Document splitting failed: {str(e)}"
                    })
                    continue
                
                # Check if chunks were created
                if not chunks:
                    failed_files.append({
                        "file": filename,
                        "error": "No text chunks could be created from PDF"
                    })
                    continue

                # Add metadata to chunks
                for i, chunk in enumerate(chunks):
                    chunk.metadata["doc_id"] = f"{filename}_chunk_{i}"
                    chunk.metadata["source_file"] = filename
                    chunk.metadata["chunk_index"] = i

                # Add chunks to vector database
                try:
                    vectordb.add_documents(chunks)
                    total_chunks += len(chunks)
                    processed_files.append({
                        "file": filename,
                        "chunks": len(chunks)
                    })
                except Exception as e:
                    failed_files.append({
                        "file": filename,
                        "error": f"Failed to add to vector database: {str(e)}"
                    })
                    continue

            except Exception as e:
                failed_files.append({
                    "file": os.path.basename(file_path),
                    "error": f"Unexpected error during processing: {str(e)}"
                })
                continue

        # Persist the database
        try:
            vectordb.persist()
        except Exception as e:
            raise DatabaseError(
                message=f"Failed to persist vector database: {str(e)}",
                error_code=ErrorCode.DB_OPERATION_ERROR,
                original_error=e
            )
        
        # Check if any files were processed successfully
        if total_chunks == 0:
            raise FileError(
                message="No documents could be processed successfully",
                error_code=ErrorCode.PDF_PROCESSING_ERROR,
                details={"failed_files": failed_files}
            )
        
        result = {
            "total_chunks": total_chunks,
            "processed_files": len(processed_files),
            "failed_files": len(failed_files)
        }
        
        # Add details if there were failures
        if failed_files:
            result["failures"] = failed_files
        
        if processed_files:
            result["successes"] = processed_files
        
        return result

    except AskPdfBaseException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Handle any unexpected errors
        raise FileError(
            message=f"Unexpected error during PDF ingestion: {str(e)}",
            error_code=ErrorCode.PDF_PROCESSING_ERROR,
            original_error=e
        )
