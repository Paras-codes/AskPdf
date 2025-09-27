from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
from config import PERSIST_DIR
from error_handler import (
    DatabaseError, ModelError, ErrorCode, AskPdfBaseException,
    check_required_env_vars
)

load_dotenv()

# Check required environment variables
try:
    check_required_env_vars(['GOOGLE_API_KEY'])
except Exception as e:
    # Log the error but don't crash the module import
    import logging
    logging.getLogger(__name__).warning(f"Environment variable check failed: {e}")

qa_chain = None

# Prompt Template
PROMPT_TEMPLATE = """You are a helpful assistant. 
Use the following context to answer the question.
If the answer is not in the context, say 'I don’t know'.

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def get_vectordb():
    """
    Initialize and return the vector database with error handling.
    
    Returns:
        Chroma: The vector database instance
        
    Raises:
        ModelError: If embeddings model initialization fails
        DatabaseError: If vector database initialization fails
    """
    try:
        # Initialize embeddings model
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
    
    try:
        # Check if persist directory exists
        if not os.path.exists(PERSIST_DIR):
            raise DatabaseError(
                message=f"Vector database directory does not exist: {PERSIST_DIR}",
                error_code=ErrorCode.DB_CONNECTION_ERROR,
                details={"persist_dir": PERSIST_DIR}
            )
        
        # Initialize vector database
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        return vectordb
        
    except DatabaseError:
        # Re-raise our custom database errors
        raise
    except Exception as e:
        raise DatabaseError(
            message=f"Failed to initialize vector database: {str(e)}",
            error_code=ErrorCode.VECTORDB_ERROR,
            details={"persist_dir": PERSIST_DIR},
            original_error=e
        )

def get_rag_chain():
    """
    Initialize and return the RAG chain with comprehensive error handling.
    
    Returns:
        RetrievalQA: The initialized RAG chain
        
    Raises:
        ModelError: If LLM initialization fails
        DatabaseError: If vector database operations fail
        AskPdfBaseException: For other initialization errors
    """
    global qa_chain
    
    if qa_chain is None:
        try:
            # Get vector database (this may raise DatabaseError or ModelError)
            vectordb = get_vectordb()
            
            # Verify that vector database has content
            try:
                # Check if there are any documents in the database
                test_docs = vectordb.get(limit=1)
                if not test_docs or not test_docs.get("ids"):
                    raise DatabaseError(
                        message="Vector database is empty. Please upload and process some documents first.",
                        error_code=ErrorCode.DB_OPERATION_ERROR,
                        details={"persist_dir": PERSIST_DIR}
                    )
            except Exception as db_check_error:
                # If we can't check the database, log it but continue
                import logging
                logging.getLogger(__name__).warning(f"Could not verify database content: {db_check_error}")
            
            # Initialize LLM
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
            except Exception as e:
                raise ModelError(
                    message=f"Failed to initialize Gemini LLM: {str(e)}. Please check your GOOGLE_API_KEY.",
                    error_code=ErrorCode.LLM_ERROR,
                    original_error=e
                )
            
            # Create retriever
            try:
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            except Exception as e:
                raise DatabaseError(
                    message=f"Failed to create retriever from vector database: {str(e)}",
                    error_code=ErrorCode.RETRIEVAL_ERROR,
                    original_error=e
                )
            
            # Initialize RAG chain
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
            except Exception as e:
                raise ModelError(
                    message=f"Failed to create RAG chain: {str(e)}",
                    error_code=ErrorCode.CHAIN_ERROR,
                    original_error=e
                )
                
        except AskPdfBaseException:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Handle any unexpected errors
            raise ModelError(
                message=f"Unexpected error during RAG chain initialization: {str(e)}",
                error_code=ErrorCode.MODEL_INITIALIZATION_ERROR,
                original_error=e
            )
    
    return qa_chain
