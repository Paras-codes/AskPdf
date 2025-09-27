# AskPdf Error Handling System

## Overview

The AskPdf project now includes a comprehensive error handling system that provides robust error management across all modules. The system uses custom exception classes, structured error responses, and detailed logging to ensure reliable operation and easier debugging.

## Key Features

### 1. Custom Exception Classes
- **AskPdfBaseException**: Base exception class for all application errors
- **FileError**: For file-related operations (upload, processing, validation)
- **DatabaseError**: For vector database operations
- **ModelError**: For AI/ML model operations
- **QueryError**: For query/RAG chain operations
- **ValidationError**: For input validation errors
- **ConfigurationError**: For configuration-related errors

### 2. Error Codes
Standardized error codes for different types of errors:
- `FILE_NOT_FOUND`, `FILE_UPLOAD_ERROR`, `PDF_PROCESSING_ERROR`
- `DB_CONNECTION_ERROR`, `DB_OPERATION_ERROR`, `VECTORDB_ERROR`
- `MODEL_INITIALIZATION_ERROR`, `EMBEDDING_ERROR`, `LLM_ERROR`
- `QUERY_ERROR`, `CHAIN_ERROR`, `RETRIEVAL_ERROR`
- `VALIDATION_ERROR`, `CONFIGURATION_ERROR`

### 3. Enhanced API Responses
- Structured error responses with error codes and timestamps
- Detailed error information for debugging
- Consistent response format across all endpoints

### 4. Comprehensive Logging
- Automatic logging of all errors to `askpdf_errors.log`
- Error details include timestamps, error codes, and stack traces
- Console and file logging support

## Error Handling by Module

### main.py (FastAPI Application)
- **File Upload Validation**: File type, size, and format validation
- **Request Validation**: Input sanitization and validation
- **Database Operations**: Connection and operation error handling
- **Query Processing**: RAG chain and response error handling
- **Custom Exception Handlers**: Automatic conversion to JSON error responses

### ingest.py (PDF Processing)
- **File Access Validation**: Existence and permission checks
- **PDF Processing**: Handling corrupted or empty PDFs
- **Text Extraction**: Error handling for malformed documents
- **Vector Database Operations**: Embedding and storage error handling
- **Batch Processing**: Individual file error tracking with detailed reporting

### rag_chain.py (RAG Chain Operations)
- **Model Initialization**: LLM and embedding model error handling
- **Database Connection**: Vector database connectivity and validation
- **Chain Creation**: RAG chain setup and configuration error handling
- **Environment Validation**: API key and configuration checks

### models.py (Data Models)
- **Input Validation**: Pydantic validators for request data
- **Error Response Models**: Structured error response schemas
- **Field Validation**: Type checking and constraint validation

### error_handler.py (Core Error System)
- **Exception Hierarchy**: Organized exception classes
- **Error Decorators**: Automatic error handling for functions
- **Utility Functions**: File validation, environment checks, logging setup
- **Error Response Generation**: Structured error dictionaries

## Usage Examples

### 1. File Upload with Error Handling
```python
# Automatic validation and error handling
@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    # Validates file type, size, processes PDFs
    # Returns structured error responses on failure
```

### 2. Query Processing with Error Handling
```python
# Validates input, handles model errors, returns structured responses
@app.post("/ask")
async def ask_question(req: QueryRequest):
    # Validates question, initializes RAG chain, processes query
    # Returns detailed error information on failure
```

### 3. Database Operations with Error Handling
```python
# Handles connection errors, operation failures, validation
@app.get("/list_docs")
async def list_docs():
    # Connects to vector DB, retrieves documents
    # Returns error details on database issues
```

## Configuration

### Environment Variables
The system checks for required environment variables:
- `GOOGLE_API_KEY`: Required for Gemini LLM
- `UPLOAD_DIR`: Directory for uploaded files (optional)
- `PERSIST_DIR`: Vector database directory (optional)

### Logging Configuration
```python
# Automatic setup in error_handler.py
setup_logging()  # Creates askpdf_errors.log and console output
```

### File Validation Settings
- Allowed file types: `.pdf` (configurable)
- Maximum file size: 50MB (configurable)
- File name validation and sanitization

## Error Response Format

All errors return a consistent JSON structure:
```json
{
  "error": true,
  "error_code": "FILE_UPLOAD_ERROR",
  "message": "Failed to save file example.pdf: Permission denied",
  "details": {
    "filename": "example.pdf"
  },
  "timestamp": "2025-09-27T10:30:00",
  "success": false
}
```

## Benefits

1. **Reliability**: Graceful handling of all error scenarios
2. **Debugging**: Detailed error information and logging
3. **User Experience**: Clear, actionable error messages
4. **Maintainability**: Centralized error handling logic
5. **Monitoring**: Comprehensive error logging and tracking
6. **API Consistency**: Structured error responses across all endpoints

## Testing

The system includes comprehensive error handling tests that verify:
- Exception creation and structure
- Error code assignment
- Message formatting
- Logging functionality
- Response generation

Run tests with:
```bash
python test_error_handling.py
```

## Best Practices

1. **Always use try-catch blocks** for external operations
2. **Validate input data** before processing
3. **Log errors with context** for debugging
4. **Return structured error responses** to clients
5. **Check environment variables** on startup
6. **Handle expected failures gracefully** (empty databases, missing files)
7. **Provide actionable error messages** to users

This error handling system ensures your AskPdf application is robust, maintainable, and provides excellent user experience even when things go wrong.