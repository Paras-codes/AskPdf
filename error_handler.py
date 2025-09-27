"""
Error handling module for AskPdf RAG System.

This module provides custom exception classes and error handling utilities
for the entire application stack.
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class ErrorCode(Enum):
    """Error codes for different types of errors."""
    # File/PDF related errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_UPLOAD_ERROR = "FILE_UPLOAD_ERROR"
    PDF_PROCESSING_ERROR = "PDF_PROCESSING_ERROR"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    FILE_SIZE_EXCEEDED = "FILE_SIZE_EXCEEDED"
    
    # Database/Vector Store errors
    DB_CONNECTION_ERROR = "DB_CONNECTION_ERROR"
    DB_OPERATION_ERROR = "DB_OPERATION_ERROR"
    VECTORDB_ERROR = "VECTORDB_ERROR"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    
    # AI/ML Model errors
    MODEL_INITIALIZATION_ERROR = "MODEL_INITIALIZATION_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    LLM_ERROR = "LLM_ERROR"
    API_KEY_ERROR = "API_KEY_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    
    # Query/RAG Chain errors
    QUERY_ERROR = "QUERY_ERROR"
    CHAIN_ERROR = "CHAIN_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    
    # General errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class AskPdfBaseException(Exception):
    """Base exception class for all AskPdf application errors."""
    
    def __init__(self, 
                 message: str,
                 error_code: ErrorCode,
                 details: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error
        self.timestamp = datetime.now().isoformat()
        
        # Log the error
        self._log_error()
        
        super().__init__(self.message)
    
    def _log_error(self):
        """Log the error details."""
        logger = logging.getLogger(__name__)
        error_info = {
            "timestamp": self.timestamp,
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }
        
        if self.original_error:
            error_info["original_error"] = str(self.original_error)
            error_info["traceback"] = traceback.format_exc()
        
        logger.error(f"AskPdf Error: {error_info}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": True,
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


class FileError(AskPdfBaseException):
    """Exception for file-related operations."""
    pass


class DatabaseError(AskPdfBaseException):
    """Exception for database-related operations."""
    pass


class ModelError(AskPdfBaseException):
    """Exception for AI/ML model-related operations."""
    pass


class QueryError(AskPdfBaseException):
    """Exception for query/RAG chain operations."""
    pass


class ValidationError(AskPdfBaseException):
    """Exception for validation errors."""
    pass


class ConfigurationError(AskPdfBaseException):
    """Exception for configuration errors."""
    pass


def handle_error(func):
    """
    Decorator to handle exceptions in functions and convert them to AskPdf exceptions.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AskPdfBaseException:
            # Re-raise our custom exceptions as-is
            raise
        except FileNotFoundError as e:
            raise FileError(
                message=f"File not found: {str(e)}",
                error_code=ErrorCode.FILE_NOT_FOUND,
                original_error=e
            )
        except PermissionError as e:
            raise FileError(
                message=f"Permission denied: {str(e)}",
                error_code=ErrorCode.FILE_UPLOAD_ERROR,
                original_error=e
            )
        except ValueError as e:
            raise ValidationError(
                message=f"Validation error: {str(e)}",
                error_code=ErrorCode.VALIDATION_ERROR,
                original_error=e
            )
        except ConnectionError as e:
            raise DatabaseError(
                message=f"Database connection error: {str(e)}",
                error_code=ErrorCode.DB_CONNECTION_ERROR,
                original_error=e
            )
        except Exception as e:
            # Catch all other exceptions
            raise AskPdfBaseException(
                message=f"Unexpected error: {str(e)}",
                error_code=ErrorCode.UNKNOWN_ERROR,
                original_error=e
            )
    
    return wrapper


async def handle_async_error(func):
    """
    Async version of the error handling decorator.
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AskPdfBaseException:
            # Re-raise our custom exceptions as-is
            raise
        except FileNotFoundError as e:
            raise FileError(
                message=f"File not found: {str(e)}",
                error_code=ErrorCode.FILE_NOT_FOUND,
                original_error=e
            )
        except PermissionError as e:
            raise FileError(
                message=f"Permission denied: {str(e)}",
                error_code=ErrorCode.FILE_UPLOAD_ERROR,
                original_error=e
            )
        except ValueError as e:
            raise ValidationError(
                message=f"Validation error: {str(e)}",
                error_code=ErrorCode.VALIDATION_ERROR,
                original_error=e
            )
        except ConnectionError as e:
            raise DatabaseError(
                message=f"Database connection error: {str(e)}",
                error_code=ErrorCode.DB_CONNECTION_ERROR,
                original_error=e
            )
        except Exception as e:
            # Catch all other exceptions
            raise AskPdfBaseException(
                message=f"Unexpected error: {str(e)}",
                error_code=ErrorCode.UNKNOWN_ERROR,
                original_error=e
            )
    
    return wrapper


def setup_logging(log_level: str = "INFO"):
    """
    Set up logging configuration for the application.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('askpdf_errors.log'),
            logging.StreamHandler()
        ]
    )


def validate_file_type(filename: str, allowed_extensions: list = ['.pdf']) -> bool:
    """
    Validate file type based on extension.
    
    Args:
        filename: Name of the file to validate
        allowed_extensions: List of allowed file extensions
    
    Returns:
        bool: True if file type is allowed
    
    Raises:
        ValidationError: If file type is not allowed
    """
    import os
    
    if not filename:
        raise ValidationError(
            message="Filename cannot be empty",
            error_code=ErrorCode.VALIDATION_ERROR
        )
    
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise ValidationError(
            message=f"File type {file_extension} not allowed. Allowed types: {allowed_extensions}",
            error_code=ErrorCode.INVALID_FILE_FORMAT,
            details={"filename": filename, "extension": file_extension}
        )
    
    return True


def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
    """
    Validate file size.
    
    Args:
        file_size: Size of file in bytes
        max_size_mb: Maximum allowed size in MB
    
    Returns:
        bool: True if file size is within limits
    
    Raises:
        ValidationError: If file size exceeds limit
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        raise ValidationError(
            message=f"File size {file_size / (1024*1024):.2f}MB exceeds maximum allowed size of {max_size_mb}MB",
            error_code=ErrorCode.FILE_SIZE_EXCEEDED,
            details={"file_size_mb": file_size / (1024*1024), "max_size_mb": max_size_mb}
        )
    
    return True


def check_required_env_vars(required_vars: list) -> None:
    """
    Check if required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
    
    Raises:
        ConfigurationError: If any required environment variable is missing
    """
    import os
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ConfigurationError(
            message=f"Required environment variables are missing: {missing_vars}",
            error_code=ErrorCode.CONFIGURATION_ERROR,
            details={"missing_variables": missing_vars}
        )


# Initialize logging
setup_logging()