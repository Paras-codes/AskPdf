from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime

# Input Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask the RAG system")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty or just whitespace')
        return v.strip()

class DeleteRequest(BaseModel):
    ids: List[str] = Field(..., min_items=1, description="List of document IDs to delete")
    
    @validator('ids')
    def validate_ids(cls, v):
        if not v:
            raise ValueError('At least one document ID must be provided')
        # Remove any empty or None values
        valid_ids = [id_val for id_val in v if id_val and id_val.strip()]
        if not valid_ids:
            raise ValueError('All provided IDs are empty or invalid')
        return valid_ids

# Response Models
class UploadResponse(BaseModel):
    message: str
    details: Dict[str, Any]
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)

class ListDocsResponse(BaseModel):
    ids: List[str]
    count: int
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)

class DeleteDocsResponse(BaseModel):
    message: str
    deleted_ids: List[str]
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)

class DeleteAllResponse(BaseModel):
    message: str
    count: int
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)

# Error Response Models
class ErrorResponse(BaseModel):
    error: bool = True
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    success: bool = False

class ValidationErrorResponse(ErrorResponse):
    """Specific error response for validation errors"""
    error_code: str = "VALIDATION_ERROR"
    
class FileErrorResponse(ErrorResponse):
    """Specific error response for file-related errors"""
    file_info: Optional[Dict[str, Any]] = None

class DatabaseErrorResponse(ErrorResponse):
    """Specific error response for database-related errors"""
    database_info: Optional[Dict[str, Any]] = None

class ModelErrorResponse(ErrorResponse):
    """Specific error response for AI/ML model errors"""
    model_info: Optional[Dict[str, Any]] = None
