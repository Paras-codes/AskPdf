from pydantic import BaseModel
from typing import List, Dict

# Input Models
class QueryRequest(BaseModel):
    question: str

class DeleteRequest(BaseModel):
    ids: List[str]

# Response Models
class UploadResponse(BaseModel):
    message: str
    details: Dict[str, int]

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict]

class ListDocsResponse(BaseModel):
    ids: List[str]
    count: int

class DeleteDocsResponse(BaseModel):
    message: str
    deleted_ids: List[str]

class DeleteAllResponse(BaseModel):
    message: str
    count: int
