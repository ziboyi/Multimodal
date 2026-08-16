from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    embed_model_id: Optional[str] = None
    chunk_size: int = Field(default=1000, ge=100, max_value=10000)
    chunk_overlap: int = Field(default=200, ge=0, max_value=2000)


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    embed_model_id: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=100, max_value=10000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, max_value=2000)


class KnowledgeBaseResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str]
    embed_model_id: Optional[str]
    chunk_size: int
    chunk_overlap: int
    document_count: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
