from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    kb_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # user, assistant
    content: str
    references: Optional[List[dict]] = None
    created_at: Optional[datetime] = None


class ConversationCreate(BaseModel):
    kb_id: Optional[str] = None
    title: str = "新对话"


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    kb_id: Optional[str]
    title: str
    message_count: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    references: Optional[List[dict]]
    created_at: datetime

    class Config:
        from_attributes = True
