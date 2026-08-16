from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class ModelConfigCreate(BaseModel):
    provider_type: str = Field(..., pattern="^(llm|text_embed|vision_embed|vision_llm)$")
    provider_name: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_config: Optional[Dict[str, Any]] = None
    is_default: bool = False
    is_active: bool = True


class ModelConfigUpdate(BaseModel):
    provider_name: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_config: Optional[Dict[str, Any]] = None
    is_default: Optional[bool] = None
    is_active: Optional[bool] = None


class ModelConfigResponse(BaseModel):
    id: str
    user_id: Optional[str]
    provider_type: str
    provider_name: str
    model_name: str
    api_base: Optional[str]
    extra_config: Optional[Dict[str, Any]]
    is_default: bool
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
