from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    kb_ids: Optional[List[str]] = None  # 不指定则检索所有知识库
    top_k: int = Field(default=10, ge=1, max_value=100)
    search_mode: str = Field(default="hybrid")  # hybrid, semantic, keyword


class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    kb_id: str
    document_name: str
    content: str
    chunk_type: str
    page_number: Optional[int]
    score: float
    image_url: Optional[str] = None
    highlight: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total: int
    elapsed_ms: float
