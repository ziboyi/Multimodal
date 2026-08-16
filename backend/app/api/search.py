from fastapi import APIRouter, Depends
from app.core.dependencies import get_current_user_id
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.services.retriever import RetrieverService
import time

router = APIRouter(prefix="/search", tags=["检索"])


@router.post("", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    user_id: str = Depends(get_current_user_id),
):
    """多模态混合检索"""
    start_time = time.time()
    
    retriever = RetrieverService()
    try:
        results = await retriever.search(
            user_id=user_id,
            query=req.query,
            kb_ids=req.kb_ids,
            top_k=req.top_k,
            search_mode=req.search_mode,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=req.query,
            results=[
                SearchResultItem(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    kb_id=r.kb_id,
                    document_name=r.document_name,
                    content=r.content,
                    chunk_type=r.chunk_type,
                    page_number=r.page_number,
                    score=r.score,
                    image_url=r.image_url or "",
                    highlight=r.highlight or "",
                )
                for r in results
            ],
            total=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )
    finally:
        await retriever.close()
