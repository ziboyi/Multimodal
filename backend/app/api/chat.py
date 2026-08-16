from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.core.dependencies import get_current_user_id
from app.schemas.chat import ChatRequest, ConversationCreate, ConversationResponse

router = APIRouter(prefix="/chat", tags=["问答"])


@router.post("/ask")
async def chat_ask(
    req: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    """流式问答"""
    # TODO: 接入 rag_pipeline 服务
    pass


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    req: ConversationCreate,
    user_id: str = Depends(get_current_user_id),
):
    """创建对话"""
    # TODO: 实现
    pass


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: str = Depends(get_current_user_id),
):
    """获取对话列表"""
    # TODO: 实现
    pass
