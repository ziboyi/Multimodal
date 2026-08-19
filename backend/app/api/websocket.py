"""
WebSocket 实时推送端点
文档处理进度通知
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.ws_manager import ws_manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
):
    """
    WebSocket 实时连接
    
    前端连接后接收文档处理进度推送：
    {
        "type": "document_progress",
        "doc_id": "...",
        "filename": "...",
        "status": "parsing|chunking|indexing|completed|failed",
        "progress": 0-100,
        "message": "..."
    }
    """
    await ws_manager.connect(websocket, user_id)
    try:
        # 发送连接成功消息
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connected",
        })
        # 保持连接并处理心跳
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, user_id)
    except Exception:
        await ws_manager.disconnect(websocket, user_id)
