"""
WebSocket 连接管理器
支持按 user_id 广播消息
"""
import asyncio
import json
from typing import Dict, List, Optional
from fastapi import WebSocket
import structlog

logger = structlog.get_logger()


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        # user_id -> List[WebSocket]
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str):
        """注册新连接"""
        await websocket.accept()
        async with self._lock:
            if user_id not in self._connections:
                self._connections[user_id] = []
            self._connections[user_id].append(websocket)
        logger.info("WebSocket connected", user_id=user_id, total=len(self._connections[user_id]))

    async def disconnect(self, websocket: WebSocket, user_id: str):
        """断开连接"""
        async with self._lock:
            if user_id in self._connections:
                self._connections[user_id].remove(websocket)
                if not self._connections[user_id]:
                    del self._connections[user_id]
        logger.info("WebSocket disconnected", user_id=user_id)

    async def send_to_user(self, user_id: str, message: dict):
        """向指定用户的所有连接发送消息"""
        connections = self._connections.get(user_id, [])
        dead = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        # 清理断开的连接
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections.get(user_id, []):
                        self._connections[user_id].remove(ws)

    async def broadcast(self, message: dict):
        """向所有连接广播"""
        all_ws = []
        for conns in self._connections.values():
            all_ws.extend(conns)
        for ws in all_ws:
            try:
                await ws.send_json(message)
            except Exception:
                pass


# 全局单例
ws_manager = ConnectionManager()
