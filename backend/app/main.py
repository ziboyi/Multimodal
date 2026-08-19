"""
FastAPI 应用入口
"""
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import init_db
from app.core.ws_manager import ws_manager
from app.api import auth, kb, search, chat, model_config, websocket

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(kb.router, prefix=settings.API_V1_PREFIX)
app.include_router(search.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)
app.include_router(model_config.router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket.router)


async def redis_progress_subscriber():
    """后台任务：异步订阅 Redis 进度频道，转发到 WebSocket"""
    import json
    import structlog
    from redis import asyncio as aioredis

    logger = structlog.get_logger()
    logger.info("Starting Redis progress subscriber...")

    r = aioredis.from_url(settings.REDIS_URL_COMPUTED)
    pubsub = r.pubsub()
    await pubsub.subscribe("doc_progress")

    logger.info("Subscribed to doc_progress channel")

    try:
        while True:
            try:
                message = await asyncio.wait_for(pubsub.get_message(timeout=1.0), timeout=2.0)
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        user_id = data.get("user_id")
                        if user_id:
                            await ws_manager.send_to_user(user_id, data)
                    except Exception as e:
                        logger.error("Error forwarding progress", error=str(e))
            except asyncio.TimeoutError:
                continue  # 正常超时，继续循环
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe("doc_progress")
        await r.aclose()


@app.on_event("startup")
async def startup():
    await init_db()
    # 启动 Redis 进度订阅后台任务
    asyncio.create_task(redis_progress_subscriber())


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}
