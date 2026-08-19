"""
文档处理进度 Pub/Sub
Celery worker 发布进度 -> Redis -> WebSocket 转发到前端
"""
import json
from app.core.config import settings

PROGRESS_CHANNEL = "doc_progress"


def publish_progress(user_id: str, doc_id: str, filename: str,
                     status: str, progress: int, message: str = ""):
    """Celery worker 调用：发布进度到 Redis"""
    import redis
    r = redis.Redis.from_url(settings.REDIS_URL_COMPUTED)
    payload = json.dumps({
        "type": "document_progress",
        "user_id": user_id,
        "doc_id": doc_id,
        "filename": filename,
        "status": status,
        "progress": progress,
        "message": message,
    })
    r.publish(PROGRESS_CHANNEL, payload)
    r.close()
