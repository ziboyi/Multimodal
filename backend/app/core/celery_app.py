from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "multimodal",
    broker=settings.REDIS_URL_COMPUTED,
    backend=settings.REDIS_URL_COMPUTED,
    include=[
        "app.services.tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 分钟超时
    task_soft_time_limit=500,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=100,
)
