from minio import Minio
from minio.error import S3Error
from app.core.config import settings
from typing import Optional
import io

# MinIO 客户端单例
_minio_client: Optional[Minio] = None


def get_minio_client() -> Minio:
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_SECURE,
        )
        # 确保 bucket 存在
        if not _minio_client.bucket_exists(settings.MINIO_BUCKET):
            _minio_client.make_bucket(settings.MINIO_BUCKET)
    return _minio_client


class MinioService:
    """MinIO 文件操作服务"""

    def __init__(self):
        self.client = get_minio_client()
        self.bucket = settings.MINIO_BUCKET

    async def upload_file(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """上传文件，返回文件路径"""
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return f"minio://{self.bucket}/{object_name}"

    async def download_file(self, object_name: str) -> bytes:
        """下载文件"""
        response = self.client.get_object(
            bucket_name=self.bucket,
            object_name=object_name,
        )
        return response.read()

    async def delete_file(self, object_name: str) -> None:
        """删除文件"""
        self.client.remove_object(
            bucket_name=self.bucket,
            object_name=object_name,
        )

    async def get_file_url(self, object_name: str, expires: int = 3600) -> str:
        """获取预签名 URL"""
        return self.client.presigned_get_object(
            bucket_name=self.bucket,
            object_name=object_name,
            expires=expires,
        )
