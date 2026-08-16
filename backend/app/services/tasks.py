"""
Celery 异步任务
文档处理 Pipeline: 解析 → 分块 → 嵌入 → 索引
"""
import asyncio
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document, DocumentStatus
from app.services.doc_parser import DocParserService
from app.services.chunker import ChunkerService
from app.services.indexer import IndexerService
from app.services.minio_client import MinioService
from app.services.model_manager import model_registry

# 每个 worker 进程复用同一个 event loop，避免 asyncpg 跨 loop 问题
_worker_loop: asyncio.AbstractEventLoop = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """获取/创建当前进程的持久 event loop"""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def run_async(coro):
    """在同步上下文中运行异步代码（复用同一个 loop）"""
    loop = _get_loop()
    return loop.run_until_complete(coro)


@celery_app.task(name="process_document", bind=True, max_retries=2, time_limit=3600, soft_time_limit=3000)
def process_document_task(self, doc_id: str):
    """文档处理主流程"""
    return run_async(_process_document(doc_id))


async def _process_document(doc_id: str):
    """异步文档处理"""
    async with AsyncSessionLocal() as db:
        # 获取文档
        from sqlalchemy import select
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return {"error": "Document not found"}

        minio_svc = MinioService()

        try:
            # 1. 更新状态：解析中
            doc.status = DocumentStatus.PARSING.value
            await db.commit()

            # 2. 从 MinIO 下载文件
            file_data = await minio_svc.download_file(doc.file_path)

            # 写入临时文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=f".{doc.file_type}", delete=False) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            try:
                # 3. 解析为 Markdown
                markdown_content, extracted_images = await DocParserService.parse_to_markdown(
                    tmp_path, doc.file_type
                )
                doc.markdown_content = markdown_content

                # 上传抽取的图片到 MinIO
                for img in extracted_images:
                    img_object_name = f"{doc.kb_id}/{doc.id}/images/{img['path']}"
                    await minio_svc.upload_file(
                        object_name=img_object_name,
                        data=img["data"],
                        content_type="image/png",
                    )

                # 4. 更新状态：分块中
                doc.status = DocumentStatus.CHUNKING.value
                await db.commit()

                # 5. 获取知识库配置
                from app.models.knowledge_base import KnowledgeBase
                kb_result = await db.execute(
                    select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
                )
                kb = kb_result.scalar_one()

                # 6. 分块
                chunks = ChunkerService.chunk_markdown(
                    markdown_content,
                    chunk_size=kb.chunk_size,
                    chunk_overlap=kb.chunk_overlap,
                )

                # 检测语言
                for chunk in chunks:
                    chunk.metadata["language"] = ChunkerService.detect_language(chunk.content)

                # 7. 更新状态：索引中
                doc.status = DocumentStatus.INDEXING.value
                await db.commit()

                # 8. 获取嵌入模型
                embed_provider = model_registry.get_text_embed(
                    provider=settings.DEFAULT_EMBED_PROVIDER,
                    model=settings.DEFAULT_EMBED_MODEL,
                )

                # 9. 生成嵌入
                texts = [chunk.content for chunk in chunks]
                embeddings = await embed_provider.embed(texts)

                # 10. 索引到 ES
                indexer = IndexerService()
                indexed_count = await indexer.index_chunks(
                    kb_id=doc.kb_id,
                    doc_id=doc.id,
                    user_id=doc.user_id,
                    document_name=doc.filename,
                    document_path=doc.file_path,
                    chunks=chunks,
                    embeddings=embeddings,
                    dim=embed_provider.dim,
                )
                await indexer.close()

                # 11. 更新完成状态
                doc.status = DocumentStatus.COMPLETED.value
                doc.chunk_count = indexed_count
                await db.commit()

                return {
                    "status": "completed",
                    "doc_id": doc_id,
                    "chunks": indexed_count,
                }

            finally:
                # 清理临时文件
                os.unlink(tmp_path)

        except Exception as e:
            doc.status = DocumentStatus.FAILED.value
            doc.error_message = str(e)[:500]
            await db.commit()
            raise
