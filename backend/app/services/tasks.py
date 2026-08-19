"""
Celery 异步任务
文档处理 Pipeline: 解析 → 分块 → 嵌入 → 索引
"""
import asyncio
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.progress_pubsub import publish_progress
from app.models.document import Document, DocumentStatus
from app.services.doc_parser import DocParserService
from app.services.chunker import ChunkerService, TextChunk
from app.services.indexer import IndexerService
from app.services.minio_client import MinioService
from app.services.model_manager import model_registry
from datetime import timedelta

_worker_loop: asyncio.AbstractEventLoop = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def run_async(coro):
    loop = _get_loop()
    return loop.run_until_complete(coro)


def _notify(user_id: str, doc_id: str, filename: str,
            status: str, progress: int, message: str = ""):
    try:
        publish_progress(user_id, doc_id, filename, status, progress, message)
    except Exception:
        pass


@celery_app.task(name="process_document", bind=True, max_retries=2, time_limit=3600, soft_time_limit=3000)
def process_document_task(self, doc_id: str):
    return run_async(_process_document(doc_id))


async def _process_document(doc_id: str):
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return {"error": "Document not found"}

        user_id = doc.user_id
        filename = doc.filename
        minio_svc = MinioService()

        try:
            # 1. 解析中
            doc.status = DocumentStatus.PARSING.value
            await db.commit()
            _notify(user_id, doc_id, filename, "parsing", 10, "正在解析文档...")

            # 2. 从 MinIO 下载文件
            file_data = await minio_svc.download_file(doc.file_path)

            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=f".{doc.file_type}", delete=False) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            try:
                # 3. 快速模式：pymupdf 提取文本+图片（不跑 OCR）
                markdown_content, extracted_images, page_info = await DocParserService.parse_pdf_fast(
                    tmp_path
                )
                doc.markdown_content = markdown_content

                # 4. 分块 - 两次解析策略
                doc.status = DocumentStatus.CHUNKING.value
                await db.commit()
                _notify(user_id, doc_id, filename, "chunking", 40, "正在分块...")

                from app.models.knowledge_base import KnowledgeBase
                kb_result = await db.execute(
                    select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
                )
                kb = kb_result.scalar_one()

                # 第一次：纯文本分块（按段落，不切散句子）
                text_chunks = ChunkerService.chunk_by_paragraph(
                    markdown_content,
                    chunk_size=kb.chunk_size,
                    chunk_overlap=kb.chunk_overlap,
                )
                for chunk in text_chunks:
                    chunk.chunk_type = "text"

                # 第二次：提取图片，每个图片+caption 独立成块
                image_chunks = []
                import re
                img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
                for match in img_pattern.finditer(markdown_content):
                    img_path = match.group(2)
                    # 查找 caption（图片后的第一行非空文本）
                    after = markdown_content[match.end():]
                    caption = ""
                    for line in after.split('\n'):
                        line = line.strip()
                        if line:
                            caption = line
                            break
                    # 查找图片信息
                    img_info = None
                    for img in extracted_images:
                        if img['path'] == img_path:
                            img_info = img
                            break
                    image_chunks.append(TextChunk(
                        content=caption or img_info.get('caption', '') if img_info else '',
                        chunk_index=0,  # 稍后重新编号
                        chunk_type="image",
                        metadata={"image_info": img_info, "image_path": img_path}
                    ))

                # 合并：文本块 + 图片块，重新编号
                all_chunks = text_chunks + image_chunks
                for i, chunk in enumerate(all_chunks):
                    chunk.chunk_index = i
                    chunk.metadata["language"] = ChunkerService.detect_language(chunk.content)

                # 5. 为 chunk 分配页码
                ChunkerService.assign_page_numbers(all_chunks, page_info, len(markdown_content))
                chunks = all_chunks

                # 6. 上传图片到 MinIO，建立图片 chunk 的关联
                for idx, img in enumerate(extracted_images):
                    img_filename = f"image_{idx}.png"
                    img_object_name = f"{doc.kb_id}/{doc.id}/images/{img_filename}"
                    await minio_svc.upload_file(
                        object_name=img_object_name,
                        data=img["data"],
                        content_type="image/png",
                    )
                    img["url"] = minio_svc.client.presigned_get_object(
                        bucket_name=minio_svc.bucket,
                        object_name=img_object_name,
                        expires=timedelta(days=7),
                    )
                    img["path"] = img_object_name
                    print(f"图片已上传: {img_object_name}")

                # 为 image chunk 关联图片信息（移除 data bytes，已上传 MinIO）
                for chunk in chunks:
                    if chunk.chunk_type == "image" and chunk.metadata.get("image_info"):
                        img_info = chunk.metadata["image_info"]
                        chunk.metadata["image_url"] = img_info.get("url", "")
                        chunk.metadata["image_path"] = img_info.get("path", "")
                        # 移除 data 字段（bytes 无法序列化到 ES）
                        chunk.metadata["image_info"].pop("data", None)

                # 7. 索引中
                doc.status = DocumentStatus.INDEXING.value
                await db.commit()
                _notify(user_id, doc_id, filename, "indexing", 70, f"正在生成嵌入并索引 ({len(chunks)} 个块)...")

                # 8. 获取嵌入模型
                api_key = settings.get_api_key(settings.DEFAULT_EMBED_PROVIDER)
                api_base = settings.get_api_base(settings.DEFAULT_EMBED_PROVIDER)
                embed_provider = model_registry.get_text_embed(
                    provider=settings.DEFAULT_EMBED_PROVIDER,
                    model=settings.DEFAULT_EMBED_MODEL,
                    api_key=api_key,
                    api_base=api_base,
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

                # 11. 完成
                doc.status = DocumentStatus.COMPLETED.value
                doc.chunk_count = indexed_count
                await db.commit()
                img_info = f", {len(extracted_images)} 张图片" if extracted_images else ""
                _notify(user_id, doc_id, filename, "completed", 100,
                        f"处理完成 ({indexed_count} 个块{img_info}已索引)")

                return {"status": "completed", "doc_id": doc_id, "chunks": indexed_count}

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            doc.status = DocumentStatus.FAILED.value
            doc.error_message = str(e)[:500]
            await db.commit()
            _notify(user_id, doc_id, filename, "failed", 0, f"处理失败: {str(e)[:100]}")
            raise
