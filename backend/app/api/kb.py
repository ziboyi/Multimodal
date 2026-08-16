from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.dependencies import get_current_user_id
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentStatus
from app.schemas.kb import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
)
from app.schemas.document import DocumentResponse, DocumentListResponse
from typing import List
import uuid

router = APIRouter(prefix="/kb", tags=["知识库"])


# ===== 知识库 CRUD =====

@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_kb(
    req: KnowledgeBaseCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """创建知识库"""
    kb = KnowledgeBase(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=req.name,
        description=req.description,
        embed_model_id=req.embed_model_id,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


@router.get("", response_model=List[KnowledgeBaseResponse])
async def list_kb(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """获取当前用户的知识库列表"""
    result = await db.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.user_id == user_id)
        .order_by(KnowledgeBase.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_kb(
    kb_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """获取知识库详情"""
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id,
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.patch("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_kb(
    kb_id: str,
    req: KnowledgeBaseUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """更新知识库"""
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id,
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    update_data = req.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(kb, field, value)

    await db.commit()
    await db.refresh(kb)
    return kb


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(
    kb_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """删除知识库"""
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id,
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    await db.delete(kb)
    await db.commit()


# ===== 文档上传与管理 =====

@router.post("/{kb_id}/documents")
async def upload_documents(
    kb_id: str,
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """上传文档到知识库（异步处理）"""
    # 验证知识库归属
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id,
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    from app.services.minio_client import MinioService
    from app.services.tasks import process_document_task

    minio_svc = MinioService()
    uploaded_docs = []

    for file in files:
        doc_id = str(uuid.uuid4())
        file_type = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        file_data = await file.read()
        file_size = len(file_data)

        # 上传到 MinIO
        object_name = f"{kb_id}/{doc_id}/{file.filename}"
        await minio_svc.upload_file(
            object_name=object_name,
            data=file_data,
            content_type=file.content_type or "application/octet-stream",
        )

        # 创建文档记录
        doc = Document(
            id=doc_id,
            kb_id=kb_id,
            user_id=user_id,
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            file_path=object_name,
            status=DocumentStatus.PENDING.value,
        )
        db.add(doc)
        uploaded_docs.append(doc)

    await db.commit()

    # 触发异步处理任务
    for doc in uploaded_docs:
        process_document_task.delay(doc.id)

    return {
        "message": f"已上传 {len(uploaded_docs)} 个文件，正在处理中",
        "documents": [{"id": d.id, "filename": d.filename} for d in uploaded_docs],
    }


@router.get("/{kb_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    kb_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """获取知识库下的文档列表"""
    # 验证知识库归属
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    result = await db.execute(
        select(Document)
        .where(Document.kb_id == kb_id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()

    return DocumentListResponse(
        total=len(documents),
        items=[DocumentResponse.model_validate(doc) for doc in documents],
    )


@router.delete("/{kb_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb_id: str,
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """删除文档"""
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.kb_id == kb_id,
            Document.user_id == user_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 删除 ES 索引
    from app.services.indexer import IndexerService
    indexer = IndexerService()
    await indexer.delete_document(kb_id=kb_id, doc_id=doc_id)
    await indexer.close()

    # 删除 MinIO 文件
    from app.services.minio_client import MinioService
    minio_svc = MinioService()
    await minio_svc.delete_file(doc.file_path)

    # 删除记录
    await db.delete(doc)
    await db.commit()
