from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text, Enum
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    kb_id = Column(String, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # 文件信息
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, docx, pptx, md, etc.
    file_size = Column(Integer, default=0)
    file_path = Column(String, nullable=True)  # MinIO 路径
    
    # 解析结果
    markdown_content = Column(Text, nullable=True)
    
    # 状态
    status = Column(String, default=DocumentStatus.PENDING.value)
    error_message = Column(Text, nullable=True)
    
    # 统计
    chunk_count = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
