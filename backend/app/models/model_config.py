from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)  # NULL = 全局配置
    
    # 模型类型
    provider_type = Column(String, nullable=False)  # llm, text_embed, vision_embed, vision_llm
    provider_name = Column(String, nullable=False)  # openai, qwen, deepseek, etc.
    model_name = Column(String, nullable=False)  # gpt-4o, BAAI/BGE-M3, etc.
    
    # API 配置（加密存储）
    api_key_encrypted = Column(Text, nullable=True)
    api_base = Column(String, nullable=True)
    extra_config = Column(JSON, nullable=True)  # 额外配置
    
    # 状态
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
