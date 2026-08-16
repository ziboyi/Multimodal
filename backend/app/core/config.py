from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ===== 应用 =====
    APP_NAME: str = "Multimodal RAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    API_V1_PREFIX: str = "/api"

    # ===== Token =====
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ===== 加密 =====
    API_ENCRYPTION_KEY: str = ""  # Fernet key for API key encryption

    # ===== 数据库 =====
    POSTGRES_USER: str = "multimodal"
    POSTGRES_PASSWORD: str = "changeme"
    POSTGRES_DB: str = "multimodal"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # ===== Redis =====
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = ""

    @property
    def REDIS_URL_COMPUTED(self) -> str:
        return self.REDIS_URL or f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    # ===== Elasticsearch =====
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_URL: str = ""
    ELASTICSEARCH_USERNAME: str = "elastic"
    ELASTICSEARCH_PASSWORD: str = "changeme"

    @property
    def ELASTICSEARCH_URL_COMPUTED(self) -> str:
        return self.ELASTICSEARCH_URL or f"http://{self.ELASTICSEARCH_HOST}:{self.ELASTICSEARCH_PORT}"

    # ===== MinIO =====
    MINIO_HOST: str = "localhost"
    MINIO_PORT: int = 9000
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_BUCKET: str = "multimodal"
    MINIO_SECURE: bool = False

    @property
    def MINIO_ENDPOINT(self) -> str:
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"

    # ===== 默认模型配置 =====
    DEFAULT_LLM_PROVIDER: str = "longcat"
    DEFAULT_LLM_MODEL: str = "LongCat-Flash-Chat"
    DEFAULT_EMBED_PROVIDER: str = "bge-m3"
    DEFAULT_EMBED_MODEL: str = "BAAI/BGE-M3"
    DEFAULT_VISION_EMBED_PROVIDER: str = "clip"
    DEFAULT_VISION_EMBED_MODEL: str = "openai/clip-vit-base-32"
    DEFAULT_VISION_LLM_PROVIDER: str = "longcat"
    DEFAULT_VISION_LLM_MODEL: str = "LongCat-Flash-Chat"

    # ===== 模型 API Keys =====
    OPENAI_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    LONGCAT_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # ===== LongCat 配置 =====
    LONGCAT_API_BASE: str = "https://api.longcat.chat/openai"

    def get_api_key(self, provider: str) -> str:
        """根据提供商名称获取 API Key"""
        key_map = {
            "openai": self.OPENAI_API_KEY,
            "qwen": self.DASHSCOPE_API_KEY,
            "deepseek": self.DEEPSEEK_API_KEY,
            "longcat": self.LONGCAT_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "gemini": self.GEMINI_API_KEY,
        }
        return key_map.get(provider, "")

    def get_api_base(self, provider: str) -> str | None:
        """根据提供商名称获取 API Base URL"""
        base_map = {
            "longcat": self.LONGCAT_API_BASE,
        }
        return base_map.get(provider)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
