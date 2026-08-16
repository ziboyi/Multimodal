from typing import Optional
from sentence_transformers import SentenceTransformer
from app.models_lib.base import TextEmbedProvider


class BGEEmbed(TextEmbedProvider):
    """BGE-M3 文本嵌入实现（本地模型）"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # sentence-transformers 是同步的，用线程池
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, normalize_embeddings=True),
        )
        return embeddings.tolist()
