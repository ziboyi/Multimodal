from openai import AsyncOpenAI
from app.models_lib.base import TextEmbedProvider


class OpenAIEmbed(TextEmbedProvider):
    """OpenAI 文本嵌入实现"""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str = "",
        api_base: str | None = None,
        dim: int = 3072,
    ):
        self._model = model
        self._dim = dim
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]
