from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.models_lib.base import LLMProvider


class DeepSeekLLM(LLMProvider):
    """DeepSeek LLM 实现（兼容 OpenAI 格式）"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = "",
        api_base: str = "https://api.deepseek.com/v1",
    ):
        self._model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(self, messages: list[dict], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
