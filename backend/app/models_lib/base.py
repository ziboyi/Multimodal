from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMProvider(ABC):
    """大语言模型接口"""

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """非流式对话"""
        ...

    @abstractmethod
    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        """流式对话"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class TextEmbedProvider(ABC):
    """文本嵌入接口"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量"""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class VisionEmbedProvider(ABC):
    """视觉嵌入接口"""

    @abstractmethod
    async def embed(self, images: list[bytes]) -> list[list[float]]:
        """将图片列表转为向量"""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class VisionLLMProvider(ABC):
    """多模态视觉语言模型接口"""

    @abstractmethod
    async def describe(self, image: bytes, prompt: str = "描述这张图片") -> str:
        """生成图片描述"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
