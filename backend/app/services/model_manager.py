"""
模型管理注册中心
根据配置动态加载和缓存模型实例
"""
from typing import Dict, Optional
from app.models_lib.base import LLMProvider, TextEmbedProvider, VisionEmbedProvider, VisionLLMProvider
from app.core.config import settings


class ModelRegistry:
    """模型注册中心 - 管理所有模型实例"""

    def __init__(self):
        self._llm_cache: Dict[str, LLMProvider] = {}
        self._text_embed_cache: Dict[str, TextEmbedProvider] = {}
        self._vision_embed_cache: Dict[str, VisionEmbedProvider] = {}
        self._vision_llm_cache: Dict[str, VisionLLMProvider] = {}

    def get_llm(self, provider: str, model: str, api_key: str = "", api_base: str | None = None) -> LLMProvider:
        cache_key = f"llm:{provider}:{model}"
        if cache_key not in self._llm_cache:
            if not api_key:
                api_key = settings.get_api_key(provider)
            if not api_base:
                api_base = settings.get_api_base(provider)
            self._llm_cache[cache_key] = self._create_llm(provider, model, api_key, api_base)
        return self._llm_cache[cache_key]

    def _create_llm(self, provider: str, model: str, api_key: str, api_base: str | None) -> LLMProvider:
        if provider == "openai":
            from app.models_lib.llm.openai_llm import OpenAILLM
            return OpenAILLM(model=model, api_key=api_key, api_base=api_base)
        elif provider == "qwen":
            from app.models_lib.llm.qwen_llm import QwenLLM
            return QwenLLM(model=model, api_key=api_key, api_base=api_base)
        elif provider == "deepseek":
            from app.models_lib.llm.deepseek_llm import DeepSeekLLM
            return DeepSeekLLM(model=model, api_key=api_key, api_base=api_base)
        elif provider == "longcat":
            from app.models_lib.llm.longcat_llm import LongCatLLM
            return LongCatLLM(model=model, api_key=api_key, api_base=api_base)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def get_text_embed(self, provider: str, model: str, api_key: str = "", api_base: str | None = None) -> TextEmbedProvider:
        cache_key = f"embed:{provider}:{model}"
        if cache_key not in self._text_embed_cache:
            if not api_key:
                api_key = settings.get_api_key(provider)
            self._text_embed_cache[cache_key] = self._create_text_embed(provider, model, api_key, api_base)
        return self._text_embed_cache[cache_key]

    def _create_text_embed(self, provider: str, model: str, api_key: str, api_base: str | None) -> TextEmbedProvider:
        if provider == "bge-m3":
            from app.models_lib.text_embed.bge_embed import BGEEmbed
            return BGEEmbed(model_name=model)
        elif provider == "openai":
            from app.models_lib.text_embed.openai_embed import OpenAIEmbed
            return OpenAIEmbed(model=model, api_key=api_key, api_base=api_base)
        elif provider == "siliconflow":
            from app.models_lib.text_embed.siliconflow_embed import SiliconFlowEmbed
            return SiliconFlowEmbed(
                model_name=model,
                api_key=api_key,
                api_base=api_base or "https://api.siliconflow.cn/v1",
            )
        else:
            raise ValueError(f"Unsupported embed provider: {provider}")

    def get_vision_embed(self, provider: str, model: str) -> VisionEmbedProvider:
        cache_key = f"vision_embed:{provider}:{model}"
        if cache_key not in self._vision_embed_cache:
            self._vision_embed_cache[cache_key] = self._create_vision_embed(provider, model)
        return self._vision_embed_cache[cache_key]

    def _create_vision_embed(self, provider: str, model: str) -> VisionEmbedProvider:
        if provider == "clip":
            from app.models_lib.vision_embed.clip_embed import CLIPEmbed
            return CLIPEmbed(model_name=model)
        else:
            raise ValueError(f"Unsupported vision embed provider: {provider}")

    def get_vision_llm(self, provider: str, model: str, api_key: str = "", api_base: str | None = None) -> VisionLLMProvider:
        cache_key = f"vision_llm:{provider}:{model}"
        if cache_key not in self._vision_llm_cache:
            if not api_key:
                api_key = settings.get_api_key(provider)
            if not api_base:
                api_base = settings.get_api_base(provider)
            self._vision_llm_cache[cache_key] = self._create_vision_llm(provider, model, api_key, api_base)
        return self._vision_llm_cache[cache_key]

    def _create_vision_llm(self, provider: str, model: str, api_key: str, api_base: str | None) -> VisionLLMProvider:
        if provider == "openai":
            from app.models_lib.vision_llm.gpt4o_vision import GPT4oVision
            return GPT4oVision(model=model, api_key=api_key, api_base=api_base)
        elif provider == "qwen":
            from app.models_lib.vision_llm.qwen_vl import QwenVL
            return QwenVL(model=model, api_key=api_key, api_base=api_base)
        elif provider == "longcat":
            from app.models_lib.llm.longcat_llm import LongCatLLM
            return LongCatLLM(model=model, api_key=api_key, api_base=api_base)
        else:
            raise ValueError(f"Unsupported vision LLM provider: {provider}")


# 全局单例
model_registry = ModelRegistry()
