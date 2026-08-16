"""
RAG Pipeline 服务
检索 + 上下文构建 + LLM 生成
"""
from typing import AsyncGenerator, Optional, List
from app.services.retriever import RetrieverService, SearchResult
from app.services.model_manager import model_registry
from app.core.config import settings


# 系统提示词
SYSTEM_PROMPT = """你是一个智能问答助手，基于用户的知识库内容回答问题。

规则：
1. 仅基于提供的参考资料回答问题
2. 如果参考资料中没有相关信息，请明确告知用户
3. 回答要简洁、准确、有条理
4. 引用参考资料时标注来源
5. 使用与用户问题相同的语言回答"""


class RAGPipeline:
    """RAG 检索增强生成 Pipeline"""

    def __init__(self):
        self.retriever = RetrieverService()

    async def generate(self, user_id: str, query: str,
                       kb_ids: Optional[List[str]] = None,
                       top_k: int = 5) -> AsyncGenerator[str, None]:
        """流式生成回答"""
        # 1. 检索
        results = await self.retriever.search(
            user_id=user_id,
            query=query,
            kb_ids=kb_ids,
            top_k=top_k
        )

        # 2. 构建上下文
        context = self._build_context(results)

        # 3. 构建消息
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"## 参考资料\n\n{context}\n\n## 用户问题\n\n{query}"}
        ]

        # 4. 调用 LLM
        llm = model_registry.get_llm(
            provider=settings.DEFAULT_LLM_PROVIDER,
            model=settings.DEFAULT_LLM_MODEL
        )

        async for chunk in llm.chat_stream(messages):
            yield chunk

    async def generate_with_references(self, user_id: str, query: str,
                                        kb_ids: Optional[List[str]] = None,
                                        top_k: int = 5) -> dict:
        """非流式生成，返回回答和引用"""
        results = await self.retriever.search(
            user_id=user_id,
            query=query,
            kb_ids=kb_ids,
            top_k=top_k
        )

        context = self._build_context(results)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"## 参考资料\n\n{context}\n\n## 用户问题\n\n{query}"}
        ]

        llm = model_registry.get_llm(
            provider=settings.DEFAULT_LLM_PROVIDER,
            model=settings.DEFAULT_LLM_MODEL
        )

        answer = await llm.chat(messages)

        references = [
            {
                "document_name": r.document_name,
                "page_number": r.page_number,
                "content": r.content[:200],
                "chunk_id": r.chunk_id
            }
            for r in results
        ]

        return {"answer": answer, "references": references}

    def _build_context(self, results: List[SearchResult]) -> str:
        """构建上下文"""
        if not results:
            return "无相关参考资料。"

        context_parts = []
        for i, r in enumerate(results, 1):
            ref = f"[来源{i}] {r.document_name}"
            if r.page_number:
                ref += f" (第{r.page_number}页)"
            context_parts.append(f"{ref}:\n{r.content}")

        return "\n\n".join(context_parts)