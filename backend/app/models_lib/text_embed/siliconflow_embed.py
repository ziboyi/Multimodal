"""
SiliconFlow 远程嵌入 API
支持批量并行调用，自动重试，速率控制
"""
from typing import List
import asyncio
import httpx
from app.models_lib.base import TextEmbedProvider


class SiliconFlowEmbed(TextEmbedProvider):
    """SiliconFlow 嵌入 API 封装"""

    def __init__(self, model_name: str = "BAAI/bge-m3",
                 api_key: str = "", api_base: str = "https://api.siliconflow.cn/v1",
                 batch_size: int = 16, max_concurrency: int = 1,
                 max_retries: int = 5, retry_base_delay: float = 2.0):
        self._model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self._dim = 1024
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入（自动分批+串行+重试）"""
        if not texts:
            return []

        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def embed_batch(batch, batch_idx):
            async with semaphore:
                for attempt in range(self.max_retries):
                    try:
                        result = await self._embed_single(batch)
                        await asyncio.sleep(0.5)
                        return result
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 429:
                            wait = self.retry_base_delay * (2 ** attempt)
                            print(f"[SiliconFlow] 429 速率限制，批次 {batch_idx} 第 {attempt + 1} 次重试，等待 {wait:.1f}s")
                            await asyncio.sleep(wait)
                        else:
                            raise
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            wait = self.retry_base_delay * (2 ** attempt)
                            print(f"[SiliconFlow] 请求失败 ({e})，批次 {batch_idx} 第 {attempt + 1} 次重试，等待 {wait:.1f}s")
                            await asyncio.sleep(wait)
                        else:
                            raise
                raise Exception(f"[SiliconFlow] 批次 {batch_idx} 超过最大重试次数")

        tasks = [embed_batch(b, idx) for idx, b in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        embeddings = []
        for batch_result in results:
            embeddings.extend(batch_result)

        return embeddings

    async def _embed_single(self, texts: List[str]) -> List[List[float]]:
        """单次 API 调用"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": texts,
                    "model": self._model_name,
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            embeddings.append(item["embedding"])

        if embeddings:
            self._dim = len(embeddings[0])

        return embeddings
