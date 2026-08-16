"""
混合检索服务
向量检索 + BM25 + 应用层 RRF 融合
"""
from typing import List, Optional
from elasticsearch import AsyncElasticsearch
from app.core.config import settings
from app.services.model_manager import model_registry


class SearchResult:
    def __init__(self, chunk_id: str, doc_id: str, kb_id: str,
                 document_name: str, content: str, chunk_type: str,
                 page_number: int | None, score: float,
                 image_url: str = "", highlight: str = ""):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.kb_id = kb_id
        self.document_name = document_name
        self.content = content
        self.chunk_type = chunk_type
        self.page_number = page_number
        self.score = score
        self.image_url = image_url
        self.highlight = highlight


class RetrieverService:
    """混合检索服务"""

    def __init__(self):
        self.client = AsyncElasticsearch(
            hosts=[settings.ELASTICSEARCH_URL_COMPUTED],
            basic_auth=(settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD),
        )

    def _get_index_name(self, kb_id: str) -> str:
        """索引名格式与 indexer.py 保持一致"""
        return f"kb_{kb_id.replace('-', '_')}"

    async def search(self, user_id: str, query: str,
                     kb_ids: Optional[List[str]] = None,
                     top_k: int = 10,
                     search_mode: str = "hybrid") -> List[SearchResult]:
        """混合检索入口"""
        # 确定搜索的索引
        if kb_ids:
            indices = ",".join(self._get_index_name(kid) for kid in kb_ids)
        else:
            indices = "kb_*"

        if search_mode == "semantic":
            return await self._semantic_search(indices, query, user_id, top_k)
        elif search_mode == "keyword":
            return await self._bm25_search(indices, query, user_id, top_k)
        else:
            return await self._hybrid_search(indices, query, user_id, top_k)

    async def _semantic_search(self, indices: str, query: str,
                               user_id: str, top_k: int) -> List[SearchResult]:
        """纯向量检索"""
        embed_provider = model_registry.get_text_embed(
            settings.DEFAULT_EMBED_PROVIDER, settings.DEFAULT_EMBED_MODEL
        )
        query_vec = (await embed_provider.embed([query]))[0]

        body = {
            "query": {"term": {"user_id": user_id}},
            "knn": {
                "field": "dense_vector",
                "query_vector": query_vec,
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "size": top_k,
        }

        response = await self.client.search(index=indices, body=body)
        return self._parse_results(response)

    async def _bm25_search(self, indices: str, query: str,
                           user_id: str, top_k: int) -> List[SearchResult]:
        """纯关键词检索"""
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {"query": query, "fields": ["content", "content.cn", "content.en"]}}
                    ],
                    "filter": [{"term": {"user_id": user_id}}],
                }
            },
            "highlight": {"fields": {"content": {}}},
            "size": top_k,
        }

        response = await self.client.search(index=indices, body=body)
        return self._parse_results(response)

    async def _hybrid_search(self, indices: str, query: str,
                             user_id: str, top_k: int) -> List[SearchResult]:
        """混合检索：kNN + BM25 + 应用层 RRF 融合"""
        embed_provider = model_registry.get_text_embed(
            settings.DEFAULT_EMBED_PROVIDER, settings.DEFAULT_EMBED_MODEL
        )
        query_vec = (await embed_provider.embed([query]))[0]

        # 并行执行 kNN 和 BM25
        knn_body = {
            "query": {"term": {"user_id": user_id}},
            "knn": {
                "field": "dense_vector",
                "query_vector": query_vec,
                "k": top_k * 2,
                "num_candidates": top_k * 20,
            },
            "size": top_k * 2,
        }

        bm25_body = {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {"query": query, "fields": ["content", "content.cn", "content.en"]}}
                    ],
                    "filter": [{"term": {"user_id": user_id}}],
                }
            },
            "highlight": {"fields": {"content": {}}},
            "size": top_k * 2,
        }

        import asyncio
        knn_resp, bm25_resp = await asyncio.gather(
            self.client.search(index=indices, body=knn_body),
            self.client.search(index=indices, body=bm25_body),
        )

        knn_results = self._parse_results(knn_resp)
        bm25_results = self._parse_results(bm25_resp)

        # RRF 融合
        return self._rrf_fusion(knn_results, bm25_results, top_k)

    def _rrf_fusion(self, knn_results: List[SearchResult],
                    bm25_results: List[SearchResult], top_k: int,
                    k: int = 60) -> List[SearchResult]:
        """应用层 Reciprocal Rank Fusion"""
        scores = {}  # chunk_id -> score
        result_map = {}  # chunk_id -> SearchResult

        for rank, r in enumerate(knn_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(bm25_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            r = result_map[cid]
            r.score = scores[cid]
            results.append(r)

        return results

    def _parse_results(self, response: dict) -> List[SearchResult]:
        """解析 ES 响应"""
        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            highlight = ""
            if "highlight" in hit and "content" in hit["highlight"]:
                highlight = " ... ".join(hit["highlight"]["content"])
            results.append(SearchResult(
                chunk_id=src.get("chunk_id", ""),
                doc_id=src.get("doc_id", ""),
                kb_id=src.get("kb_id", ""),
                document_name=src.get("document_name", ""),
                content=src.get("content", ""),
                chunk_type=src.get("chunk_type", "text"),
                page_number=src.get("page_number"),
                score=hit.get("_score", 0.0),
                image_url=src.get("image_url", ""),
                highlight=highlight,
            ))
        return results

    async def close(self):
        await self.client.close()
