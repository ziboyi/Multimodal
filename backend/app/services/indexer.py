"""
Elasticsearch 索引服务
"""
from typing import List, Optional
from elasticsearch import AsyncElasticsearch
from app.core.config import settings
from app.services.chunker import TextChunk


class IndexerService:
    """ES 索引服务"""

    def __init__(self):
        self.client = AsyncElasticsearch(
            hosts=[settings.ELASTICSEARCH_URL_COMPUTED],
            basic_auth=(settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD),
        )

    def _get_index_name(self, kb_id: str) -> str:
        return f"kb_{kb_id.replace('-', '_')}"

    async def create_index(self, kb_id: str, dim: int) -> None:
        """创建知识库索引"""
        index_name = self._get_index_name(kb_id)
        exists = await self.client.indices.exists(index=index_name)
        if exists:
            return

        mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "kb_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "chunk_type": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "cn": {"type": "text", "analyzer": "standard"},
                            "en": {"type": "text", "analyzer": "english"}
                        }
                    },
                    "language": {"type": "keyword"},
                    "dense_vector": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "image_url": {"type": "keyword"},
                    "image_caption": {"type": "text"},
                    "page_number": {"type": "integer"},
                    "section_heading": {"type": "text"},
                    "document_name": {"type": "keyword"},
                    "document_path": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": True},
                    "created_at": {"type": "date"}
                }
            }
        }

        await self.client.indices.create(index=index_name, body=mapping)

    async def index_chunks(self, kb_id: str, doc_id: str, user_id: str,
                           document_name: str, document_path: str,
                           chunks: List[TextChunk],
                           embeddings: list[list[float]],
                           dim: int) -> int:
        """批量索引文本块"""
        await self.create_index(kb_id, dim)
        index_name = self._get_index_name(kb_id)

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        operations = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            operations.append({"index": {"_index": index_name}})
            operations.append({
                "chunk_id": f"{doc_id}_{chunk.chunk_index}",
                "doc_id": doc_id,
                "kb_id": kb_id,
                "user_id": user_id,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "language": chunk.metadata.get("language", "unknown"),
                "dense_vector": embedding,
                "image_url": chunk.metadata.get("image_url", ""),
                "image_caption": chunk.metadata.get("image_caption", ""),
                "page_number": chunk.page_number,
                "section_heading": chunk.metadata.get("section_heading", ""),
                "document_name": document_name,
                "document_path": document_path,
                "metadata": chunk.metadata,
                "created_at": now
            })

        if operations:
            response = await self.client.bulk(operations=operations, refresh="wait_for")
            if response.get("errors"):
                failed = sum(1 for item in response["items"] if "error" in item.get("index", {}))
                return len(chunks) - failed
        return len(chunks)

    async def delete_document(self, kb_id: str, doc_id: str) -> None:
        """删除文档的所有块"""
        index_name = self._get_index_name(kb_id)
        await self.client.delete_by_query(
            index=index_name,
            body={"query": {"term": {"doc_id": doc_id}}},
            refresh=True
        )

    async def delete_kb_index(self, kb_id: str) -> None:
        """删除整个知识库索引"""
        index_name = self._get_index_name(kb_id)
        await self.client.indices.delete(index=index_name, ignore=[404])

    async def close(self):
        await self.client.close()
