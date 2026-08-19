#!/usr/bin/env python3
"""
Phase 2b: Qwen3-VL-Embedding-8B 多模态统一 embedding 评测

使用 SiliconFlow 的 Qwen/Qwen3-VL-Embedding-8B 模型:
- 对 COCO 图片生成 image embedding (统一向量空间)
- 对 text query 生成 text embedding (同一向量空间)
- ES 单一 dense_vector 字段做 KNN 检索
- 与纯文本检索 (BGE-M3) 做对比

API 格式 (SiliconFlow 特殊):
  {"model": "...", "input": "text", "image": "data:image/jpeg;base64,..."}
"""

import asyncio
import base64
import json
import os
import time
import uuid
from collections import defaultdict
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from elasticsearch import AsyncElasticsearch

# ============================================================
# 配置
# ============================================================
ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "changeme"
ES_INDEX_PREFIX = "eval_qwen3vl_"

SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl"
)
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
EMBED_DIM = 4096
EMBED_API_URL = "https://api.siliconflow.cn/v1/embeddings"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "mm_eval")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "images")
COCO_BASE_URL = "http://images.cocodataset.org/val2017"

NUM_DOCS = 200
TOP_K = [5, 10, 20, 50]
METRIC_K = [5, 10]

USER_ID = "eval_user"
KB_ID = str(uuid.uuid4())

# ============================================================
# SiliconFlow Qwen3-VL Embedding
# ============================================================
def embed_text(text: str) -> List[float]:
    """文本 embedding"""
    resp = requests.post(
        EMBED_API_URL,
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBED_MODEL,
            "input": text,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def embed_image(base64_data: str) -> List[float]:
    """图片 embedding"""
    resp = requests.post(
        EMBED_API_URL,
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBED_MODEL,
            "input": " ",
            "image": f"data:image/jpeg;base64,{base64_data}",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def download_image(file_name: str) -> str:
    """下载 COCO 图片并返回 base64"""
    local_path = os.path.join(IMAGES_DIR, file_name)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    url = f"{COCO_BASE_URL}/{file_name}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return base64.b64encode(resp.content).decode()


def download_images_parallel(file_names: List[str], max_workers: int = 8):
    """并行下载图片"""
    results = {}

    def _download(fn):
        try:
            b64 = download_image(fn)
            return fn, b64, None
        except Exception as e:
            return fn, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download, fn): fn for fn in file_names}
        for i, future in enumerate(futures):
            fn, b64, err = future.result()
            if err:
                print(f"  ⚠ Failed {fn}: {err}")
            else:
                results[fn] = b64
            if (i + 1) % 50 == 0:
                print(f"  Downloaded {i + 1}/{len(file_names)}")

    return results


# ============================================================
# ES 操作
# ============================================================
def get_index_name(kb_id: str) -> str:
    return f"{ES_INDEX_PREFIX}{kb_id.replace('-', '_')}"


async def create_es_index(client: AsyncElasticsearch, kb_id: str) -> str:
    index_name = get_index_name(kb_id)
    exists = await client.indices.exists(index=index_name)
    if exists:
        await client.indices.delete(index=index_name)

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
                },
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": EMBED_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
                "image_url": {"type": "keyword"},
                "image_caption": {"type": "text"},
                "document_name": {"type": "keyword"},
                "is_multimodal": {"type": "boolean"},
                "created_at": {"type": "date"},
            }
        }
    }
    await client.indices.create(index=index_name, body=mapping)
    return index_name


async def knn_search(
    client: AsyncElasticsearch,
    index_name: str,
    query_vec: List[float],
    top_k: int = 20,
) -> List[Dict]:
    """KNN 检索 (统一向量空间)"""
    body = {
        "query": {"term": {"user_id": USER_ID}},
        "knn": {
            "field": "dense_vector",
            "query_vector": query_vec,
            "k": top_k,
            "num_candidates": top_k * 10,
        },
        "size": top_k,
    }
    resp = await client.search(index=index_name, body=body)
    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "chunk_id": src["chunk_id"],
            "doc_id": src.get("doc_id", ""),
            "score": hit.get("_score", 0),
            "content": src.get("content", "")[:100],
            "is_multimodal": src.get("is_multimodal", False),
        })
    return results


# ============================================================
# 评测指标
# ============================================================
def dcg_at_k(relevances: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / np.log2(i + 2)
    return dcg


def ndcg_at_k(relevances: List[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevances: List[int], k: int) -> float:
    rel_count = sum(1 for r in relevances if r > 0)
    if rel_count == 0:
        return 0.0
    return sum(1 for r in relevances[:k] if r > 0) / rel_count


def average_precision(relevances: List[int]) -> float:
    hits = 0
    s = 0.0
    for i, rel in enumerate(relevances):
        if rel > 0:
            hits += 1
            s += hits / (i + 1)
    return s / hits if hits > 0 else 0.0


def compute_metrics(results: Dict, qrels: Dict, top_k_values: List[int] = None) -> Dict:
    if top_k_values is None:
        top_k_values = METRIC_K

    all_ndcg = defaultdict(list)
    all_recall = defaultdict(list)
    all_ap = []

    for q_id, retrieved in results.items():
        query_qrels = qrels.get(q_id, {})
        relevances = [query_qrels.get(r["doc_id"], 0) for r in retrieved]

        for k in top_k_values:
            all_ndcg[k].append(ndcg_at_k(relevances, k))
            all_recall[k].append(recall_at_k(relevances, k))
        all_ap.append(average_precision(relevances))

    metrics = {}
    for k in top_k_values:
        metrics[f"nDCG@{k}"] = float(np.mean(all_ndcg[k])) if all_ndcg[k] else 0.0
        metrics[f"Recall@{k}"] = float(np.mean(all_recall[k])) if all_recall[k] else 0.0
    metrics["MAP"] = float(np.mean(all_ap)) if all_ap else 0.0
    return metrics


# ============================================================
# Main
# ============================================================
async def main():
    print("=" * 60)
    print("Phase 2b: Qwen3-VL-Embedding-8B 多模态统一检索")
    print("=" * 60)

    # Load data
    print(f"\n[1/5] Loading multimodal dataset...")
    with open(os.path.join(DATA_DIR, "documents.json")) as f:
        documents = json.load(f)
    with open(os.path.join(DATA_DIR, "queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(DATA_DIR, "qrels.json")) as f:
        qrels = json.load(f)

    documents = documents[:NUM_DOCS]
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    # Download images
    print(f"\n[2/5] Downloading COCO images...")
    file_names = [doc.get("file_name", "") for doc in documents if doc.get("file_name")]
    t0 = time.time()
    image_data = download_images_parallel(file_names)
    print(f"  Downloaded {len(image_data)} images in {time.time() - t0:.1f}s")

    # Generate image embeddings
    print(f"\n[3/5] Generating image embeddings ({EMBED_MODEL})...")
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    index_docs = []
    failed_images = 0

    def _embed_one(args):
        i, doc = args
        fn = doc.get("file_name", "")
        if fn not in image_data:
            return None, fn, "no_image"
        try:
            img_emb = embed_image(image_data[fn])
            return (i, doc, img_emb, fn), None, None
        except Exception as e:
            return None, fn, str(e)

    print(f"  Embedding {len(documents)} images in parallel (16 workers)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = list(pool.map(_embed_one, enumerate(documents)))

    for result, fn, err in futures:
        if err:
            failed_images += 1
            continue
        i, doc, img_emb, img_fn = result
        index_docs.append({
            "chunk_id": f"q3vl_{doc['doc_id']}",
            "doc_id": doc["doc_id"],
            "kb_id": KB_ID,
            "user_id": USER_ID,
            "chunk_index": i,
            "chunk_type": "qwen3vl_multimodal",
            "content": doc["content"],
            "dense_vector": img_emb,
            "image_url": doc.get("image_url", ""),
            "image_caption": doc.get("image_caption", ""),
            "document_name": img_fn,
            "is_multimodal": True,
            "created_at": now,
        })

    print(f"  Embedded {len(index_docs)} docs in {time.time() - t0:.1f}s ({failed_images} failed)")

    # Index to ES
    print(f"\n[4/5] Indexing to Elasticsearch...")
    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        index_name = await create_es_index(client, KB_ID)

        operations = []
        for doc in index_docs:
            operations.append({"index": {"_index": index_name}})
            operations.append(doc)

        if operations:
            resp = await client.bulk(operations=operations, refresh="wait_for")
            if resp.get("errors"):
                failed = sum(1 for item in resp["items"] if "error" in item.get("index", {}))
                print(f"  Indexed {len(index_docs) - failed} docs ({failed} failed)")
            else:
                print(f"  Indexed {len(index_docs)} docs OK")

        # Run retrieval
        print(f"\n[5/5] Running Qwen3-VL multimodal retrieval...")
        results = {}
        q_ids = list(queries.keys())[:NUM_DOCS]
        t0 = time.time()

        for qi, q_id in enumerate(q_ids):
            query_text = queries[q_id]
            try:
                query_vec = embed_text(query_text)
            except Exception as e:
                print(f"  ⚠ Query embedding failed for {q_id}: {e}")
                continue

            retrieved = await knn_search(client, index_name, query_vec, top_k=max(TOP_K))
            results[q_id] = retrieved

            if (qi + 1) % 50 == 0:
                print(f"  Progress: {qi + 1}/{len(q_ids)}")

        elapsed = time.time() - t0
        metrics = compute_metrics(results, qrels, TOP_K)
        metrics["avg_query_time_ms"] = (elapsed / max(len(q_ids), 1)) * 1000

        print(f"\n  Qwen3-VL Results:")
        print(f"    Queries: {len(results)} in {elapsed:.1f}s ({metrics['avg_query_time_ms']:.0f}ms/query)")
        for k in TOP_K:
            ndcg = metrics.get(f"nDCG@{k}", 0)
            recall = metrics.get(f"Recall@{k}", 0)
            print(f"    nDCG@{k}: {ndcg:.4f}  |  Recall@{k}: {recall:.4f}")
        print(f"    MAP: {metrics['MAP']:.4f}")

        # Cleanup
        await client.indices.delete(index=index_name, ignore=[404])

    finally:
        await client.close()

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "results_qwen3vl.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")

    return metrics


if __name__ == "__main__":
    asyncio.run(main())
