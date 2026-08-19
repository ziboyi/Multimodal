#!/usr/bin/env python3
"""
Phase 3: 融合检索 — 文本检索(BGE-M3 Hybrid) + 多模态统一检索(Qwen3-VL)
使用 RRF (Reciprocal Rank Fusion) 整合两路结果，提升 Recall
"""

import asyncio
import base64
import json
import math
import os
import time
import uuid
from collections import defaultdict
from typing import Dict, List

import numpy as np
import requests
from elasticsearch import AsyncElasticsearch

# ============================================================
# 配置
# ============================================================
ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "changeme"

SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl"
)
TEXT_MODEL = "BAAI/bge-m3"
TEXT_DIM = 1024
MM_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
MM_DIM = 4096
EMBED_API_URL = "https://api.siliconflow.cn/v1/embeddings"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "mm_eval")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "images")
COCO_BASE_URL = "http://images.cocodataset.org/val2017"

NUM_DOCS = 200
TOP_K = [5, 10, 20, 50]
METRIC_K = [5, 10]

USER_ID = "eval_user"
KB_ID_TEXT = str(uuid.uuid4())
KB_ID_MM = str(uuid.uuid4())

# ============================================================
# Embedding
# ============================================================
def embed_text_bge(texts: List[str]) -> List[List[float]]:
    """BGE-M3 文本 embedding"""
    resp = requests.post(
        EMBED_API_URL,
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": TEXT_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def embed_text_qwen3vl(text: str) -> List[float]:
    """Qwen3-VL 文本 embedding"""
    resp = requests.post(
        EMBED_API_URL,
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MM_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def embed_image_qwen3vl(base64_data: str) -> List[float]:
    """Qwen3-VL 图片 embedding"""
    resp = requests.post(
        EMBED_API_URL,
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MM_MODEL,
            "input": " ",
            "image": f"data:image/jpeg;base64,{base64_data}",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# ============================================================
# ES 操作
# ============================================================
def get_index_name(prefix: str, kb_id: str) -> str:
    return f"{prefix}{kb_id.replace('-', '_')}"


async def create_index(client: AsyncElasticsearch, prefix: str, kb_id: str, dim: int) -> str:
    index_name = get_index_name(prefix, kb_id)
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
                "content": {"type": "text", "analyzer": "standard"},
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": dim,
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


async def hybrid_search_bge(
    client: AsyncElasticsearch,
    index_name: str,
    query_vec: List[float],
    query_text: str,
    top_k: int = 20,
) -> List[Dict]:
    """BGE-M3 混合检索 (knn + bm25 + RRF)"""
    knn_body = {
        "query": {"term": {"user_id": USER_ID}},
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
                "must": [{"multi_match": {"query": query_text, "fields": ["content"]}}],
                "filter": [{"term": {"user_id": USER_ID}}],
            }
        },
        "size": top_k * 2,
    }

    knn_resp, bm25_resp = await asyncio.gather(
        client.search(index=index_name, body=knn_body),
        client.search(index=index_name, body=bm25_body),
    )

    # RRF
    scores = {}
    result_map = {}
    k = 60
    for rank, hit in enumerate(knn_resp["hits"]["hits"]):
        cid = hit["_source"]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        result_map[cid] = hit["_source"]
    for rank, hit in enumerate(bm25_resp["hits"]["hits"]):
        cid = hit["_source"]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in result_map:
            result_map[cid] = hit["_source"]

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for cid in sorted_ids:
        src = result_map[cid]
        results.append({
            "chunk_id": cid,
            "doc_id": src.get("doc_id", ""),
            "score": scores[cid],
        })
    return results


async def knn_search(
    client: AsyncElasticsearch,
    index_name: str,
    query_vec: List[float],
    top_k: int = 20,
) -> List[Dict]:
    """KNN 检索"""
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
        })
    return results


# ============================================================
# RRF 融合
# ============================================================
def rrf_fuse(list_of_results: List[List[Dict]], top_k: int = 20, k: int = 60) -> List[Dict]:
    """多路结果 RRF 融合"""
    scores = {}
    doc_ids = {}

    for results in list_of_results:
        for rank, r in enumerate(results):
            doc_id = r["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_ids[doc_id] = r

    sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    return [{"doc_id": d, "score": scores[d]} for d in sorted_docs]


# ============================================================
# 评测指标
# ============================================================
def dcg_at_k(relevances: List[int], k: int) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevances: List[int], k: int) -> float:
    rc = sum(1 for r in relevances if r > 0)
    return sum(1 for r in relevances[:k] if r > 0) / rc if rc > 0 else 0.0


def avg_precision(relevances: List[int]) -> float:
    hits, s = 0, 0.0
    for i, r in enumerate(relevances):
        if r > 0:
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
        all_ap.append(avg_precision(relevances))

    metrics = {}
    for k in top_k_values:
        metrics[f"nDCG@{k}"] = float(np.mean(all_ndcg[k]))
        metrics[f"Recall@{k}"] = float(np.mean(all_recall[k]))
    metrics["MAP"] = float(np.mean(all_ap))
    return metrics


# ============================================================
# Main
# ============================================================
async def main():
    print("=" * 60)
    print("Phase 3: 融合检索 (BGE-M3 Hybrid + Qwen3-VL)")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    with open(os.path.join(DATA_DIR, "documents.json")) as f:
        documents = json.load(f)[:NUM_DOCS]
    with open(os.path.join(DATA_DIR, "queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(DATA_DIR, "qrels.json")) as f:
        qrels = json.load(f)

    print(f"  Documents: {len(documents)}, Queries: {len(queries)}")

    # Phase 3a: 构建 BGE-M3 文本索引 (Hybrid)
    print("\n[2/6] Building BGE-M3 text index...")
    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        idx_bge = await create_index(client, "eval_fusion_bge_", KB_ID_TEXT, TEXT_DIM)

        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        texts = [doc["content"] for doc in documents]
        text_embs = embed_text_bge(texts)

        ops = []
        for i, (doc, emb) in enumerate(zip(documents, text_embs)):
            ops.append({"index": {"_index": idx_bge}})
            ops.append({
                "chunk_id": f"bge_{doc['doc_id']}",
                "doc_id": doc["doc_id"],
                "kb_id": KB_ID_TEXT,
                "user_id": USER_ID,
                "chunk_index": i,
                "chunk_type": "text",
                "content": doc["content"],
                "dense_vector": emb,
                "image_url": doc.get("image_url", ""),
                "image_caption": doc.get("image_caption", ""),
                "document_name": doc.get("file_name", ""),
                "is_multimodal": False,
                "created_at": now,
            })
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Indexed {len(documents)} docs (BGE-M3)")

        # Phase 3b: 构建 Qwen3-VL 多模态索引
        print("\n[3/6] Building Qwen3-VL multimodal index...")
        idx_mm = await create_index(client, "eval_fusion_mm_", KB_ID_MM, MM_DIM)

        # Download images
        from concurrent.futures import ThreadPoolExecutor
        def download_img(fn):
            path = os.path.join(IMAGES_DIR, fn)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return fn, base64.b64encode(f.read()).decode()
            url = f"{COCO_BASE_URL}/{fn}"
            resp = requests.get(url, timeout=30)
            os.makedirs(IMAGES_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(resp.content)
            return fn, base64.b64encode(resp.content).decode()

        file_names = [doc.get("file_name", "") for doc in documents]
        with ThreadPoolExecutor(max_workers=16) as pool:
            img_data = dict(pool.map(download_img, file_names))
        print(f"  Downloaded {len(img_data)} images")

        def embed_img(doc):
            fn = doc.get("file_name", "")
            if fn in img_data:
                try:
                    emb = embed_image_qwen3vl(img_data[fn])
                    return doc, emb, fn
                except Exception as e:
                    return doc, None, fn
            return doc, None, None

        print("  Embedding images...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            mm_results = list(pool.map(embed_img, documents))

        mm_docs = [(d, e) for d, e, _ in mm_results if e is not None]
        print(f"  Embedded {len(mm_docs)} images in {time.time() - t0:.1f}s")

        ops = []
        for doc, emb in mm_docs:
            ops.append({"index": {"_index": idx_mm}})
            ops.append({
                "chunk_id": f"mm_{doc['doc_id']}",
                "doc_id": doc["doc_id"],
                "kb_id": KB_ID_MM,
                "user_id": USER_ID,
                "chunk_type": "qwen3vl_multimodal",
                "content": doc["content"],
                "dense_vector": emb,
                "image_url": doc.get("image_url", ""),
                "image_caption": doc.get("image_caption", ""),
                "document_name": doc.get("file_name", ""),
                "is_multimodal": True,
                "created_at": now,
            })
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Indexed {len(mm_docs)} docs (Qwen3-VL)")

        # Phase 3c: 三路检索 + 融合
        print("\n[4/6] Running retrieval (BGE-Hybrid + BGE-KNN + Qwen3-VL)...")
        fusion_results = {}
        bge_hybrid_results = {}
        qwen3vl_results = {}
        bge_knn_results = {}

        q_ids = list(queries.keys())[:NUM_DOCS]
        t0 = time.time()

        for qi, q_id in enumerate(q_ids):
            q_text = queries[q_id]
            q_vec_bge = embed_text_bge([q_text])[0]
            q_vec_mm = embed_text_qwen3vl(q_text)

            # 三路并行检索
            bge_hyb, bge_knn, mm_knn = await asyncio.gather(
                hybrid_search_bge(client, idx_bge, q_vec_bge, q_text, top_k=max(TOP_K)),
                knn_search(client, idx_bge, q_vec_bge, top_k=max(TOP_K)),
                knn_search(client, idx_mm, q_vec_mm, top_k=max(TOP_K)),
            )

            bge_hybrid_results[q_id] = bge_hyb
            bge_knn_results[q_id] = bge_knn
            qwen3vl_results[q_id] = mm_knn

            # 融合
            fusion_results[q_id] = rrf_fuse([bge_hyb, mm_knn], top_k=max(TOP_K))

            if (qi + 1) % 50 == 0:
                print(f"  Progress: {qi + 1}/{len(q_ids)}")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/len(q_ids)*1000:.0f}ms/query)")

        # 计算各方法指标
        print("\n[5/6] Computing metrics...")
        metrics_bge_hybrid = compute_metrics(bge_hybrid_results, qrels, TOP_K)
        metrics_bge_knn = compute_metrics(bge_knn_results, qrels, TOP_K)
        metrics_qwen3vl = compute_metrics(qwen3vl_results, qrels, TOP_K)
        metrics_fusion = compute_metrics(fusion_results, qrels, TOP_K)

        # Cleanup
        await client.indices.delete(index=idx_bge, ignore=[404])
        await client.indices.delete(index=idx_mm, ignore=[404])

    finally:
        await client.close()

    # 输出结果
    print("\n" + "=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)

    all_results = {
        "BGE-M3 Hybrid\n  (text hybrid)": metrics_bge_hybrid,
        "BGE-M3 KNN\n  (text semantic)": metrics_bge_knn,
        "Qwen3-VL\n  (unified emb)": metrics_qwen3vl,
        "★ FUSION\n  (RRF: Hybrid+Qwen3VL)": metrics_fusion,
    }

    for name, m in all_results.items():
        print(f"\n{name}:")
        for k in TOP_K:
            print(f"  nDCG@{k}: {m[f'nDCG@{k}']:.4f}  |  Recall@{k}: {m[f'Recall@{k}']:.4f}")
        print(f"  MAP: {m['MAP']:.4f}")

    # 融合提升分析
    print("\n" + "=" * 60)
    print("📈 FUSION IMPROVEMENT over best single method")
    print("=" * 60)
    for k in METRIC_K:
        best_single = max(metrics_bge_hybrid[f'nDCG@{k}'], metrics_qwen3vl[f'nDCG@{k}'])
        fusion_val = metrics_fusion[f'nDCG@{k}']
        delta = fusion_val - best_single
        print(f"  nDCG@{k}: {fusion_val:.4f} vs best={best_single:.4f}  ({delta:+.4f})")

    for k in TOP_K:
        best_single = max(metrics_bge_hybrid[f'Recall@{k}'], metrics_qwen3vl[f'Recall@{k}'])
        fusion_val = metrics_fusion[f'Recall@{k}']
        delta = fusion_val - best_single
        print(f"  Recall@{k}: {fusion_val:.4f} vs best={best_single:.4f}  ({delta:+.4f})")

    # Save
    output = {
        "bge_hybrid": metrics_bge_hybrid,
        "bge_knn": metrics_bge_knn,
        "qwen3vl": metrics_qwen3vl,
        "fusion": metrics_fusion,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_fusion.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
