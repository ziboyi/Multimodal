#!/usr/bin/env python3
"""
Phase 4: 两阶段检索 — 召回 + Caption 重排序

Stage 1 (Recall): Qwen3-VL 统一 embedding KNN → top-50 候选
Stage 2 (Rerank): BGE-M3 caption 文本相似度 → 重排序 top-50 → top-k

对比:
- BGE-M3 Hybrid (baseline)
- Qwen3-VL KNN only
- Qwen3-VL recall + caption rerank (两阶段)
- BGE-M3 recall + caption rerank
"""

import asyncio
import base64
import json
import math
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
RECALL_K = 50  # 第一阶段召回数量

USER_ID = "eval_user"
KB_ID_TEXT = str(uuid.uuid4())
KB_ID_MM = str(uuid.uuid4())

# ============================================================
# Embedding
# ============================================================
def embed_batch(texts: List[str], model: str) -> List[List[float]]:
    resp = requests.post(
        EMBED_API_URL,
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def embed_image(base64_data: str) -> List[float]:
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
# ES
# ============================================================
def idx_name(prefix, kb_id):
    return f"{prefix}{kb_id.replace('-', '_')}"


async def create_idx(client, prefix, kb_id, dim):
    name = idx_name(prefix, kb_id)
    if await client.indices.exists(index=name):
        await client.indices.delete(index=name)
    mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "kb_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
                "caption": {"type": "text"},
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "document_name": {"type": "keyword"},
                "created_at": {"type": "date"},
            }
        }
    }
    await client.indices.create(index=name, body=mapping)
    return name


async def knn_search(client, index, vec, top_k):
    body = {
        "query": {"term": {"user_id": USER_ID}},
        "knn": {"field": "dense_vector", "query_vector": vec, "k": top_k, "num_candidates": top_k * 10},
        "size": top_k,
    }
    resp = await client.search(index=index, body=body)
    return [(hit["_source"]["doc_id"], hit["_source"].get("caption", hit["_source"].get("content", "")), hit.get("_score", 0))
            for hit in resp["hits"]["hits"]]


async def hybrid_search_bge(client, index, vec, text, top_k):
    knn_body = {
        "query": {"term": {"user_id": USER_ID}},
        "knn": {"field": "dense_vector", "query_vector": vec, "k": top_k * 2, "num_candidates": top_k * 20},
        "size": top_k * 2,
    }
    bm25_body = {
        "query": {"bool": {"must": [{"multi_match": {"query": text, "fields": ["content", "caption"]}}],
                           "filter": [{"term": {"user_id": USER_ID}}]}},
        "size": top_k * 2,
    }
    r1, r2 = await asyncio.gather(
        client.search(index=index, body=knn_body),
        client.search(index=index, body=bm25_body),
    )
    scores, docs = {}, {}
    k = 60
    for rank, hit in enumerate(r1["hits"]["hits"]):
        d = hit["_source"]
        did = d["doc_id"]
        scores[did] = scores.get(did, 0) + 1/(k+rank+1)
        docs[did] = (d.get("caption", d.get("content", "")), hit.get("_score", 0))
    for rank, hit in enumerate(r2["hits"]["hits"]):
        d = hit["_source"]
        did = d["doc_id"]
        scores[did] = scores.get(did, 0) + 1/(k+rank+1)
        if did not in docs:
            docs[did] = (d.get("caption", d.get("content", "")), hit.get("_score", 0))
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    return [(did, docs[did][0], scores[did]) for did in sorted_ids]


# ============================================================
# 评测指标
# ============================================================
def dcg(rel, k):
    return sum((2**r - 1) / math.log2(i+2) for i, r in enumerate(rel[:k]))

def ndcg(rel, k):
    d = dcg(rel, k)
    i = dcg(sorted(rel, reverse=True), k)
    return d/i if i > 0 else 0.0

def recall(rel, k):
    rc = sum(1 for r in rel if r > 0)
    return sum(1 for r in rel[:k] if r > 0) / rc if rc > 0 else 0.0

def ap(rel):
    hits, s = 0, 0.0
    for i, r in enumerate(rel):
        if r > 0:
            hits += 1; s += hits/(i+1)
    return s/hits if hits > 0 else 0.0

def compute_metrics(results, qrels, top_k_vals):
    metrics = {}
    for k in top_k_vals:
        nd, rc = [], []
        for q_id, retrieved in results.items():
            rels = [qrels.get(q_id, {}).get(r[0], 0) for r in retrieved]
            nd.append(ndcg(rels, k))
            rc.append(recall(rels, k))
        metrics[f"nDCG@{k}"] = float(np.mean(nd))
        metrics[f"Recall@{k}"] = float(np.mean(rc))
    aps = []
    for q_id, retrieved in results.items():
        rels = [qrels.get(q_id, {}).get(r[0], 0) for r in retrieved]
        aps.append(ap(rels))
    metrics["MAP"] = float(np.mean(aps))
    return metrics


# ============================================================
# Main
# ============================================================
async def main():
    print("=" * 60)
    print("Phase 4: 两阶段检索 — 召回 + Caption 重排序")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    with open(os.path.join(DATA_DIR, "documents.json")) as f:
        docs = json.load(f)[:NUM_DOCS]
    with open(os.path.join(DATA_DIR, "queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(DATA_DIR, "qrels.json")) as f:
        qrels = json.load(f)

    # Build doc_id -> caption mapping
    doc_caption = {d["doc_id"]: d.get("caption", d["captions"][0] if d.get("captions") else d["content"][:100])
                   for d in docs}
    doc_content = {d["doc_id"]: d["content"] for d in docs}

    print(f"  {len(docs)} docs, {len(queries)} queries")

    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        # === Index 1: BGE-M3 text (content + caption) ===
        print("\n[2/5] Building BGE-M3 text index...")
        idx_bge = await create_idx(client, "eval_rerank_bge_", KB_ID_TEXT, TEXT_DIM)
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        bge_texts = [d["content"] for d in docs]
        bge_embs = embed_batch(bge_texts, TEXT_MODEL)

        ops = []
        for i, (d, emb) in enumerate(zip(docs, bge_embs)):
            ops.append({"index": {"_index": idx_bge}})
            ops.append({
                "chunk_id": f"bge_{d['doc_id']}", "doc_id": d["doc_id"],
                "kb_id": KB_ID_TEXT, "user_id": USER_ID,
                "content": d["content"], "caption": doc_caption[d["doc_id"]],
                "dense_vector": emb, "document_name": d.get("file_name", ""),
                "created_at": now,
            })
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Indexed {len(docs)} docs")

        # === Index 2: Qwen3-VL multimodal ===
        print("\n[3/5] Building Qwen3-VL multimodal index...")
        idx_mm = await create_idx(client, "eval_rerank_mm_", KB_ID_MM, MM_DIM)

        def download_img(fn):
            path = os.path.join(IMAGES_DIR, fn)
            if os.path.exists(path):
                with open(path, "rb") as f: return fn, base64.b64encode(f.read()).decode()
            resp = requests.get(f"{COCO_BASE_URL}/{fn}", timeout=30)
            os.makedirs(IMAGES_DIR, exist_ok=True)
            with open(path, "wb") as f: f.write(resp.content)
            return fn, base64.b64encode(resp.content).decode()

        fns = [d.get("file_name", "") for d in docs]
        with ThreadPoolExecutor(max_workers=16) as pool:
            img_data = dict(pool.map(download_img, fns))
        print(f"  Downloaded {len(img_data)} images")

        def embed_img(d):
            fn = d.get("file_name", "")
            if fn in img_data:
                try: return d, embed_image(img_data[fn]), fn
                except: return d, None, fn
            return d, None, None

        print("  Embedding images...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            mm_res = list(pool.map(embed_img, docs))
        mm_docs = [(d, e) for d, e, _ in mm_res if e is not None]
        print(f"  Embedded {len(mm_docs)} images in {time.time()-t0:.1f}s")

        ops = []
        for d, emb in mm_docs:
            ops.append({"index": {"_index": idx_mm}})
            ops.append({
                "chunk_id": f"mm_{d['doc_id']}", "doc_id": d["doc_id"],
                "kb_id": KB_ID_MM, "user_id": USER_ID,
                "content": d["content"], "caption": doc_caption[d["doc_id"]],
                "dense_vector": emb, "document_name": d.get("file_name", ""),
                "created_at": now,
            })
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Indexed {len(mm_docs)} docs")

        # === Retrieval ===
        print("\n[4/5] Running retrieval (4 methods)...")
        q_ids = list(queries.keys())[:NUM_DOCS]

        results_bge = {}
        results_q3vl = {}
        results_q3vl_rerank = {}  # Qwen3-VL recall + caption rerank
        results_bge_rerank = {}   # BGE recall + caption rerank

        t0 = time.time()
        for qi, q_id in enumerate(q_ids):
            q_text = queries[q_id]

            # BGE query embedding
            q_bge = embed_batch([q_text], TEXT_MODEL)[0]
            # Qwen3-VL query embedding
            q_mm = embed_batch([q_text], MM_MODEL)[0]

            # Stage 1: Recall
            bge_hyb = await hybrid_search_bge(client, idx_bge, q_bge, q_text, top_k=max(TOP_K))
            q3vl_knn = await knn_search(client, idx_mm, q_mm, top_k=RECALL_K)

            results_bge[q_id] = bge_hyb
            results_q3vl[q_id] = q3vl_knn[:max(TOP_K)]

            # Stage 2: Caption reranking
            # For Qwen3-VL candidates, embed captions and compute similarity
            q3dl_captions = [doc_caption.get(did, "") for did, _, _ in q3vl_knn]
            q3dl_cap_embs = np.array(embed_batch(q3dl_captions, TEXT_MODEL))
            q_bge_arr = np.array(q_bge)
            # Cosine similarity
            cap_norms = np.linalg.norm(q3dl_cap_embs, axis=1)
            q_norm = np.linalg.norm(q_bge_arr)
            sims = q3dl_cap_embs @ q_bge_arr / (cap_norms * q_norm + 1e-8)
            # Re-rank by caption similarity
            ranked_idx = np.argsort(sims)[::-1][:max(TOP_K)]
            results_q3vl_rerank[q_id] = [q3vl_knn[i] for i in ranked_idx]

            # For BGE candidates, also caption rerank
            bge_captions = [doc_caption.get(did, "") for did, _, _ in bge_hyb]
            bge_cap_embs = np.array(embed_batch(bge_captions, TEXT_MODEL))
            bge_sims = bge_cap_embs @ q_bge_arr / (np.linalg.norm(bge_cap_embs, axis=1) * q_norm + 1e-8)
            bge_ranked = np.argsort(bge_sims)[::-1][:max(TOP_K)]
            results_bge_rerank[q_id] = [bge_hyb[i] for i in bge_ranked]

            if (qi + 1) % 50 == 0:
                print(f"  Progress: {qi+1}/{len(q_ids)}")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Cleanup
        await client.indices.delete(index=idx_bge, ignore=[404])
        await client.indices.delete(index=idx_mm, ignore=[404])

    finally:
        await client.close()

    # === Metrics ===
    print("\n[5/5] Computing metrics...")
    m_bge = compute_metrics(results_bge, qrels, TOP_K)
    m_q3vl = compute_metrics(results_q3vl, qrels, TOP_K)
    m_q3vl_rr = compute_metrics(results_q3vl_rerank, qrels, TOP_K)
    m_bge_rr = compute_metrics(results_bge_rerank, qrels, TOP_K)

    print("\n" + "=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)

    all_results = {
        "BGE-M3 Hybrid": m_bge,
        "Qwen3-VL KNN": m_q3vl,
        "★ Qwen3-VL recall\n   + caption rerank": m_q3vl_rr,
        "BGE Hybrid recall\n   + caption rerank": m_bge_rr,
    }

    for name, m in all_results.items():
        print(f"\n{name}:")
        for k in TOP_K:
            print(f"  nDCG@{k}: {m[f'nDCG@{k}']:.4f}  |  Recall@{k}: {m[f'Recall@{k}']:.4f}")
        print(f"  MAP: {m['MAP']:.4f}")

    # Rerank improvement
    print("\n" + "=" * 60)
    print("📈 CAPTION RERANK IMPROVEMENT")
    print("=" * 60)
    print(f"\nQwen3-VL KNN → +caption rerank:")
    for k in METRIC_K:
        before = m_q3vl[f'nDCG@{k}']
        after = m_q3vl_rr[f'nDCG@{k}']
        print(f"  nDCG@{k}: {before:.4f} → {after:.4f} ({after-before:+.4f})")
    print(f"  MAP: {m_q3vl['MAP']:.4f} → {m_q3vl_rr['MAP']:.4f} ({m_q3vl_rr['MAP']-m_q3vl['MAP']:+.4f})")

    print(f"\nBGE Hybrid → +caption rerank:")
    for k in METRIC_K:
        before = m_bge[f'nDCG@{k}']
        after = m_bge_rr[f'nDCG@{k}']
        print(f"  nDCG@{k}: {before:.4f} → {after:.4f} ({after-before:+.4f})")
    print(f"  MAP: {m_bge['MAP']:.4f} → {m_bge_rr['MAP']:.4f} ({m_bge_rr['MAP']-m_bge['MAP']:+.4f})")

    # Save
    output = {
        "bge_hybrid": m_bge,
        "qwen3vl_knn": m_q3vl,
        "qwen3vl_rerank": m_q3vl_rr,
        "bge_rerank": m_bge_rr,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_rerank.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
