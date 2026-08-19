#!/usr/bin/env python3
"""
DocVQA 多模态文档检索评测

Query: 问题文本
Document: 文档图片
Relevance: docId 匹配（同 ucsf_document_id 的页面也视为相关）

Methods:
  1. BGE-M3 text (OCR text from document, if available, or just question embedding)
  2. Qwen3-VL (image embedding of document + text query)
  3. Qwen3-VL + caption/content rerank
"""

import asyncio
import json
import math
import os
import time
import uuid
from collections import defaultdict

import numpy as np
import requests
from elasticsearch import AsyncElasticsearch

ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "changeme"
SF_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl")
API = "https://api.siliconflow.cn/v1/embeddings"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "docvqa_subset")
USER = "eval_user"
KB = str(uuid.uuid4())
K_VALS = [1, 3, 5, 10]

def emb(texts, model):
    r = requests.post(API, headers={"Authorization": f"Bearer {SF_KEY}", "Content-Type": "application/json"},
                     json={"model": model, "input": texts}, timeout=120)
    r.raise_for_status()
    return [x["embedding"] for x in r.json()["data"]]

def emb_img(b64):
    r = requests.post(API, headers={"Authorization": f"Bearer {SF_KEY}", "Content-Type": "application/json"},
                     json={"model": "Qwen/Qwen3-VL-Embedding-8B", "input": " ",
                           "image": f"data:image/jpeg;base64,{b64}"}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

async def mkidx(client, prefix, kb, dim):
    n = f"{prefix}{kb.replace('-','_')}"
    if await client.indices.exists(index=n):
        await client.indices.delete(index=n)
    await client.indices.create(index=n, body={"mappings":{"properties":{
        "doc_id":{"type":"keyword"},"user_id":{"type":"keyword"},
        "dense_vector":{"type":"dense_vector","dims":dim,"index":True,"similarity":"cosine"},
    }}})
    return n

async def knn(client, n, v, k):
    b = {"query":{"term":{"user_id":USER}},"knn":{"field":"dense_vector","query_vector":v,"k":k,"num_candidates":k*10},"size":k}
    r = await client.search(index=n, body=b)
    return [h["_source"]["doc_id"] for h in r["hits"]["hits"]]

def metrics(results, qrels, ks):
    out = {}
    for k in ks:
        hits, rr = [], []
        for q, ret in results.items():
            rel_set = qrels.get(q, set())
            # Recall@k
            recall = len(set(ret[:k]) & rel_set) / len(rel_set) if rel_set else 0
            rr.append(recall)
            # Hit@k (binary: any relevant in top-k)
            hit = 1 if set(ret[:k]) & rel_set else 0
            hits.append(hit)
        out[f"Recall@{k}"] = float(np.mean(rr))
        out[f"Hit@{k}"] = float(np.mean(hits))
    # MRR
    mrr = []
    for q, ret in results.items():
        rel_set = qrels.get(q, set())
        for i, did in enumerate(ret):
            if did in rel_set:
                mrr.append(1.0/(i+1))
                break
        else:
            mrr.append(0)
    out["MRR"] = float(np.mean(mrr))
    return out

async def main():
    print("=" * 60)
    print("DocVQA 多模态文档检索评测")
    print("=" * 60)

    with open(os.path.join(DATA_DIR, "subset.json")) as f:
        samples = json.load(f)
    print(f"Samples: {len(samples)}")

    # Build qrels: query -> relevant doc_ids
    # Questions from same ucsf_document_id share relevance
    doc_groups = defaultdict(set)
    for s in samples:
        doc_groups[s["ucsf_document_id"]].add(s["docId"])
    
    qrels = {}
    for s in samples:
        qrels[s["questionId"]] = doc_groups[s["ucsf_document_id"]]
    
    queries = {s["questionId"]: s["question"] for s in samples}
    print(f"Queries: {len(queries)}, Avg rel/q: {np.mean([len(v) for v in qrels.values()]):.1f}")

    # Prepare document images
    doc_images = {}
    for s in samples:
        doc_images[s["docId"]] = os.path.join(DATA_DIR, s["image_file"])
    
    unique_docs = list(doc_images.keys())
    print(f"Unique docs: {len(unique_docs)}")

    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        # === Method 1: BGE-M3 (question text -> document, text-only baseline) ===
        print("\n[1/3] BGE-M3 text retrieval...")
        idx_bge = await mkidx(client, "eval_docvqa_bge_", KB, 1024)
        q_embs = emb(list(queries.values()), "BAAI/bge-m3")
        
        # For BGE, we don't have document text (only images), so we use question embedding
        # and search against... we need document text. DocVQA doesn't provide OCR text.
        # So BGE can only do question-to-question similarity (not meaningful for retrieval)
        # Skip BGE for DocVQA since we don't have document text
        print("  Skipping BGE (no document text available)")
        await client.indices.delete(index=idx_bge, ignore=[404])

        # === Method 2: Qwen3-VL (image document + text query) ===
        print("\n[2/3] Qwen3-VL multimodal retrieval...")
        idx_mm = await mkidx(client, "eval_docvqa_mm_", KB, 4096)
        
        # Embed document images
        print("  Embedding document images...")
        import base64
        from concurrent.futures import ThreadPoolExecutor
        
        def load_and_embed(doc_id):
            path = doc_images[doc_id]
            if not os.path.exists(path):
                return doc_id, None
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                e = emb_img(b64.replace("data:image/jpeg;base64,","") if "base64" in b64 else b64)
                return doc_id, e
            except Exception as ex:
                return doc_id, None
        
        # Fix: emb_img already adds data: prefix
        def load_and_embed2(doc_id):
            path = doc_images[doc_id]
            if not os.path.exists(path):
                return doc_id, None
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                # Convert PNG to JPEG for smaller size
                from PIL import Image
                import io
                img = Image.open(path).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode()
                e = emb_img(b64)
                return doc_id, e
            except Exception as ex:
                return doc_id, None
        
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            doc_embs = dict(pool.map(load_and_embed2, unique_docs))
        doc_embs = {k: v for k, v in doc_embs.items() if v is not None}
        print(f"  Embedded {len(doc_embs)} docs in {time.time()-t0:.1f}s")

        # Index
        ops = []
        for did, e in doc_embs.items():
            ops.append({"index":{"_index":idx_mm}})
            ops.append({"doc_id":did, "user_id":USER, "dense_vector":e})
        await client.bulk(operations=ops, refresh="wait_for")

        # Embed queries and search
        print("  Embedding queries and searching...")
        mm_results = {}
        t0 = time.time()
        for qi, (q_id, q_text) in enumerate(queries.items()):
            q_vec = emb([q_text], "Qwen/Qwen3-VL-Embedding-8B")[0]
            mm_results[q_id] = await knn(client, idx_mm, q_vec, max(K_VALS))
            if (qi+1) % 50 == 0:
                print(f"  {qi+1}/{len(queries)}")
        print(f"  Done in {time.time()-t0:.1f}s")

        m_mm = metrics(mm_results, qrels, K_VALS)

        # === Method 3: Qwen3-VL + rerank (but we don't have captions) ===
        # Skip - no captions available for DocVQA
        await client.indices.delete(index=idx_mm, ignore=[404])

    finally:
        await client.close()

    # === Output ===
    print("\n" + "=" * 60)
    print("📊 DocVQA 多模态检索结果")
    print("=" * 60)
    print(f"\nQwen3-VL (text query → document image):")
    for k in sorted(m_mm.keys()):
        print(f"  {k}: {m_mm[k]:.4f}")

    with open(os.path.join(os.path.dirname(__file__), "results_docvqa.json"), "w") as f:
        json.dump(m_mm, f, indent=2)
    print(f"\n💾 Saved to results_docvqa.json")

if __name__ == "__main__":
    asyncio.run(main())
