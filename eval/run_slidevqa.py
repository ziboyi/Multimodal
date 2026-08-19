#!/usr/bin/env python3
"""SlideVQA 多模态文档检索 (官方 qrels, 使用预下载图片)"""

import asyncio, json, os, time, uuid, base64
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from datasets import load_dataset
from elasticsearch import AsyncElasticsearch

ES_URL, ES_USER, ES_PASS = "http://localhost:9200", "elastic", "changeme"
SF_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl")
API = "https://api.siliconflow.cn/v1/embeddings"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "slidevqa_subset")
USER = "eval_user"
KB = str(uuid.uuid4())
K_VALS = [1, 3, 5, 10]

def emb(texts, model, bs=20):
    all_out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        r = requests.post(API, headers={"Authorization": f"Bearer {SF_KEY}", "Content-Type": "application/json"},
                         json={"model": model, "input": batch}, timeout=120)
        r.raise_for_status()
        all_out.extend([x["embedding"] for x in r.json()["data"]])
    return all_out

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

def compute_metrics(results, qrels, ks):
    out = {}
    for k in ks:
        hits, recalls = [], []
        for q, ret in results.items():
            rel = qrels.get(q, set())
            hits.append(1 if set(ret[:k]) & rel else 0)
            recalls.append(len(set(ret[:k]) & rel) / len(rel) if rel else 0)
        out[f"Hit@{k}"] = float(np.mean(hits))
        out[f"Recall@{k}"] = float(np.mean(recalls))
    mrr = []
    for q, ret in results.items():
        rel = qrels.get(q, set())
        for i, did in enumerate(ret):
            if did in rel: mrr.append(1.0/(i+1)); break
        else: mrr.append(0)
    out["MRR"] = float(np.mean(mrr))
    return out

async def main():
    print("=" * 60)
    print("SlideVQA 多模态文档检索 (官方 qrels)")
    print("=" * 60)

    # Load queries and qrels
    print("Loading queries/qrels...")
    ds_q = load_dataset("openbmb/VisRAG-Ret-Test-SlideVQA", "queries", split="train", streaming=True)
    queries = {}
    for s in ds_q:
        queries[s["query-id"]] = s["query"]
    
    ds_r = load_dataset("openbmb/VisRAG-Ret-Test-SlideVQA", "qrels", split="train", streaming=True)
    qrels = {}
    for s in ds_r:
        qrels[s["query-id"]] = {s["corpus-id"]}
    
    valid_q = {q: t for q, t in queries.items() if q in qrels}
    print(f"Queries: {len(valid_q)}")

    # Use pre-downloaded images
    print("Loading pre-downloaded images...")
    corpus_imgs = {}
    for f in os.listdir(DATA_DIR):
        if f.endswith(".jpg"):
            # Extract corpus-id from filename
            corpus_imgs[f.replace(".jpg", "")] = os.path.join(DATA_DIR, f)
    
    print(f"Pre-downloaded images: {len(corpus_imgs)}")

    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        print("\nEmbedding corpus images...")
        idx = await mkidx(client, "eval_svqa_", KB, 4096)
        
        def load_embed(item):
            cid, path = item
            try:
                from PIL import Image
                import io
                img = Image.open(path).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode()
                e = emb_img(b64)
                return cid, e
            except:
                return cid, None
        
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            img_embs = dict(pool.map(load_embed, corpus_imgs.items()))
        img_embs = {k: v for k, v in img_embs.items() if v is not None}
        print(f"  Embedded {len(img_embs)} in {time.time()-t0:.1f}s")

        ops = []
        for cid, e in img_embs.items():
            ops.append({"index":{"_index":idx}})
            ops.append({"doc_id":cid, "user_id":USER, "dense_vector":e})
        for i in range(0, len(ops), 200):
            await client.bulk(operations=ops[i:i+200], refresh="wait_for")

        print("Querying...")
        q_ids = list(valid_q.keys())[:300]
        q_texts = [valid_q[q] for q in q_ids]
        
        print("  Embedding queries...")
        q_embs = emb(q_texts, "Qwen/Qwen3-VL-Embedding-8B", bs=20)
        
        results = {}
        t0 = time.time()
        for qi, (q_id, q_vec) in enumerate(zip(q_ids, q_embs)):
            results[q_id] = await knn(client, idx, q_vec, max(K_VALS))
            if (qi+1) % 100 == 0:
                print(f"  {qi+1}/{len(q_ids)}")
        print(f"  Done in {time.time()-t0:.1f}s")

        m = compute_metrics(results, qrels, K_VALS)

        await client.indices.delete(index=idx, ignore=[404])
    finally:
        await client.close()

    print("\n" + "=" * 60)
    print("📊 SlideVQA Results (Qwen3-VL)")
    print("=" * 60)
    for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR", "Recall@1", "Recall@3", "Recall@5", "Recall@10"]:
        print(f"  {k}: {m[k]:.4f}")

    with open(os.path.join(os.path.dirname(__file__), "results_slidevqa.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"\n💾 Saved to results_slidevqa.json")

if __name__ == "__main__":
    asyncio.run(main())
