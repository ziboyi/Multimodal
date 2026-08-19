#!/usr/bin/env python3
"""VisualMRC 多模态文档检索评测"""

import asyncio, json, os, time, uuid, base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from elasticsearch import AsyncElasticsearch

ES_URL, ES_USER, ES_PASS = "http://localhost:9200", "elastic", "changeme"
SF_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl")
API = "https://api.siliconflow.cn/v1/embeddings"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "visualmrc_subset")
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
        hits, recalls, rr = [], [], []
        for q, ret in results.items():
            rel = qrels.get(q, set())
            hits.append(1 if set(ret[:k]) & rel else 0)
            recalls.append(len(set(ret[:k]) & rel) / len(rel) if rel else 0)
            for i, did in enumerate(ret):
                if did in rel:
                    rr.append(1.0/(i+1)); break
            else:
                rr.append(0)
        out[f"Hit@{k}"] = float(np.mean(hits))
        out[f"Recall@{k}"] = float(np.mean(recalls))
    out["MRR"] = float(np.mean(rr))
    return out

async def main():
    print("=" * 60)
    print("VisualMRC 多模态文档检索")
    print("=" * 60)

    with open(os.path.join(DATA_DIR, "samples.json")) as f:
        samples = json.load(f)
    print(f"Samples: {len(samples)}")

    # Build qrels
    url_groups = defaultdict(set)
    for s in samples:
        url_groups[s["url"]].add(s["image_file"])
    
    qrels = {}
    queries = {}
    for s in samples:
        q_key = f"{s['id']}_{s['qa_idx']}"
        qrels[q_key] = url_groups[s["url"]]
        queries[q_key] = s["question"]
    
    print(f"Queries: {len(queries)}, Avg rel/q: {np.mean([len(v) for v in qrels.values()]):.1f}")

    unique_imgs = {s["image_file"] for s in samples}
    print(f"Unique images: {len(unique_imgs)}")

    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        # Embed document images
        print("\nEmbedding document images...")
        idx = await mkidx(client, "eval_vmrc_", KB, 4096)
        
        def load_embed(img_f):
            path = os.path.join(DATA_DIR, img_f)
            if not os.path.exists(path):
                return img_f, None
            try:
                from PIL import Image
                import io
                img = Image.open(path).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode()
                e = emb_img(b64)
                return img_f, e
            except:
                return img_f, None
        
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            img_embs = dict(pool.map(load_embed, unique_imgs))
        img_embs = {k: v for k, v in img_embs.items() if v is not None}
        print(f"  Embedded {len(img_embs)} images in {time.time()-t0:.1f}s")

        ops = []
        for img_f, e in img_embs.items():
            ops.append({"index":{"_index":idx}})
            ops.append({"doc_id":img_f, "user_id":USER, "dense_vector":e})
        await client.bulk(operations=ops, refresh="wait_for")

        # Query in batches
        print("Querying...")
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        
        # Batch embed queries
        print("  Embedding queries...")
        q_embs = emb(q_texts, "Qwen/Qwen3-VL-Embedding-8B", bs=20)
        
        results = {}
        t0 = time.time()
        for qi, (q_id, q_vec) in enumerate(zip(q_ids, q_embs)):
            results[q_id] = await knn(client, idx, q_vec, max(K_VALS))
            if (qi+1) % 50 == 0:
                print(f"  {qi+1}/{len(q_ids)}")
        print(f"  Done in {time.time()-t0:.1f}s")

        m = compute_metrics(results, qrels, K_VALS)

        await client.indices.delete(index=idx, ignore=[404])
    finally:
        await client.close()

    print("\n" + "=" * 60)
    print("📊 VisualMRC Results (Qwen3-VL)")
    print("=" * 60)
    for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR", "Recall@1", "Recall@3", "Recall@5", "Recall@10"]:
        print(f"  {k}: {m[k]:.4f}")

    with open(os.path.join(os.path.dirname(__file__), "results_visualmrc.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"\n💾 Saved to results_visualmrc.json")

if __name__ == "__main__":
    asyncio.run(main())
