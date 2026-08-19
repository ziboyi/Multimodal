#!/usr/bin/env python3
"""
整合评测 v2：批量 embedding + 并行 ES 搜索

关键优化：
- 所有 doc embedding 一次性批量调用
- 所有 query embedding 一次性批量调用
- caption rerank embedding 预计算
- 每个 query 只执行 ES 搜索 + numpy 余弦相似度
"""

import asyncio
import base64
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from elasticsearch import AsyncElasticsearch

# ============================================================
# Config
# ============================================================
ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "changeme"
SF_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl")
API = "https://api.siliconflow.cn/v1/embeddings"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "mm_eval")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "images")
COCO_URL = "http://images.cocodataset.org/val2017"

NUM = 200
K_VALS = [5, 10, 20, 50]
USER = "eval_user"
KB_BGE = str(uuid.uuid4())
KB_MM = str(uuid.uuid4())

# ============================================================
# Embedding (batch)
# ============================================================
def emb_batch(texts, model, bs=50):
    """批量 embedding"""
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

# ============================================================
# ES
# ============================================================
def idx(prefix, kb): return f"{prefix}{kb.replace('-', '_')}"

async def mkidx(client, prefix, kb, dim):
    n = idx(prefix, kb)
    if await client.indices.exists(index=n):
        await client.indices.delete(index=n)
    await client.indices.create(index=n, body={"mappings":{"properties":{
        "doc_id":{"type":"keyword"},"user_id":{"type":"keyword"},
        "caption":{"type":"text"},
        "vec":{"type":"dense_vector","dims":dim,"index":True,"similarity":"cosine"},
    }}})
    return n

async def knn(client, n, v, k):
    b = {"query":{"term":{"user_id":USER}},"knn":{"field":"vec","query_vector":v,"k":k,"num_candidates":k*10},"size":k}
    r = await client.search(index=n, body=b)
    return [(h["_source"]["doc_id"], h.get("_source",{}).get("caption",""), h.get("_score",0))
            for h in r["hits"]["hits"]]

async def hybrid_bge(client, n, v, text, k):
    k1 = {"query":{"term":{"user_id":USER}},"knn":{"field":"vec","query_vector":v,"k":k*2,"num_candidates":k*20},"size":k*2}
    k2 = {"query":{"bool":{"must":[{"multi_match":{"query":text,"fields":["caption"]}}],
                     "filter":[{"term":{"user_id":USER}}]}},"size":k*2}
    r1, r2 = await asyncio.gather(client.search(index=n, body=k1), client.search(index=n, body=k2))
    sc, dm, kk = {}, {}, 60
    for rank, h in enumerate(r1["hits"]["hits"]):
        d = h["_source"]; did = d["doc_id"]
        sc[did] = sc.get(did, 0) + 1/(kk+rank+1); dm[did] = d.get("caption","")
    for rank, h in enumerate(r2["hits"]["hits"]):
        d = h["_source"]; did = d["doc_id"]
        sc[did] = sc.get(did, 0) + 1/(kk+rank+1)
        if did not in dm: dm[did] = d.get("caption","")
    s = sorted(sc.keys(), key=lambda x: sc[x], reverse=True)[:k]
    return [(did, dm[did], sc[did]) for did in s]

# ============================================================
# Metrics
# ============================================================
def metrics(results, qrels, ks):
    out = {}
    for k in ks:
        nd, rc, ap = [], [], []
        for q, ret in results.items():
            rels = [qrels.get(q,{}).get(r[0],0) for r in ret]
            nd.append(_ndcg(rels, k)); rc.append(_recall(rels, k)); ap.append(_ap(rels))
        out[f"nDCG@{k}"] = float(np.mean(nd))
        out[f"Recall@{k}"] = float(np.mean(rc))
        out["MAP"] = float(np.mean(ap))
    return out

def _ndcg(rel, k):
    d = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(rel[:k]))
    i = sum((2**r-1)/math.log2(j+2) for j,r in enumerate(sorted(rel,reverse=True)[:k]))
    return d/i if i>0 else 0
def _recall(rel, k):
    c = sum(1 for r in rel if r>0)
    return sum(1 for r in rel[:k] if r>0)/c if c>0 else 0
def _ap(rel):
    h,s=0,0.
    for i,r in enumerate(rel):
        if r>0: h+=1; s+=h/(i+1)
    return s/h if h>0 else 0

def rrf_fuse(lists, k=50, kk=60):
    sc = {}
    for lst in lists:
        for rank, (did, _, _) in enumerate(lst):
            sc[did] = sc.get(did, 0) + 1/(kk+rank+1)
    return sorted(sc.keys(), key=lambda x: sc[x], reverse=True)[:k]

# ============================================================
# Main
# ============================================================
async def main():
    print("=" * 60)
    print("整合评测 v2：批量 embedding")
    print("=" * 60)

    with open(os.path.join(DATA_DIR, "documents.json")) as f: docs = json.load(f)[:NUM]
    with open(os.path.join(DATA_DIR, "queries.json")) as f: queries = json.load(f)
    with open(os.path.join(DATA_DIR, "qrels_precise.json")) as f: qrels = json.load(f)

    doc_cap = {d["doc_id"]: d.get("captions",[""])[0] if d.get("captions") else d["content"][:200] for d in docs}
    doc_ids = list(doc_cap.keys())
    qids = list(queries.keys())[:NUM]
    print(f"Docs: {len(docs)}, Queries: {len(qids)}")

    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        # === Step 1: Batch embedding all docs ===
        print("\n[1/5] Embedding docs...")
        # BGE for text index
        t0 = time.time()
        bge_doc_embs = emb_batch([d["content"] for d in docs], "BAAI/bge-m3")
        print(f"  BGE doc embs: {len(bge_doc_embs)} in {time.time()-t0:.1f}s")

        # Qwen3-VL for multimodal index (images)
        t0 = time.time()
        def dl(fn):
            p = os.path.join(IMAGES_DIR, fn)
            if os.path.exists(p):
                with open(p,"rb") as f: return fn, base64.b64encode(f.read()).decode()
            r = requests.get(f"{COCO_URL}/{fn}", timeout=30)
            os.makedirs(IMAGES_DIR, exist_ok=True)
            with open(p,"wb") as f: f.write(r.content)
            return fn, base64.b64encode(r.content).decode()
        with ThreadPoolExecutor(max_workers=16) as pool:
            idata = dict(pool.map(dl, [d.get("file_name","") for d in docs]))
        print(f"  Downloaded {len(idata)} images in {time.time()-t0:.1f}s")

        t0 = time.time()
        def em(d):
            fn = d.get("file_name","")
            if fn in idata:
                try: return d, emb_img(idata[fn])
                except: return d, None
            return d, None
        with ThreadPoolExecutor(max_workers=16) as pool:
            mres = list(pool.map(em, docs))
        mmdocs = [(d,e) for d,e in mres if e is not None]
        print(f"  Qwen3-VL image embs: {len(mmdocs)} in {time.time()-t0:.1f}s")

        # Pre-compute caption embeddings for rerank
        t0 = time.time()
        cap_embs = np.array(emb_batch([doc_cap[did] for did in doc_ids], "BAAI/bge-m3"))
        cap_embs_norm = cap_embs / (np.linalg.norm(cap_embs, axis=1, keepdims=True) + 1e-8)
        print(f"  Caption embs: {len(cap_embs)} in {time.time()-t0:.1f}s")

        # === Step 2: Build ES indices ===
        print("\n[2/5] Building ES indices...")
        ib = await mkidx(client, "eval_all_bge_", KB_BGE, 1024)
        im = await mkidx(client, "eval_all_mm_", KB_MM, 4096)
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00")

        ops = []
        for d, e in zip(docs, bge_doc_embs):
            ops.append({"index":{"_index":ib}})
            ops.append({"doc_id":d["doc_id"],"user_id":USER,"caption":doc_cap[d["doc_id"]],"vec":e})
        await client.bulk(operations=ops, refresh="wait_for")

        ops = []
        for d, e in mmdocs:
            ops.append({"index":{"_index":im}})
            ops.append({"doc_id":d["doc_id"],"user_id":USER,"caption":doc_cap[d["doc_id"]],"vec":e})
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Indexed {len(docs)} (BGE) + {len(mmdocs)} (Qwen3-VL)")

        # === Step 3: Batch embedding all queries ===
        print("\n[3/5] Embedding all queries (batch)...")
        t0 = time.time()
        q_texts = [queries[q] for q in qids]
        qb_embs = emb_batch(q_texts, "BAAI/bge-m3")
        qm_embs = emb_batch(q_texts, "Qwen/Qwen3-VL-Embedding-8B")
        print(f"  Query embs: {len(qb_embs)} BGE + {len(qm_embs)} Qwen3-VL in {time.time()-t0:.1f}s")

        # === Step 4: Per-query retrieval (ES search + numpy rerank) ===
        print("\n[4/5] Per-query retrieval...")
        R = {m: {} for m in ["bge_hybrid","bge_knn","q3vl_knn","q3vl_rerank","fusion"]}
        t0 = time.time()

        for qi, q in enumerate(qids):
            qb = qb_embs[qi]
            qm = qm_embs[qi]

            # ES searches (parallel)
            bh, bk, qk = await asyncio.gather(
                hybrid_bge(client, ib, qb, queries[q], max(K_VALS)),
                knn(client, ib, qb, max(K_VALS)),
                knn(client, im, qm, 50),
            )

            R["bge_hybrid"][q] = bh
            R["bge_knn"][q] = bk
            R["q3vl_knn"][q] = qk[:max(K_VALS)]

            # Qwen3-VL + caption rerank (numpy, fast)
            qk_doc_ids = [did for did, _, _ in qk]
            qk_cap_indices = [doc_ids.index(did) if did in doc_ids else -1 for did in qk_doc_ids]
            valid_idx = [i for i, idx in enumerate(qk_cap_indices) if idx >= 0]
            if valid_idx:
                valid_cap_embs = cap_embs_norm[[qk_cap_indices[i] for i in valid_idx]]
                qb_arr = np.array(qb)
                qb_norm = qb_arr / (np.linalg.norm(qb_arr) + 1e-8)
                sims = valid_cap_embs @ qb_norm
                sorted_local = np.argsort(sims)[::-1][:max(K_VALS)]
                R["q3vl_rerank"][q] = [qk[valid_idx[i]] for i in sorted_local]
            else:
                R["q3vl_rerank"][q] = qk[:max(K_VALS)]

            # Fusion
            fused = rrf_fuse([bh, qk], k=max(K_VALS))
            R["fusion"][q] = [(did, doc_cap.get(did,""), 0) for did in fused]

            if (qi+1) % 50 == 0:
                print(f"  {qi+1}/{len(qids)}")

        print(f"  Done in {time.time()-t0:.1f}s")

        await client.indices.delete(index=ib, ignore=[404])
        await client.indices.delete(index=im, ignore=[404])
    finally:
        await client.close()

    # === Metrics ===
    print("\n[5/5] Computing metrics...")
    all_m = {m: metrics(R[m], qrels, K_VALS) for m in R}

    # === Output ===
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS (精确 qrels, avg 5 rel/query)")
    print("=" * 70)

    display = [
        ("BGE-M3 Hybrid", "bge_hybrid"),
        ("BGE-M3 KNN", "bge_knn"),
        ("Qwen3-VL KNN", "q3vl_knn"),
        ("★ Qwen3-VL + caption rerank", "q3vl_rerank"),
        ("★ RRF Fusion", "fusion"),
    ]

    header = f"{'Method':<32s} | {'nDCG@5':>7s} {'Recall@5':>9s} | {'nDCG@10':>8s} {'Recall@10':>10s} | {'MAP':>6s}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, key in display:
        m = all_m[key]
        print(f"{name:<32s} | {m['nDCG@5']:>7.4f} {m['Recall@5']:>9.4f} | {m['nDCG@10']:>8.4f} {m['Recall@10']:>10.4f} | {m['MAP']:>6.4f}")

    with open(os.path.join(os.path.dirname(__file__), "results_final.json"), "w") as f:
        json.dump(all_m, f, indent=2)
    print(f"\n💾 Saved to results_final.json")


if __name__ == "__main__":
    asyncio.run(main())
