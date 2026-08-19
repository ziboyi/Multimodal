#!/usr/bin/env python3
"""
COCO 图文对齐数据集评测

数据集：eval/data/coco/mm_eval/
- 每张图片 = 一个文档（含 captions + categories）
- query = 一条 held-out caption
- qrels = 同类别的图片

对比方法：
1. BGE-M3 Hybrid (baseline): caption text → content text (KNN + BM25)
2. Qwen3-VL Text KNN: caption text → content text (KNN)
3. Qwen3-VL Image KNN: caption text → image (跨模态)
4. Qwen3-VL Image + Caption BM25 (综合优化)
5. BGE-M3 Image KNN: caption text → image (对比跨模态)
"""

import json, os, time, base64, asyncio
from collections import defaultdict

import numpy as np
import requests
from elasticsearch import AsyncElasticsearch

# Config
ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "changeme"
SF_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
API = "https://api.siliconflow.cn/v1/embeddings"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "mm_eval")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "images")
COCO_URL = "http://images.cocodataset.org/val2017"

USER = "eval_user"
K_VALS = [1, 3, 5, 10, 20]


def get_api_key():
    if SF_KEY:
        return SF_KEY
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    with open(env_path) as f:
        for line in f:
            if line.startswith("SILICONFLOW_API_KEY="):
                return line.strip().split("=", 1)[1]
    return ""


def emb_text(texts, model, bs=20):
    """批量文本 embedding"""
    key = get_api_key()
    all_out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        r = requests.post(API,
                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                         json={"model": model, "input": batch}, timeout=120)
        r.raise_for_status()
        all_out.extend([x["embedding"] for x in r.json()["data"]])
        time.sleep(0.2)
    return all_out


def emb_img(b64):
    """图片 embedding (Qwen3-VL)"""
    key = get_api_key()
    r = requests.post(API,
                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                     json={"model": "Qwen/Qwen3-VL-Embedding-8B", "input": " ",
                           "image": f"data:image/jpeg;base64,{b64}"}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def download_image(file_name):
    """下载 COCO 图片"""
    local_path = os.path.join(IMAGES_DIR, file_name)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    r = requests.get(f"{COCO_URL}/{file_name}", timeout=30)
    r.raise_for_status()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(r.content)
    return base64.b64encode(r.content).decode()


# ES operations
async def create_idx(client, name, dim):
    if await client.indices.exists(index=name):
        await client.indices.delete(index=name)
    await client.indices.create(index=name, body={"mappings":{"properties":{
        "doc_id": {"type": "keyword"},
        "user_id": {"type": "keyword"},
        "text": {"type": "text"},
        "caption": {"type": "text"},
        "vec": {"type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine"},
    }}})
    return name


async def knn_search(client, name, vec, k):
    body = {"query": {"term": {"user_id": USER}},
            "knn": {"field": "vec", "query_vector": vec, "k": k, "num_candidates": k*10},
            "size": k}
    r = await client.search(index=name, body=body)
    return [h["_source"]["doc_id"] for h in r["hits"]["hits"]]


async def hybrid_search(client, name, vec, text, k):
    """KNN + BM25 RRF fusion"""
    knn_body = {"query": {"term": {"user_id": USER}},
                "knn": {"field": "vec", "query_vector": vec, "k": k*2, "num_candidates": k*20},
                "size": k*2}
    bm25_body = {"query": {"bool": {"must": [{"multi_match": {"query": text, "fields": ["text", "caption"]}}],
                           "filter": [{"term": {"user_id": USER}}]}},
                "size": k*2}
    r1, r2 = await asyncio.gather(
        client.search(index=name, body=knn_body),
        client.search(index=name, body=bm25_body),
    )
    scores = {}
    kk = 60
    for rank, h in enumerate(r1["hits"]["hits"]):
        did = h["_source"]["doc_id"]
        scores[did] = scores.get(did, 0) + 1/(kk+rank+1)
    for rank, h in enumerate(r2["hits"]["hits"]):
        did = h["_source"]["doc_id"]
        scores[did] = scores.get(did, 0) + 1/(kk+rank+1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]


def compute_metrics(results, qrels, ks):
    """计算 Hit@k 和 MRR"""
    out = {}
    rr = []
    for q, ret in results.items():
        rel = set(qrels.get(q, {}).keys())
        for i, did in enumerate(ret):
            if did in rel:
                rr.append(1.0/(i+1))
                break
        else:
            rr.append(0)
    out["MRR"] = float(np.mean(rr))
    for k in ks:
        hits = []
        for q, ret in results.items():
            rel = set(qrels.get(q, {}).keys())
            hits.append(1 if set(ret[:k]) & rel else 0)
        out[f"Hit@{k}"] = float(np.mean(hits))
    return out


async def main():
    print("=" * 60)
    print("COCO 图文对齐数据集评测")
    print("=" * 60)

    # 1. Load dataset
    with open(os.path.join(DATA_DIR, "documents.json")) as f:
        documents = json.load(f)
    with open(os.path.join(DATA_DIR, "queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(DATA_DIR, "qrels.json")) as f:
        qrels = json.load(f)

    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels entries: {sum(len(v) for v in qrels.values())}")

    doc_ids = [d["doc_id"] for d in documents]

    # 2. Prepare texts
    # content text = categories + supercategories + captions
    doc_content = [d["content"] for d in documents]
    # caption = first caption
    doc_captions = [d["captions"][0] if d.get("captions") else "" for d in documents]

    # 3. Download images
    print("\n[1/6] Downloading images...")
    t0 = time.time()
    img_data = {}
    for i, d in enumerate(documents):
        fn = d.get("file_name", "")
        if fn:
            try:
                img_data[fn] = download_image(fn)
            except Exception as e:
                print(f"  Failed {fn}: {e}")
        if (i+1) % 40 == 0:
            print(f"  {i+1}/{len(documents)}")
    print(f"  Downloaded {len(img_data)} images in {time.time()-t0:.1f}s")

    # 4. Embedding
    print("\n[2/6] Embedding content texts (BGE-M3)...")
    t0 = time.time()
    bge_content_embs = emb_text(doc_content, "BAAI/bge-m3", bs=20)
    print(f"  BGE-M3 content: {len(bge_content_embs)} in {time.time()-t0:.1f}s")

    print("\n[3/6] Embedding content texts (Qwen3-VL)...")
    t0 = time.time()
    qwen_content_embs = emb_text(doc_content, "Qwen/Qwen3-VL-Embedding-8B", bs=20)
    print(f"  Qwen3-VL content: {len(qwen_content_embs)} in {time.time()-t0:.1f}s")

    print("\n[4/6] Embedding images (Qwen3-VL)...")
    t0 = time.time()
    qwen_img_embs = []
    for d in documents:
        fn = d.get("file_name", "")
        if fn and fn in img_data:
            try:
                e = emb_img(img_data[fn])
                qwen_img_embs.append(e)
            except Exception as ex:
                print(f"  Failed {fn}: {ex}")
                qwen_img_embs.append(None)
        else:
            qwen_img_embs.append(None)
        time.sleep(0.2)
    valid_img = sum(1 for e in qwen_img_embs if e is not None)
    print(f"  Qwen3-VL image: {valid_img} in {time.time()-t0:.1f}s")

    # 5. Build ES indices
    print("\n[5/6] Building ES indices...")
    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        # BGE-M3 content index
        idx_bge = await create_idx(client, "eval_coco_bge", 1024)
        ops = []
        for i, (did, emb) in enumerate(zip(doc_ids, bge_content_embs)):
            ops.append({"index": {"_index": idx_bge}})
            ops.append({"doc_id": did, "user_id": USER, "text": doc_content[i],
                        "caption": doc_captions[i], "vec": emb})
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  BGE-M3 content index: {len(doc_ids)} docs")

        # Qwen3-VL content index
        idx_qwen_text = await create_idx(client, "eval_coco_qwen_text", 4096)
        ops = []
        for i, (did, emb) in enumerate(zip(doc_ids, qwen_content_embs)):
            ops.append({"index": {"_index": idx_qwen_text}})
            ops.append({"doc_id": did, "user_id": USER, "text": doc_content[i],
                        "caption": doc_captions[i], "vec": emb})
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Qwen3-VL content index: {len(doc_ids)} docs")

        # Qwen3-VL image index
        idx_qwen_img = await create_idx(client, "eval_coco_qwen_img", 4096)
        ops = []
        for i, (did, emb) in enumerate(zip(doc_ids, qwen_img_embs)):
            if emb:
                ops.append({"index": {"_index": idx_qwen_img}})
                ops.append({"doc_id": did, "user_id": USER, "text": doc_content[i],
                            "caption": doc_captions[i], "vec": emb})
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Qwen3-VL image index: {len(ops)//2} docs")

        # BGE-M3 image index (对比)
        bge_img_embs = []
        for d in documents:
            fn = d.get("file_name", "")
            if fn and fn in img_data:
                try:
                    # BGE-M3 不支持图片，跳过
                    pass
                except:
                    pass
            bge_img_embs.append(None)
        # BGE-M3 不支持图片，跳过

        # 6. Query embedding
        print("\n[6/6] Query embedding + retrieval...")
        q_ids = list(queries.keys())
        q_texts = [queries[q] for q in q_ids]

        q_bge = emb_text(q_texts, "BAAI/bge-m3", bs=20)
        q_qwen = emb_text(q_texts, "Qwen/Qwen3-VL-Embedding-8B", bs=20)

        # 7. Retrieval
        results_bge_hybrid = {}
        results_qwen_text_knn = {}
        results_qwen_img_knn = {}
        results_qwen_img_hybrid = {}

        t0 = time.time()
        for qi, q in enumerate(q_ids):
            # BGE-M3 hybrid (baseline)
            results_bge_hybrid[q] = await hybrid_search(client, idx_bge, q_bge[qi], queries[q], max(K_VALS))

            # Qwen3-VL text KNN
            results_qwen_text_knn[q] = await knn_search(client, idx_qwen_text, q_qwen[qi], max(K_VALS))

            # Qwen3-VL image KNN (跨模态)
            results_qwen_img_knn[q] = await knn_search(client, idx_qwen_img, q_qwen[qi], max(K_VALS))

            # Qwen3-VL image + caption BM25 (综合优化)
            results_qwen_img_hybrid[q] = await hybrid_search(client, idx_qwen_img, q_qwen[qi], queries[q], max(K_VALS))

            if (qi+1) % 40 == 0:
                print(f"  {qi+1}/{len(q_ids)}")

        print(f"  Done in {time.time()-t0:.1f}s")

        # Cleanup
        await client.indices.delete(index=idx_bge, ignore=[404])
        await client.indices.delete(index=idx_qwen_text, ignore=[404])
        await client.indices.delete(index=idx_qwen_img, ignore=[404])
    finally:
        await client.close()

    # 8. Metrics
    print("\n" + "=" * 60)
    print("📊 COCO 图文对齐评测结果")
    print("=" * 60)

    m_bge = compute_metrics(results_bge_hybrid, qrels, K_VALS)
    m_qwen_text = compute_metrics(results_qwen_text_knn, qrels, K_VALS)
    m_qwen_img = compute_metrics(results_qwen_img_knn, qrels, K_VALS)
    m_qwen_img_hybrid = compute_metrics(results_qwen_img_hybrid, qrels, K_VALS)

    print(f"\n{'Method':<40s} | {'Hit@1':>7s} | {'Hit@3':>7s} | {'Hit@5':>7s} | {'Hit@10':>7s} | {'Hit@20':>7s} | {'MRR':>7s}")
    print("-" * 100)
    methods = [
        ("BGE-M3 Hybrid (baseline)", m_bge),
        ("Qwen3-VL Text KNN", m_qwen_text),
        ("Qwen3-VL Image KNN (cross-modal)", m_qwen_img),
        ("★ Qwen3-VL Image + Caption BM25", m_qwen_img_hybrid),
    ]
    for name, m in methods:
        print(f"{name:<40s} | {m['Hit@1']:>7.4f} | {m['Hit@3']:>7.4f} | {m['Hit@5']:>7.4f} | {m['Hit@10']:>7.4f} | {m['Hit@20']:>7.4f} | {m['MRR']:>7.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("📈 分析")
    print("=" * 60)

    print("\nQwen3-VL Image + BM25 vs BGE-M3 Hybrid:")
    for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR"]:
        base = m_bge[k]
        opt = m_qwen_img_hybrid[k]
        delta = opt - base
        sign = "✅" if delta > 0 else "❌"
        print(f"  {sign} {k}: {base:.4f} → {opt:.4f} ({delta:+.4f})")

    print("\nQwen3-VL Image vs Text:")
    for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR"]:
        base = m_qwen_text[k]
        opt = m_qwen_img[k]
        delta = opt - base
        sign = "✅" if delta > 0 else "❌"
        print(f"  {sign} {k}: Text={base:.4f} → Image={opt:.4f} ({delta:+.4f})")

    # Save
    output = {
        "bge_hybrid": m_bge,
        "qwen3vl_text_knn": m_qwen_text,
        "qwen3vl_img_knn": m_qwen_img,
        "qwen3vl_img_hybrid": m_qwen_img_hybrid,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_coco_alignment.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
