#!/usr/bin/env python3
"""
公平比较评测：eval_pdfs + retrieval_dataset.json

对比方法（全部使用文本-文本匹配）：
1. BGE-M3 Hybrid (baseline: KNN + BM25 on text)
2. Qwen3-VL KNN (纯文本 embedding KNN)
3. Qwen3-VL Hybrid (文本 KNN + BM25)

还保留跨模态比较：
4. Qwen3-VL 图片 KNN (图片 embedding)
5. Qwen3-VL 图片 + Caption BM25
"""

import json, os, time, base64, asyncio, io, re
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RETRIEVAL_DATASET = os.path.join(DATA_DIR, "retrieval_dataset.json")
EVAL_PDFS_DIR = os.path.join(DATA_DIR, "eval_pdfs")

USER = "eval_user"
K_VALS = [1, 3, 5, 10]

# Image directories for each dataset
IMG_DIRS = {
    "coco": os.path.join(DATA_DIR, "coco", "images"),
    "docvqa": os.path.join(DATA_DIR, "docvqa_subset"),
    "slidevqa": os.path.join(DATA_DIR, "slidevqa_subset"),
    "visualmrc": os.path.join(DATA_DIR, "visualmrc_subset"),
}


def get_api_key():
    """从 .env 读取 SiliconFlow key"""
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


def parse_pdf_images(pdf_path):
    """从 PDF 提取图片（每页渲染为一张图片）"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = pymupdf.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(base64.b64encode(img_data).decode())
        pix = None
    doc.close()
    return images


def parse_pdf_text(pdf_path):
    """从 PDF 提取文本（pymupdf 快速模式）"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(text_parts).strip()


def load_image_from_dataset(image_path, dataset):
    """从数据集目录加载原图"""
    if not image_path:
        return None
    paths_to_try = [
        os.path.join(IMG_DIRS.get(dataset, ""), image_path),
        os.path.join(DATA_DIR, image_path),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None


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
        rel = set(qrels.get(q, []))
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
            rel = set(qrels.get(q, []))
            hits.append(1 if set(ret[:k]) & rel else 0)
        out[f"Hit@{k}"] = float(np.mean(hits))
    return out


async def main():
    print("=" * 60)
    print("公平比较评测：eval_pdfs + retrieval_dataset.json")
    print("=" * 60)

    # 1. Load dataset
    with open(RETRIEVAL_DATASET) as f:
        dataset = json.load(f)

    docs = dataset["documents"]
    queries = dataset["queries"]
    qrels = {k: set(v) for k, v in dataset["qrels"].items()}

    print(f"Documents: {len(docs)}")
    print(f"Queries: {len(queries)}")

    # 2. Parse PDFs / extract content
    doc_ids = list(docs.keys())
    doc_texts = []
    doc_images = []  # base64

    print("\n[Preparing documents]")
    for i, did in enumerate(doc_ids):
        d = docs[did]
        pdf_path = os.path.join(EVAL_PDFS_DIR, f"{did}.pdf")

        # 尝试从 PDF 提取
        text = ""
        images = []
        if os.path.exists(pdf_path):
            text = parse_pdf_text(pdf_path)
            images = parse_pdf_images(pdf_path)

        # PDF 文本为空时，使用 JSON 中的 text 字段
        if not text:
            text = d.get("text", "")

        doc_texts.append(text)

        # 优先使用 PDF 渲染的第一页图片，否则使用数据集原图
        if images:
            doc_images.append(images[0])
        else:
            img_b64 = load_image_from_dataset(d.get("image", ""), d.get("dataset", ""))
            doc_images.append(img_b64)

        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(doc_ids)}")

    valid_imgs = sum(1 for i in doc_images if i)
    print(f"  Total docs: {len(doc_ids)}, with images: {valid_imgs}")

    # 3. Embedding
    print("\n[1/5] Embedding texts (BGE-M3)...")
    t0 = time.time()
    bge_text_embs = emb_text(doc_texts, "BAAI/bge-m3", bs=20)
    print(f"  BGE-M3: {len(bge_text_embs)} in {time.time()-t0:.1f}s")

    print("\n[2/5] Embedding texts (Qwen3-VL)...")
    t0 = time.time()
    qwen_text_embs = emb_text(doc_texts, "Qwen/Qwen3-VL-Embedding-8B", bs=20)
    print(f"  Qwen3-VL text: {len(qwen_text_embs)} in {time.time()-t0:.1f}s")

    print("\n[3/5] Embedding images (Qwen3-VL)...")
    t0 = time.time()
    qwen_img_embs = []
    for i, img_b64 in enumerate(doc_images):
        if img_b64:
            try:
                e = emb_img(img_b64)
                qwen_img_embs.append(e)
            except Exception as ex:
                print(f"  Failed {doc_ids[i]}: {ex}")
                qwen_img_embs.append(None)
        else:
            qwen_img_embs.append(None)
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(doc_images)}")
        time.sleep(0.2)
    valid_qwen_img = sum(1 for e in qwen_img_embs if e is not None)
    print(f"  Qwen3-VL image: {valid_qwen_img} in {time.time()-t0:.1f}s")

    # 4. Build ES indices
    print("\n[4/5] Building ES indices...")
    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        # BGE-M3 text index
        idx_bge = await create_idx(client, "eval_bge_m3", 1024)
        ops = []
        for did, emb in zip(doc_ids, bge_text_embs):
            idx = doc_ids.index(did)
            ops.append({"index": {"_index": idx_bge}})
            ops.append({"doc_id": did, "user_id": USER, "text": doc_texts[idx],
                        "caption": doc_texts[idx][:200], "vec": emb})
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  BGE-M3 index: {len(doc_ids)} docs")

        # Qwen3-VL text index
        idx_qwen_text = await create_idx(client, "eval_qwen3vl_text", 4096)
        ops = []
        for did, emb in zip(doc_ids, qwen_text_embs):
            idx = doc_ids.index(did)
            ops.append({"index": {"_index": idx_qwen_text}})
            ops.append({"doc_id": did, "user_id": USER, "text": doc_texts[idx],
                        "caption": doc_texts[idx][:200], "vec": emb})
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Qwen3-VL text index: {len(doc_ids)} docs")

        # Qwen3-VL image index
        idx_qwen_img = await create_idx(client, "eval_qwen3vl_img", 4096)
        ops = []
        for did, emb in zip(doc_ids, qwen_img_embs):
            if emb:
                idx = doc_ids.index(did)
                ops.append({"index": {"_index": idx_qwen_img}})
                ops.append({"doc_id": did, "user_id": USER, "text": doc_texts[idx],
                            "caption": doc_texts[idx][:200], "vec": emb})
        await client.bulk(operations=ops, refresh="wait_for")
        print(f"  Qwen3-VL image index: {len(ops)//2} docs")

        # 5. Query embedding
        print("\n[5/5] Query embedding + retrieval...")
        q_ids = list(queries.keys())
        q_texts = [queries[q] for q in q_ids]

        q_bge = emb_text(q_texts, "BAAI/bge-m3", bs=20)
        q_qwen = emb_text(q_texts, "Qwen/Qwen3-VL-Embedding-8B", bs=20)

        # 6. Retrieval
        results_bge_hybrid = {}
        results_qwen_knn = {}
        results_qwen_hybrid = {}
        results_qwen_img_knn = {}
        results_qwen_img_hybrid = {}

        t0 = time.time()
        for qi, q in enumerate(q_ids):
            # BGE-M3 hybrid (baseline)
            results_bge_hybrid[q] = await hybrid_search(client, idx_bge, q_bge[qi], queries[q], max(K_VALS))

            # Qwen3-VL text KNN (公平比较)
            results_qwen_knn[q] = await knn_search(client, idx_qwen_text, q_qwen[qi], max(K_VALS))

            # Qwen3-VL text + BM25 (公平比较)
            results_qwen_hybrid[q] = await hybrid_search(client, idx_qwen_text, q_qwen[qi], queries[q], max(K_VALS))

            # Qwen3-VL image KNN (跨模态)
            results_qwen_img_knn[q] = await knn_search(client, idx_qwen_img, q_qwen[qi], max(K_VALS))

            # Qwen3-VL image + BM25 (跨模态)
            results_qwen_img_hybrid[q] = await hybrid_search(client, idx_qwen_img, q_qwen[qi], queries[q], max(K_VALS))

            if (qi+1) % 20 == 0:
                print(f"  {qi+1}/{len(q_ids)}")

        print(f"  Done in {time.time()-t0:.1f}s")

        # Cleanup
        await client.indices.delete(index=idx_bge, ignore=[404])
        await client.indices.delete(index=idx_qwen_text, ignore=[404])
        await client.indices.delete(index=idx_qwen_img, ignore=[404])
    finally:
        await client.close()

    # 7. Metrics
    print("\n" + "=" * 60)
    print("📊 FAIR COMPARISON RESULTS")
    print("=" * 60)

    m_bge = compute_metrics(results_bge_hybrid, qrels, K_VALS)
    m_qwen_knn = compute_metrics(results_qwen_knn, qrels, K_VALS)
    m_qwen_hybrid = compute_metrics(results_qwen_hybrid, qrels, K_VALS)
    m_qwen_img_knn = compute_metrics(results_qwen_img_knn, qrels, K_VALS)
    m_qwen_img_hybrid = compute_metrics(results_qwen_img_hybrid, qrels, K_VALS)

    print(f"\n{'Method':<40s} | {'Hit@1':>7s} | {'Hit@3':>7s} | {'Hit@5':>7s} | {'Hit@10':>7s} | {'MRR':>7s}")
    print("-" * 90)
    methods = [
        ("BGE-M3 Hybrid (baseline)", m_bge),
        ("Qwen3-VL Text KNN", m_qwen_knn),
        ("★ Qwen3-VL Text + BM25", m_qwen_hybrid),
        ("---", None),
        ("Qwen3-VL Image KNN (cross-modal)", m_qwen_img_knn),
        ("Qwen3-VL Image + BM25 (cross-modal)", m_qwen_img_hybrid),
    ]
    for name, m in methods:
        if m is None:
            print("-" * 90)
            continue
        print(f"{name:<40s} | {m['Hit@1']:>7.4f} | {m['Hit@3']:>7.4f} | {m['Hit@5']:>7.4f} | {m['Hit@10']:>7.4f} | {m['MRR']:>7.4f}")

    # Improvement analysis
    print("\n" + "=" * 60)
    print("📈 FAIR COMPARISON: Qwen3-VL Text vs BGE-M3")
    print("=" * 60)
    for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        base = m_bge[k]
        opt = m_qwen_hybrid[k]
        delta = opt - base
        sign = "✅" if delta > 0 else "❌"
        print(f"  {sign} {k}: BGE={base:.4f} → Qwen3VL={opt:.4f} ({delta:+.4f})")

    print("\n" + "=" * 60)
    print("📈 CROSS-MODAL: Qwen3-VL Image vs Text")
    print("=" * 60)
    for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        base = m_qwen_knn[k]
        opt = m_qwen_img_knn[k]
        delta = opt - base
        sign = "✅" if delta > 0 else "❌"
        print(f"  {sign} {k}: Text={base:.4f} → Image={opt:.4f} ({delta:+.4f})")

    # Save results
    output = {
        "bge_hybrid": m_bge,
        "qwen3vl_text_knn": m_qwen_knn,
        "qwen3vl_text_hybrid": m_qwen_hybrid,
        "qwen3vl_img_knn": m_qwen_img_knn,
        "qwen3vl_img_hybrid": m_qwen_img_hybrid,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_fair_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
