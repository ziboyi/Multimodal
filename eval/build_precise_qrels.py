#!/usr/bin/env python3
"""
用 caption 语义相似度构建精确 qrels

方法:
1. 对所有 doc captions 做 BGE-M3 embedding
2. 对每个 query caption，计算与所有 doc captions 的余弦相似度
3. 取相似度 top-N (N=10) 的作为相关文档
4. 同时保留 category overlap 作为辅助信号
"""

import json
import os
import numpy as np
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "coco", "mm_eval")
SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl"
)
EMBED_API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"
TOP_N = 10  # 每个 query 取 top-10 最相似作为相关


def embed_batch(texts):
    resp = requests.post(
        EMBED_API_URL,
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def main():
    print("Building precise qrels from caption semantic similarity...")
    
    with open(os.path.join(DATA_DIR, "documents.json")) as f:
        docs = json.load(f)
    with open(os.path.join(DATA_DIR, "queries.json")) as f:
        queries = json.load(f)

    # Build doc_id -> best caption (use first caption as representative)
    doc_captions = {}
    for d in docs:
        captions = d.get("captions", [])
        doc_captions[d["doc_id"]] = captions[0] if captions else d.get("content", "")[:200]

    # Embed all doc captions
    doc_ids = list(doc_captions.keys())
    doc_caps = [doc_captions[did] for did in doc_ids]
    print(f"Embedding {len(doc_caps)} doc captions...")
    doc_embs = np.array(embed_batch(doc_caps))
    # Normalize
    doc_embs_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)

    # Compute precise qrels
    qrels = {}
    rel_counts = []

    for q_id, q_text in queries.items():
        # Embed query
        q_emb = np.array(embed_batch([q_text]))
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        # Cosine similarity with all doc captions
        sims = doc_embs_norm @ q_norm.T.squeeze()

        # Top-N most similar
        top_indices = np.argsort(sims)[::-1][:TOP_N]
        qrels[q_id] = {doc_ids[i]: 1 for i in top_indices}
        rel_counts.append(len(qrels[q_id]))

    # Stats
    print(f"\nPrecise qrels built:")
    print(f"  Queries: {len(qrels)}")
    print(f"  Avg relevant per query: {np.mean(rel_counts):.1f}")
    print(f"  Min: {min(rel_counts)}, Max: {max(rel_counts)}")
    print(f"  Recall@5 theoretical max: {5/np.mean(rel_counts)*100:.1f}%")

    # Save
    out_path = os.path.join(DATA_DIR, "qrels_precise.json")
    with open(out_path, "w") as f:
        json.dump(qrels, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
