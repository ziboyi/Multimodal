#!/usr/bin/env python3
"""
多模态检索系统 Baseline 评测脚本
Phase 1: BEIR NFCorpus (200 docs) — 纯文本检索 pipeline 验证
Phase 2: M-BEIR 多模态子集 (200 docs) — 图文混合检索

使用方式:
  python run_baseline.py --phase 1    # Phase 1: NFCorpus
  python run_baseline.py --phase 2    # Phase 2: M-BEIR
  python run_baseline.py --phase all  # 两者都跑

评测指标: nDCG@k, Recall@k, MAP
"""

import argparse
import asyncio
import json
import os
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import requests
from elasticsearch import AsyncElasticsearch

# ============================================================
# 配置
# ============================================================
ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "changeme"
ES_INDEX_PREFIX = "eval_baseline_"

SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-ujaqoxvtoetonfjdlruryjqbykkxcqxpluywibbiboohelrl"
)
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_API_URL = "https://api.siliconflow.cn/v1/embeddings"

NUM_DOCS = 200
TOP_K = [5, 10, 20, 50]
METRIC_K = [5, 10]

KB_ID = str(uuid.uuid4())
DOC_ID_PREFIX = "eval_doc"

USER_ID = "eval_user"

# ============================================================
# Embedding
# ============================================================
async def get_embeddings(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """调用 SiliconFlow API 获取 embeddings"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            EMBED_API_URL,
            headers={
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "input": batch,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(embeddings)
    return all_embeddings


def get_embeddings_sync(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """同步版 embedding"""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # 如果已经在 async 上下文中，用 thread 执行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, get_embeddings(texts, batch_size)).result()
    return asyncio.run(get_embeddings(texts, batch_size))


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
                    "fields": {
                        "cn": {"type": "text", "analyzer": "standard"},
                        "en": {"type": "text", "analyzer": "english"},
                    },
                },
                "language": {"type": "keyword"},
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": EMBED_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
                "image_url": {"type": "keyword"},
                "image_caption": {"type": "text"},
                "images": {
                    "type": "nested",
                    "properties": {
                        "url": {"type": "keyword"},
                        "caption": {"type": "text"},
                        "path": {"type": "keyword"},
                    },
                },
                "page_number": {"type": "integer"},
                "document_name": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "created_at": {"type": "date"},
                "is_multimodal": {"type": "boolean"},
            }
        }
    }
    await client.indices.create(index=index_name, body=mapping)
    return index_name


async def index_documents(
    client: AsyncElasticsearch,
    index_name: str,
    documents: List[Dict],
) -> int:
    """批量索引文档，获取 embedding 并写入 ES"""
    if not documents:
        return 0

    # 获取 texts 并 embedding
    texts = [doc["content"] for doc in documents]
    print(f"  → Embedding {len(texts)} documents...")
    t0 = time.time()
    embeddings = await get_embeddings(texts)
    print(f"  → Embedding done in {time.time() - t0:.1f}s")

    # 构建 bulk 操作
    operations = []
    for doc, emb in zip(documents, embeddings):
        operations.append({"index": {"_index": index_name}})
        operations.append({
            "chunk_id": doc["chunk_id"],
            "doc_id": doc["doc_id"],
            "kb_id": doc["kb_id"],
            "user_id": doc["user_id"],
            "chunk_index": doc.get("chunk_index", 0),
            "chunk_type": doc.get("chunk_type", "text"),
            "content": doc["content"],
            "language": doc.get("language", "en"),
            "dense_vector": emb,
            "image_url": doc.get("image_url", ""),
            "image_caption": doc.get("image_caption", ""),
            "images": doc.get("images", []),
            "page_number": doc.get("page_number"),
            "document_name": doc.get("document_name", ""),
            "metadata": doc.get("metadata", {}),
            "created_at": doc.get("created_at", ""),
            "is_multimodal": doc.get("is_multimodal", False),
        })

    response = await client.bulk(operations=operations, refresh="wait_for")
    if response.get("errors"):
        failed = sum(1 for item in response["items"] if "error" in item.get("index", {}))
        print(f"  ⚠ {failed} documents failed to index")
        return len(documents) - failed
    return len(documents)


async def hybrid_search(
    client: AsyncElasticsearch,
    index_name: str,
    query: str,
    top_k: int = 20,
) -> List[Dict]:
    """混合检索: semantic + BM25 + RRF"""
    # 获取 query embedding
    query_vec = (await get_embeddings([query]))[0]

    # Semantic search
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

    # BM25 search
    bm25_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "content.cn", "content.en"],
                        }
                    }
                ],
                "filter": [{"term": {"user_id": USER_ID}}],
            }
        },
        "highlight": {"fields": {"content": {}}},
        "size": top_k * 2,
    }

    knn_resp, bm25_resp = await asyncio.gather(
        client.search(index=index_name, body=knn_body),
        client.search(index=index_name, body=bm25_body),
    )

    # RRF fusion
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
            "content": src.get("content", "")[:200],
            "is_multimodal": src.get("is_multimodal", False),
        })
    return results


async def semantic_search_only(
    client: AsyncElasticsearch,
    index_name: str,
    query: str,
    top_k: int = 20,
) -> List[Dict]:
    """纯语义检索"""
    query_vec = (await get_embeddings([query]))[0]
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
            "content": src.get("content", "")[:200],
            "is_multimodal": src.get("is_multimodal", False),
        })
    return results


async def bm25_search_only(
    client: AsyncElasticsearch,
    index_name: str,
    query: str,
    top_k: int = 20,
) -> List[Dict]:
    """纯 BM25 检索"""
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "content.cn", "content.en"],
                        }
                    }
                ],
                "filter": [{"term": {"user_id": USER_ID}}],
            }
        },
        "highlight": {"fields": {"content": {}}},
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
            "content": src.get("content", "")[:200],
            "is_multimodal": src.get("is_multimodal", False),
        })
    return results


# ============================================================
# 评测指标
# ============================================================
def dcg_at_k(relevances: List[int], k: int) -> float:
    """计算 DCG@k"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """计算 nDCG@k"""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(relevances: List[int], k: int) -> float:
    """计算 Recall@k"""
    relevant_count = sum(1 for r in relevances if r > 0)
    if relevant_count == 0:
        return 0.0
    retrieved_relevant = sum(1 for r in relevances[:k] if r > 0)
    return retrieved_relevant / relevant_count


def average_precision(relevances: List[int]) -> float:
    """计算 Average Precision"""
    hits = 0
    sum_prec = 0.0
    for i, rel in enumerate(relevances):
        if rel > 0:
            hits += 1
            sum_prec += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_prec / hits


def compute_metrics(
    results: List[Dict],
    qrels: Dict[str, int],
    top_k_values: List[int] = None,
) -> Dict[str, float]:
    """计算一组查询的评测指标"""
    if top_k_values is None:
        top_k_values = METRIC_K

    # 每个查询的结果
    query_metrics = defaultdict(lambda: defaultdict(float))

    all_ndcg = defaultdict(list)
    all_recall = defaultdict(list)
    all_ap = []

    for query_id, retrieved in results.items():
        query_qrels = qrels.get(query_id, {})
        relevances = []
        for r in retrieved:
            doc_id = r["doc_id"]
            rel = query_qrels.get(doc_id, 0)
            relevances.append(rel)

        for k in top_k_values:
            all_ndcg[k].append(ndcg_at_k(relevances, k))
            all_recall[k].append(recall_at_k(relevances, k))
        all_ap.append(average_precision(relevances))

    metrics = {}
    for k in top_k_values:
        metrics[f"nDCG@{k}"] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
        metrics[f"Recall@{k}"] = np.mean(all_recall[k]) if all_recall[k] else 0.0
    metrics["MAP"] = np.mean(all_ap) if all_ap else 0.0

    return metrics


# ============================================================
# Phase 1: BEIR NFCorpus
# ============================================================
async def run_phase1():
    """Phase 1: BEIR NFCorpus 200 docs — 纯文本检索 pipeline 验证"""
    print("=" * 60)
    print("Phase 1: BEIR NFCorpus (200 docs)")
    print("=" * 60)

    from datasets import load_dataset

    print("\n[1/4] Loading NFCorpus dataset from local files...")
    data_dir = "/tmp/beir_data/nfcorpus"

    # Load corpus
    corpus_list = []
    with open(os.path.join(data_dir, "corpus.jsonl")) as f:
        for line in f:
            corpus_list.append(json.loads(line))

    # 限制 200 docs
    selected_ids = [item["_id"] for item in corpus_list[:NUM_DOCS]]
    selected_set = set(selected_ids)

    corpus_dict = {}
    for item in corpus_list:
        if item["_id"] in selected_set:
            corpus_dict[item["_id"]] = {
                "text": item["text"],
                "title": item.get("title", ""),
            }

    # Load queries
    queries_list = []
    with open(os.path.join(data_dir, "queries.jsonl")) as f:
        for line in f:
            queries_list.append(json.loads(line))

    # Load qrels (test split)
    qrels = defaultdict(dict)
    with open(os.path.join(data_dir, "qrels", "test.tsv")) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                q_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if doc_id in selected_set:
                    qrels[q_id][doc_id] = score

    # 只保留有相关文档的查询
    valid_queries = {item["_id"]: item["text"] for item in queries_list if item["_id"] in qrels}

    print(f"  Corpus: {len(corpus_dict)} docs")
    print(f"  Queries: {len(valid_queries)} (with relevant docs)")
    print(f"  Qrels entries: {sum(len(v) for v in qrels.values())}")

    # 准备文档
    documents = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    for i, (doc_id, doc_data) in enumerate(corpus_dict.items()):
        content = doc_data["title"] + ". " + doc_data["text"] if doc_data["title"] else doc_data["text"]
        documents.append({
            "chunk_id": f"doc_{doc_id}",
            "doc_id": doc_id,
            "kb_id": KB_ID,
            "user_id": USER_ID,
            "chunk_index": i,
            "chunk_type": "text",
            "content": content,
            "language": "en",
            "document_name": doc_id,
            "created_at": now,
            "is_multimodal": False,
        })

    # ES 操作
    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        print("\n[2/4] Creating index...")
        index_name = await create_es_index(client, KB_ID)
        print(f"  Index: {index_name}")

        print("\n[3/4] Indexing documents...")
        t0 = time.time()
        indexed = await index_documents(client, index_name, documents)
        print(f"  Indexed {indexed} docs in {time.time() - t0:.1f}s")

        # 评测三种检索模式
        search_modes = {
            "hybrid": hybrid_search,
            "semantic": semantic_search_only,
            "bm25": bm25_search_only,
        }

        all_metrics = {}
        for mode_name, search_fn in search_modes.items():
            print(f"\n[4/4] Running {mode_name} search...")
            results = {}
            q_ids = list(valid_queries.keys())
            t0 = time.time()
            for qi, q_id in enumerate(q_ids):
                query_text = valid_queries[q_id]
                retrieved = await search_fn(client, index_name, query_text, top_k=max(TOP_K))
                results[q_id] = retrieved
                if (qi + 1) % 10 == 0:
                    print(f"  Progress: {qi + 1}/{len(q_ids)} queries")

            elapsed = time.time() - t0
            metrics = compute_metrics(results, dict(qrels), TOP_K)
            metrics["avg_query_time_ms"] = (elapsed / len(q_ids)) * 1000
            all_metrics[mode_name] = metrics

            print(f"\n  {mode_name.upper()} Results:")
            print(f"    Queries: {len(q_ids)} in {elapsed:.1f}s ({metrics['avg_query_time_ms']:.0f}ms/query)")
            for k in TOP_K:
                ndcg = metrics.get(f"nDCG@{k}", 0)
                recall = metrics.get(f"Recall@{k}", 0)
                print(f"    nDCG@{k}: {ndcg:.4f}  |  Recall@{k}: {recall:.4f}")
            print(f"    MAP: {metrics['MAP']:.4f}")

        # 清理
        await client.indices.delete(index=index_name, ignore=[404])

    finally:
        await client.close()

    return all_metrics


# ============================================================
# Phase 2: M-BEIR 多模态 (简化模拟)
# ============================================================
async def run_phase2():
    """Phase 2: COCO 合成多模态检索评测 (200 docs)"""
    print("=" * 60)
    print("Phase 2: 多模态检索评测 COCO (200 docs)")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "data", "coco", "mm_eval")

    print(f"\n[1/4] Loading multimodal dataset from {data_dir}...")
    with open(os.path.join(data_dir, "documents.json")) as f:
        mm_documents = json.load(f)
    with open(os.path.join(data_dir, "queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(data_dir, "qrels.json")) as f:
        qrels = json.load(f)

    print(f"  Documents: {len(mm_documents)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels entries: {sum(len(v) for v in qrels.values())}")

    # Prepare documents for indexing
    documents = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    doc_kb_id = str(uuid.uuid4())

    for i, doc in enumerate(mm_documents):
        documents.append({
            "chunk_id": f"mm_{doc['doc_id']}",
            "doc_id": doc["doc_id"],
            "kb_id": doc_kb_id,
            "user_id": USER_ID,
            "chunk_index": i,
            "chunk_type": "multimodal",
            "content": doc["content"],
            "language": "en",
            "document_name": doc.get("file_name", doc["doc_id"]),
            "image_url": doc.get("image_url", ""),
            "image_caption": doc.get("image_caption", ""),
            "images": [{"url": doc.get("image_url", ""), "caption": doc.get("image_caption", "")}],
            "created_at": now,
            "is_multimodal": True,
            "metadata": {
                "categories": doc.get("categories", []),
                "supercategories": doc.get("supercategories", []),
                "num_captions": len(doc.get("captions", [])),
            },
        })

    # ES operations
    client = AsyncElasticsearch(hosts=[ES_URL], basic_auth=(ES_USER, ES_PASS))
    try:
        print("\n[2/4] Creating index...")
        index_name = await create_es_index(client, doc_kb_id)
        print(f"  Index: {index_name}")

        print("\n[3/4] Indexing multimodal documents...")
        t0 = time.time()
        indexed = await index_documents(client, index_name, documents)
        print(f"  Indexed {indexed} docs in {time.time() - t0:.1f}s")

        # Run evaluation
        print("\n[4/4] Running multimodal retrieval evaluation...")
        search_modes = {
            "hybrid": hybrid_search,
            "semantic": semantic_search_only,
        }

        all_metrics = {}
        for mode_name, search_fn in search_modes.items():
            results = {}
            q_ids = list(queries.keys())
            t0 = time.time()
            for qi, q_id in enumerate(q_ids):
                query_text = queries[q_id]
                retrieved = await search_fn(client, index_name, query_text, top_k=max(TOP_K))
                results[q_id] = retrieved
                if (qi + 1) % 50 == 0:
                    print(f"  Progress: {qi + 1}/{len(q_ids)} queries")

            elapsed = time.time() - t0
            metrics = compute_metrics(results, qrels, TOP_K)
            metrics["avg_query_time_ms"] = (elapsed / len(q_ids)) * 1000
            all_metrics[mode_name] = metrics

            print(f"\n  {mode_name.upper()} Results:")
            print(f"    Queries: {len(q_ids)} in {elapsed:.1f}s ({metrics['avg_query_time_ms']:.0f}ms/query)")
            for k in TOP_K:
                ndcg = metrics.get(f"nDCG@{k}", 0)
                recall = metrics.get(f"Recall@{k}", 0)
                print(f"    nDCG@{k}: {ndcg:.4f}  |  Recall@{k}: {recall:.4f}")
            print(f"    MAP: {metrics['MAP']:.4f}")

        # Cleanup
        await client.indices.delete(index=index_name, ignore=[404])

    finally:
        await client.close()

    return all_metrics


# ============================================================
# Main
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="多模态检索系统 Baseline 评测")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="1")
    parser.add_argument("--num-docs", type=int, default=200)
    parser.add_argument("--kb-id", type=str, default=None)
    args = parser.parse_args()

# global vars set via args
    NUM_DOCS = args.num_docs
    KB_ID = args.kb_id

    print(f"\n📊 Multimodal Retrieval Baseline Evaluation")
    print(f"   Embed Model: {EMBED_MODEL} (dim={EMBED_DIM})")
    print(f"   ES: {ES_URL}")
    print(f"   Num Docs: {NUM_DOCS}")
    print()

    results = {}

    if args.phase in ("1", "all"):
        results["phase1_nfcorpus"] = await run_phase1()

    if args.phase in ("2", "all"):
        results["phase2_multimodal"] = await run_phase2()

    # 输出总结
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    for phase_name, phase_results in results.items():
        print(f"\n{phase_name}:")
        for mode, metrics in phase_results.items():
            print(f"  {mode}:")
            for metric_name, value in sorted(metrics.items()):
                if isinstance(value, float):
                    print(f"    {metric_name}: {value:.4f}")
                else:
                    print(f"    {metric_name}: {value}")

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
