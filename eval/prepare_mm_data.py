#!/usr/bin/env python3
"""
从 COCO 数据集构建合成多模态评测数据
- 每张图片 = 一个多模态文档（captions + categories + supercategories）
- query = 一条 held-out caption
- qrels = 同类别的图片为相关
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "coco")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
OUTPUT_DIR = os.path.join(DATA_DIR, "mm_eval")
NUM_DOCS = 200


def build_mm_dataset():
    print("Loading COCO annotations...")

    # Load captions
    with open(os.path.join(ANNOTATIONS_DIR, "captions_val2017.json")) as f:
        captions_data = json.load(f)

    # Load instances (for categories)
    with open(os.path.join(ANNOTATIONS_DIR, "instances_val2017.json")) as f:
        instances_data = json.load(f)

    # Build category map
    cat_map = {c["id"]: c for c in instances_data["categories"]}

    # Build image -> categories mapping from instances
    image_categories = {}
    for ann in instances_data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in image_categories:
            image_categories[img_id] = set()
        image_categories[img_id].add(cat_id)

    # Build image -> captions mapping
    image_captions = {}
    for ann in captions_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(ann["caption"])

    # Build image info map
    image_info = {img["id"]: img for img in captions_data["images"]}

    # Select images that have both captions and categories
    valid_images = [
        img_id
        for img_id in image_info
        if img_id in image_captions and img_id in image_categories
    ][:NUM_DOCS]

    print(f"Selected {len(valid_images)} images")

    # Build multimodal documents
    documents = []
    for img_id in valid_images:
        info = image_info[img_id]
        captions = image_captions[img_id]
        cat_ids = image_categories.get(img_id, set())

        # Build text content: captions + category names
        cat_names = [cat_map[cid]["name"] for cid in cat_ids if cid in cat_map]
        supercat_names = list(
            set(
                cat_map[cid]["supercategory"]
                for cid in cat_ids
                if cid in cat_map
            )
        )

        caption_text = " ".join(captions)
        category_text = (
            f"Categories: {', '.join(cat_names)}. "
            f"Supercategories: {', '.join(supercat_names)}."
        )

        content = f"{category_text}\n\nScene description: {caption_text}"

        documents.append({
            "doc_id": str(img_id),
            "image_url": info.get("coco_url", ""),
            "image_caption": captions[0] if captions else "",
            "content": content,
            "captions": captions,
            "categories": cat_names,
            "supercategories": supercat_names,
            "file_name": info.get("file_name", ""),
        })

    # Build queries and qrels
    # Strategy: use first caption as query, same-category images as relevant
    queries = {}
    qrels = {}

    # Build category -> images index
    category_images = {}
    for doc in documents:
        for cat in doc["categories"]:
            if cat not in category_images:
                category_images[cat] = []
            category_images[cat].append(doc["doc_id"])

    for doc in documents:
        q_id = f"q_{doc['doc_id']}"
        # Query = first caption (held-out simulation)
        queries[q_id] = doc["captions"][0] if doc["captions"] else ""

        # Relevant docs = same-category images (including self)
        relevant = set()
        for cat in doc["categories"]:
            relevant.update(category_images.get(cat, []))
        qrels[q_id] = {doc_id: 1 for doc_id in relevant}

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "documents.json"), "w") as f:
        json.dump(documents, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "queries.json"), "w") as f:
        json.dump(queries, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "qrels.json"), "w") as f:
        json.dump(qrels, f, indent=2)

    # Stats
    print(f"\nDataset saved to {OUTPUT_DIR}")
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels entries: {sum(len(v) for v in qrels.values())}")
    print(f"  Avg relevant per query: {sum(len(v) for v in qrels.values()) / len(qrels):.1f}")
    print(f"\nSample document:")
    print(f"  ID: {documents[0]['doc_id']}")
    print(f"  Categories: {documents[0]['categories']}")
    print(f"  Content preview: {documents[0]['content'][:150]}...")
    print(f"\nSample query:")
    q0 = list(queries.keys())[0]
    print(f"  ID: {q0}")
    print(f"  Text: {queries[q0]}")
    print(f"  Relevant docs: {len(qrels[q0])}")


if __name__ == "__main__":
    build_mm_dataset()
