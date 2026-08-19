#!/usr/bin/env python3
import json, os
BASE = "/home/zibo/桌面/Multimodal/eval/data"
OUT = os.path.join(BASE, "retrieval_dataset.json")
N = 50

doc_id_map = {}
coco_q, coco_r = {}, {}
docvqa_q, docvqa_r = {}, {}
slide_q, slide_r = {}, {}
vmrc_q, vmrc_r = {}, {}

# COCO
with open(os.path.join(BASE, "coco", "mm_eval", "documents.json")) as f:
    coco_docs = json.load(f)[:N]
for d in coco_docs:
    did = f"coco_{d['doc_id']}"
    captions = d.get("captions", [])
    doc_id_map[did] = {"dataset": "coco", "pdf": f"{did}.pdf", "docx": f"{did}.docx",
                       "image": d.get("file_name", ""), "text": "\n".join(captions)}
    qid = f"q_coco_{d['doc_id']}"
    coco_q[qid] = captions[0] if captions else ""
    coco_r[qid] = [did]

# DocVQA
with open(os.path.join(BASE, "docvqa_subset", "subset.json")) as f:
    samples = json.load(f)[:N]
for s in samples:
    did = f"docvqa_{s['questionId']}"
    doc_id_map[did] = {"dataset": "docvqa", "pdf": f"{did}.pdf", "docx": f"{did}.docx",
                       "image": s["image_file"], "text": s["question"]}
    qid = f"q_docvqa_{s['questionId']}"
    docvqa_q[qid] = s["question"]
    docvqa_r[qid] = [did]

# SlideVQA
slide_dir = os.path.join(BASE, "slidevqa_subset")
imgs = [f for f in os.listdir(slide_dir) if f.endswith(".jpg")][:N]
for i, img in enumerate(imgs):
    name = img.replace(".jpg", "")
    did = f"slidevqa_{name[:40]}"
    doc_id_map[did] = {"dataset": "slidevqa", "pdf": f"{did}.pdf", "docx": f"{did}.docx",
                       "image": img, "text": f"Slide: {name}"}
    qid = f"q_slidevqa_{i}"
    slide_q[qid] = f"slide about {name.split('-')[-2] if len(name.split('-')) > 1 else name}"
    slide_r[qid] = [did]

# VisualMRC
with open(os.path.join(BASE, "visualmrc_subset", "samples.json")) as f:
    samples = json.load(f)[:N]
for s in samples:
    did = f"visualmrc_{s['id'][:12]}_{s['qa_idx']}"
    doc_id_map[did] = {"dataset": "visualmrc", "pdf": f"{did}.pdf", "docx": f"{did}.docx",
                       "image": s["image_file"], "text": f"Q: {s['question']}\nA: {s['answer']}"}
    qid = f"q_vmrc_{s['id'][:12]}_{s['qa_idx']}"
    vmrc_q[qid] = s["question"]
    vmrc_r[qid] = [did]

dataset = {
    "documents": doc_id_map,
    "queries": {**coco_q, **docvqa_q, **slide_q, **vmrc_q},
    "qrels": {**coco_r, **docvqa_r, **slide_r, **vmrc_r},
}
with open(OUT, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Documents: {len(doc_id_map)}")
print(f"Queries: {len(dataset['queries'])}")
Qrels: {len(dataset['qrels'])}")
for ds in ["coco", "docvqa", "slidevqa", "visualmrc"]:
    cnt = sum(1 for d in doc_id_map.values() if d["dataset"] == ds)
    print(f"  {ds}: {cnt} docs")
