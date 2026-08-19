#!/usr/bin/env python3
"""下载 SlideVQA 和 VisualMRC 子集"""

import json, os, io, base64, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from PIL import Image

OUT = "/home/zibo/桌面/Multimodal/eval/data"
N = 200

# === SlideVQA ===
print("=== SlideVQA ===")
ds = load_dataset("openbmb/VisRAG-Ret-Test-SlideVQA", "corpus", split="train", streaming=True)
slide_dir = os.path.join(OUT, "slidevqa_subset")
os.makedirs(slide_dir, exist_ok=True)

slide_samples = []
for i, s in enumerate(ds):
    if i >= N:
        break
    cid = s["corpus-id"]
    img_path = os.path.join(slide_dir, f"{cid}.jpg")
    if not os.path.exists(img_path):
        try:
            s["image"].convert("RGB").save(img_path)
        except:
            pass
    slide_samples.append({"corpus-id": cid, "image_file": f"{cid}.jpg"})
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N}")

with open(os.path.join(slide_dir, "corpus.json"), "w") as f:
    json.dump(slide_samples, f, indent=2)
print(f"  Saved {len(slide_samples)} slides")

# === VisualMRC ===
print("\n=== VisualMRC ===")
ds = load_dataset("jeepliu/VisualMRC", split="test", streaming=True)
vmrc_dir = os.path.join(OUT, "visualmrc_subset")
os.makedirs(vmrc_dir, exist_ok=True)

vmrc_samples = []
for i, s in enumerate(ds):
    if i >= N:
        break
    img_path = os.path.join(vmrc_dir, s["image_filename"])
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    if not os.path.exists(img_path):
        try:
            s["image"].convert("RGB").save(img_path)
        except:
            pass
    vmrc_samples.append({
        "id": s["id"],
        "qa_idx": s["qa_idx"],
        "url": s["url"],
        "question": s["question"],
        "answer": s["answer"],
        "image_file": s["image_filename"],
    })
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N}")

with open(os.path.join(vmrc_dir, "samples.json"), "w") as f:
    json.dump(vmrc_samples, f, indent=2)
print(f"  Saved {len(vmrc_samples)} samples")

print("\nDone!")
