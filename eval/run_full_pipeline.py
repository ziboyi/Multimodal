#!/usr/bin/env python3
"""
全链路评测：生成 PDF → 上传知识库 → 搜索 → 评估

测试数据集的完整多模态检索链路：
1. 从图片生成 PDF 文档
2. 上传到知识库（后端自动解析）
3. 构建查询和 qrels
4. 调用搜索 API 评测
"""

import json, os, time, requests, io, base64
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

API_BASE = "http://localhost:8000/api"
DATA_DIR = os.path.dirname(__file__)

# ===== Step 1: Auth =====
def register_and_login():
    """注册并登录获取 token"""
    email = "eval_test@example.com"
    password = "EvalTest123!"
    
    # Try register
    r = requests.post(f"{API_BASE}/auth/register", json={
        "email": email, "password": password, "full_name": "Eval Test"
    })
    if r.status_code not in (201, 400):  # 400 = already exists
        print(f"Register: {r.status_code} {r.text[:200]}")
    
    # Login
    r = requests.post(f"{API_BASE}/auth/login", json={
        "email": email, "password": password
    })
    if r.status_code == 200:
        token = r.json()["access_token"]
        print(f"Login OK")
        return token
    else:
        print(f"Login failed: {r.status_code} {r.text[:200]}")
        return None

def create_kb(token, name="eval_test_kb"):
    """创建知识库"""
    r = requests.post(f"{API_BASE}/kb", headers={"Authorization": f"Bearer {token}"},
                     json={"name": name, "description": "Evaluation test"})
    if r.status_code == 201:
        kb_id = r.json()["id"]
        print(f"KB created: {kb_id}")
        return kb_id
    else:
        print(f"Create KB failed: {r.status_code} {r.text[:200]}")
        return None

# ===== Step 2: Generate PDFs =====
def make_pdf(image_path, text_content, output_path):
    """生成包含图片和文本的 PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Add image
    if os.path.exists(image_path):
        img = RLImage(image_path, width=5*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # Add text
    for line in text_content.split('\n')[:10]:
        if line.strip():
            story.append(Paragraph(line.strip(), styles['Normal']))
    
    doc.build(story)
    return output_path

# ===== Step 3: Upload documents =====
def upload_pdf(token, kb_id, pdf_path, doc_name=None):
    """上传 PDF 到知识库"""
    if doc_name is None:
        doc_name = os.path.basename(pdf_path)
    
    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{API_BASE}/kb/{kb_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files={"files": (doc_name, f, "application/pdf")},
        )
    return r.status_code

# ===== Step 4: Search =====
def search(token, kb_id, query, top_k=10):
    """调用搜索 API"""
    r = requests.post(f"{API_BASE}/search",
                     headers={"Authorization": f"Bearer {token}"},
                     json={"query": query, "kb_ids": [kb_id], "top_k": top_k, "search_mode": "hybrid"})
    if r.status_code == 200:
        return r.json()
    return None

# ===== Main =====
def main():
    print("=" * 60)
    print("全链路评测：PDF → 上传 → 搜索")
    print("=" * 60)
    
    # Auth
    token = register_and_login()
    if not token:
        return
    
    kb_id = create_kb(token)
    if not kb_id:
        return
    
    # Generate PDFs from COCO subset
    print("\n[1/4] Generating PDFs from COCO...")
    coco_dir = os.path.join(DATA_DIR, "data", "coco", "images")
    mm_eval_dir = os.path.join(DATA_DIR, "data", "coco", "mm_eval")
    
    with open(os.path.join(mm_eval_dir, "documents.json")) as f:
        coco_docs = json.load(f)[:30]  # 30 docs for quick test
    
    pdf_dir = os.path.join(DATA_DIR, "data", "eval_pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    pdf_paths = []
    for i, doc in enumerate(coco_docs):
        img_path = os.path.join(coco_dir, doc.get("file_name", ""))
        if not os.path.exists(img_path):
            continue
        
        # Use captions as text content
        captions = doc.get("captions", [])
        text = "\n".join(captions) if captions else doc.get("content", "")
        
        pdf_path = os.path.join(pdf_dir, f"coco_{doc['doc_id']}.pdf")
        make_pdf(img_path, text, pdf_path)
        pdf_paths.append((doc['doc_id'], pdf_path))
        
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(coco_docs)}")
    
    print(f"  Generated {len(pdf_paths)} PDFs")
    
    # Upload
    print("\n[2/4] Uploading PDFs...")
    for i, (doc_id, pdf_path) in enumerate(pdf_paths):
        status = upload_pdf(token, kb_id, pdf_path, f"doc_{doc_id}.pdf")
        if status not in (200, 201):
            print(f"  Upload {doc_id}: {status}")
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(pdf_paths)}")
    
    # Wait for processing
    print("\nWaiting for document processing...")
    time.sleep(10)
    
    # Build queries and qrels
    print("\n[3/4] Building queries...")
    queries = {}
    qrels = {}
    for doc in coco_docs:
        q_id = f"q_{doc['doc_id']}"
        captions = doc.get("captions", [])
        if captions:
            queries[q_id] = captions[0]  # Use first caption as query
            qrels[q_id] = {f"doc_{doc['doc_id']}.pdf"}
    
    print(f"  Queries: {len(queries)}")
    
    # Search
    print("\n[4/4] Running searches...")
    results = {}
    for i, (q_id, query) in enumerate(queries.items()):
        r = search(token, kb_id, query, top_k=10)
        if r:
            results[q_id] = [res["document_name"] for res in r.get("results", [])]
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(queries)}")
    
    # Metrics
    print("\nMetrics:")
    for k in [1, 3, 5, 10]:
        hits = 0
        for q_id, ret in results.items():
            if qrels.get(q_id, set()) & set(ret[:k]):
                hits += 1
        print(f"  Hit@{k}: {hits}/{len(results)} = {hits/max(len(results),1):.4f}")

if __name__ == "__main__":
    main()
