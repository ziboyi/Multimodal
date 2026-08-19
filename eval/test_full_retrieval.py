#!/usr/bin/env python3
"""
全链路评测：上传 PDF → 知识库解析 → 搜索 → 评估
用户: eval_test@example.com
"""

import json, os, time, requests

API = "http://localhost:8000/api"
EMAIL = "eval_test@example.com"
PASSWORD = "EvalTest123!"

def login():
    r = requests.post(f"{API}/auth/login", json={"email": EMAIL, "password": PASSWORD})
    if r.status_code == 200:
        return r.json()["access_token"]
    # Register if not exists
    requests.post(f"{API}/auth/register", json={"email": EMAIL, "password": PASSWORD, "full_name": "Eval User"})
    r = requests.post(f"{API}/auth/login", json={"email": EMAIL, "password": PASSWORD})
    return r.json()["access_token"] if r.status_code == 200 else None

def create_kb(token, name):
    r = requests.post(f"{API}/kb", headers={"Authorization": f"Bearer {token}"},
                     json={"name": name, "description": "Evaluation KB"})
    return r.json()["id"] if r.status_code == 201 else None

def upload_pdfs(token, kb_id, pdf_dir):
    uploaded = 0
    for f in sorted(os.listdir(pdf_dir)):
        if not f.endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, f)
        with open(path, "rb") as fh:
            r = requests.post(f"{API}/kb/{kb_id}/documents",
                            headers={"Authorization": f"Bearer {token}"},
                            files={"files": (f, fh, "application/pdf")})
        if r.status_code in (200, 201):
            uploaded += 1
        else:
            print(f"  Failed {f}: {r.status_code}")
    return uploaded

def wait_for_processing(token, kb_id, timeout=300):
    """等待文档处理完成"""
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{API}/kb/{kb_id}/documents", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            docs = r.json().get("items", [])
            statuses = {}
            for d in docs:
                s = d.get("status", "?")
                statuses[s] = statuses.get(s, 0) + 1
            print(f"  Status: {statuses}")
            if all(d.get("status") == "completed" for d in docs) and len(docs) > 0:
                return True
        time.sleep(10)
    return False

def search(token, kb_id, query, top_k=10):
    r = requests.post(f"{API}/search", headers={"Authorization": f"Bearer {token}"},
                     json={"query": query, "kb_ids": [kb_id], "top_k": top_k, "search_mode": "hybrid"})
    if r.status_code == 200:
        return [res["document_name"] for res in r.json().get("results", [])]
    return []

def main():
    print("=" * 60)
    print("全链路评测：PDF → 知识库 → 搜索")
    print("=" * 60)
    
    # 1. Login
    print("\n[1] Login...")
    token = login()
    if not token:
        print("Login failed!")
        return
    print(f"  Logged in as {EMAIL}")
    
    # 2. Create KB
    print("\n[2] Create KB...")
    kb_id = create_kb(token, "eval_multimodal")
    if not kb_id:
        print("Create KB failed!")
        return
    print(f"  KB ID: {kb_id}")
    
    # 3. Upload PDFs
    print("\n[3] Upload PDFs...")
    pdf_dir = "data/eval_pdfs"
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"  Found {len(pdfs)} PDFs")
    uploaded = upload_pdfs(token, kb_id, pdf_dir)
    print(f"  Uploaded {uploaded}/{len(pdfs)}")
    
    # 4. Wait for processing
    print("\n[4] Wait for processing...")
    if not wait_for_processing(token, kb_id):
        print("  Processing timeout or failed!")
        return
    print("  All documents processed!")
    
    # 5. Load retrieval dataset
    print("\n[5] Load retrieval dataset...")
    with open("data/retrieval_dataset.json") as f:
        dataset = json.load(f)
    
    # 6. Search
    print("\n[6] Running searches...")
    queries = dataset["queries"]
    qrels = dataset["qrels"]
    
    results = {}
    for i, (q_id, query) in enumerate(queries.items()):
        ret = search(token, kb_id, query, top_k=10)
        results[q_id] = ret
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(queries)}")
    print(f"  Searched {len(queries)} queries")
    
    # 7. Evaluate
    print("\n[7] Evaluation...")
    for k in [1, 3, 5, 10]:
        hits = 0
        total = 0
        for q_id, ret in results.items():
            rel = set(qrels.get(q_id, []))
            if not rel:
                continue
            total += 1
            if set(ret[:k]) & rel:
                hits += 1
        print(f"  Hit@{k}: {hits}/{total} = {hits/max(total,1):.4f}")

if __name__ == "__main__":
    main()
