"""
Đẩy index đã build (meta.jsonl + vectors.npy) lên Firestore kèm vector field
để dùng VECTOR_BACKEND=firestore trong app.

Yêu cầu:
- GOOGLE_APPLICATION_CREDENTIALS hoặc `gcloud auth application-default login`
- GOOGLE_CLOUD_PROJECT trong .env
- Tạo vector index trên Firestore cho collection FIRESTORE_COLLECTION, field `embedding`
  (Google Cloud Console → Firestore → Indexes → Vector).

Chạy: python scripts/firestore_ingest.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

load_dotenv(ROOT / ".env")

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
COLLECTION = os.getenv("FIRESTORE_COLLECTION", "holistic_rag_chunks")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data_index")).expanduser()


def main() -> None:
    if not PROJECT:
        raise SystemExit("Thiếu GOOGLE_CLOUD_PROJECT trong .env")
    meta = INDEX_DIR / "meta.jsonl"
    vec = INDEX_DIR / "vectors.npy"
    if not meta.is_file() or not vec.is_file():
        raise SystemExit("Chạy python ingest.py trước để tạo index cục bộ.")

    records = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    mat = np.load(vec)
    if mat.shape[0] != len(records):
        raise SystemExit("meta và vectors không khớp.")

    db = firestore.Client(project=PROJECT)
    coll = db.collection(COLLECTION)
    batch = db.batch()
    count = 0
    for i, rec in enumerate(records):
        doc_id = str(rec.get("id", f"chunk_{i}")).replace("/", "_")[:800]
        ref = coll.document(doc_id)
        emb = mat[i].astype(float).tolist()
        batch.set(
            ref,
            {
                "text": rec.get("text", ""),
                "source": rec.get("source", ""),
                "embedding": Vector(emb),
            },
        )
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()
    print(f"Đã ghi {count} document vào {COLLECTION} (project={PROJECT}).")


if __name__ == "__main__":
    main()
