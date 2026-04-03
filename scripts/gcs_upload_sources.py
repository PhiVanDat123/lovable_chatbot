"""
Upload 3 file nguồn lên Google Cloud Storage (tùy chọn, để lưu trữ / pipeline khác).

Chạy: python scripts/gcs_upload_sources.py

Cần: GCS_BUCKET trong .env (vd holistic-way-data), quyền storage.objects.create
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv(ROOT / ".env")

BUCKET = os.getenv("GCS_BUCKET", "").strip()
PREFIX = os.getenv("GCS_PREFIX", "holistic_rag_sources").strip().strip("/")


def main() -> None:
    if not BUCKET:
        raise SystemExit("Thiếu GCS_BUCKET trong .env")
    from holistic_rag.config import NUTRITION_DOCX, POLICY_DOCX, PRODUCTS_CSV

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    for p in (PRODUCTS_CSV, POLICY_DOCX, NUTRITION_DOCX):
        if not p.is_file():
            print(f"Bỏ qua (không có file): {p}")
            continue
        dest = f"{PREFIX}/{p.name}" if PREFIX else p.name
        blob = bucket.blob(dest)
        blob.upload_from_filename(str(p))
        print(f"Uploaded gs://{BUCKET}/{dest}")


if __name__ == "__main__":
    main()
