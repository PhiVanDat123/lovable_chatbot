"""
Tạo index vector cục bộ (Gemini API: gemini-embedding-001) từ CSV + 2 file DOCX.
Chạy: python ingest.py
"""
from __future__ import annotations

from pathlib import Path

from holistic_rag.config import (
    EMBED_BATCH_DELAY_SEC,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    GEMINI_API_KEY,
    INDEX_DIR,
    NUTRITION_DOCX,
    POLICY_DOCX,
    PRODUCTS_CSV,
)
from holistic_rag.embeddings import embed_texts, get_client
from holistic_rag.loaders import load_doc_chunks, load_product_chunks
from holistic_rag.memory_index import save_index


def main() -> None:
    if not GEMINI_API_KEY:
        raise SystemExit("Thiếu GEMINI_API_KEY trong .env")
    missing = [p for p in (PRODUCTS_CSV, POLICY_DOCX, NUTRITION_DOCX) if not p.is_file()]
    if missing:
        raise SystemExit(f"Không tìm thấy file: {missing}. Kiểm tra đường dẫn trong .env")

    chunks: list[dict] = []
    chunks.extend(load_product_chunks(PRODUCTS_CSV))
    chunks.extend(
        load_doc_chunks(POLICY_DOCX, "Chính sách công ty & vận chuyển (policy_data.docx)")
    )
    chunks.extend(
        load_doc_chunks(NUTRITION_DOCX, "Kiến thức dinh dưỡng (nutrition_data.docx)")
    )
    if not chunks:
        raise SystemExit("Không tạo được chunk nào từ dữ liệu.")

    print(
        f"Dang embed {len(chunks)} chunk bang {EMBED_MODEL} "
        f"(batch_size={EMBED_BATCH_SIZE})..."
    )
    client = get_client(GEMINI_API_KEY)
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(
        client,
        EMBED_MODEL,
        texts,
        batch_size=EMBED_BATCH_SIZE,
        delay_sec=EMBED_BATCH_DELAY_SEC,
    )
    save_index(INDEX_DIR, chunks, vectors)
    print(f"Đã lưu index tại: {INDEX_DIR.resolve()}")


if __name__ == "__main__":
    main()
