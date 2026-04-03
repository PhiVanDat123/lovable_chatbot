from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document


def read_docx_paragraphs(path: Path) -> str:
    doc = Document(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(parts)


def load_product_chunks(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    def pick(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    c_products = pick("Products", "Product")
    c_cat = pick("Category")
    c_disc = pick("Discounted Price (SGD $)", "Discounted Price")
    c_price = pick("Price (SGD $)", "Price")
    c_desc = pick("Description")
    c_link = pick("Product Link", "ProductLink")
    if not c_products:
        c_products = df.columns[0]

    chunks: list[dict] = []
    for i, row in df.iterrows():
        name = str(row.get(c_products, "")).strip()
        if not name:
            continue
        lines = [
            "[Nguồn: Danh mục sản phẩm Holistic Way]",
            f"Tên sản phẩm: {name}",
        ]
        if c_cat and pd.notna(row.get(c_cat)):
            lines.append(f"Danh mục: {row.get(c_cat)}")
        if c_disc and pd.notna(row.get(c_disc)):
            lines.append(f"Giá khuyến mãi (SGD): {row.get(c_disc)}")
        if c_price and pd.notna(row.get(c_price)):
            lines.append(f"Giá gốc (SGD): {row.get(c_price)}")
        if c_desc and pd.notna(row.get(c_desc)):
            lines.append(f"Mô tả: {row.get(c_desc)}")
        if c_link and pd.notna(row.get(c_link)):
            lines.append(f"Liên kết: {row.get(c_link)}")
        text = "\n".join(lines)
        chunks.append({"text": text, "source": "products", "id": f"product_row_{i}"})
    return chunks


def load_doc_chunks(path: Path, source_label: str, max_chars: int = 1600, overlap: int = 200) -> list[dict]:
    full = read_docx_paragraphs(path)
    if not full.strip():
        return []
    paragraphs = [p.strip() for p in full.split("\n\n") if p.strip()]
    merged: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                merged.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                merged.extend(_split_hard(p, max_chars, overlap))
                buf = ""
    if buf:
        merged.append(buf)

    src_key = "policy" if "policy" in path.name.lower() else "nutrition"
    chunks: list[dict] = []
    for i, block in enumerate(merged):
        for part, piece in enumerate(_split_hard(block, max_chars, overlap)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "text": f"[Nguồn: {source_label}]\n{piece}",
                    "source": src_key,
                    "id": f"{src_key}_{i}_{part}",
                }
            )
    return chunks


def _split_hard(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return out
