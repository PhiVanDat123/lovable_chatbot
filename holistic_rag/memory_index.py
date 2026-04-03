from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from holistic_rag.config import INDEX_META, INDEX_VECTORS


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (mat / n).astype(np.float32)


class MemoryVectorIndex:
    def __init__(self, vectors: np.ndarray, records: list[dict]):
        self.vectors = l2_normalize(vectors.astype(np.float32))
        self.records = records

    @classmethod
    def load(cls, index_dir: Path) -> "MemoryVectorIndex":
        meta_path = index_dir / "meta.jsonl"
        vec_path = index_dir / "vectors.npy"
        if not meta_path.is_file() or not vec_path.is_file():
            raise FileNotFoundError(
                f"Thiếu index trong {index_dir}. Chạy: python ingest.py"
            )
        records: list[dict] = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        vectors = np.load(vec_path)
        if vectors.shape[0] != len(records):
            raise ValueError("Số vector và số bản ghi meta không khớp.")
        return cls(vectors, records)

    def search(self, query_vec: list[float], top_k: int) -> list[tuple[dict, float]]:
        q = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        q = l2_normalize(q)
        sims = (self.vectors @ q.T).ravel()
        idx = np.argsort(-sims)[:top_k]
        out: list[tuple[dict, float]] = []
        for j in idx:
            out.append((self.records[int(j)], float(sims[j])))
        return out


def save_index(index_dir: Path, records: list[dict], vectors: list[list[float]]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    mat = np.array(vectors, dtype=np.float32)
    np.save(index_dir / "vectors.npy", mat)
    with open(index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
