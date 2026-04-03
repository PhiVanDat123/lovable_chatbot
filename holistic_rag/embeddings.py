from __future__ import annotations

from google import genai


def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def embed_texts(client: genai.Client, model: str, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    if not texts:
        return []
    all_vals: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.models.embed_content(model=model, contents=batch)
        embs = getattr(resp, "embeddings", None) or []
        if len(embs) != len(batch):
            raise RuntimeError(
                f"Embedding lỗi: batch {len(batch)} nhưng nhận {len(embs)} vector (offset {i})."
            )
        for e in embs:
            vals = getattr(e, "values", None)
            if vals is None:
                raise RuntimeError("Thiếu field values trong embedding response.")
            all_vals.append(list(vals))
    return all_vals
