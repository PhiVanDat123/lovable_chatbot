"""
FastAPI backend for Lovable frontend.

Contract:
POST /api/chat
{
  "session_id": "string",
  "message": "string"
}
-> { "reply": "string", "sources": "string | null" }
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: str | None = None


class Product(BaseModel):
    id: str
    name: str
    category: str | None = None
    description: str | None = None
    price_sgd: float | None = None
    promo_price_sgd: float | None = None
    product_link: str | None = None


class ProductListResponse(BaseModel):
    total: int
    items: list[Product]


@dataclass
class SessionState:
    history: list[dict[str, str]]
    last_seen: float


_rag = None
_sessions: dict[str, SessionState] = {}

# Product catalog cache (CSV -> normalized list)
_products_cache: list[Product] | None = None
_products_by_id: dict[str, Product] | None = None
_products_mtime: float | None = None

# Session memory limits (in-memory). For production, consider Redis.
_MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "2000"))
_SESSION_TTL_SEC = int(os.getenv("SESSION_TTL_SEC", "21600"))  # 6 hours
_MAX_TURNS_PER_SESSION = int(os.getenv("MAX_TURNS_PER_SESSION", "40"))


def _prune_sessions(now: float) -> None:
    if not _sessions:
        return
    # Drop expired
    expired = [sid for sid, st in _sessions.items() if (now - st.last_seen) > _SESSION_TTL_SEC]
    for sid in expired:
        _sessions.pop(sid, None)
    # Soft cap: drop least-recently-seen if too many
    if len(_sessions) <= _MAX_SESSIONS:
        return
    for sid, _ in sorted(_sessions.items(), key=lambda kv: kv[1].last_seen)[: max(1, len(_sessions) - _MAX_SESSIONS)]:
        _sessions.pop(sid, None)


def get_rag():
    global _rag
    if _rag is None:
        from pathlib import Path

        from holistic_rag.config import INDEX_DIR, VECTOR_BACKEND
        from holistic_rag.rag import RAGChat

        if VECTOR_BACKEND == "memory":
            if not (Path(INDEX_DIR) / "vectors.npy").is_file():
                raise FileNotFoundError(
                    f"Chưa có vector index tại {INDEX_DIR}. Chạy: python ingest.py"
                )
        _rag = RAGChat()
    return _rag


def _get_or_create_session(session_id: str, now: float) -> SessionState:
    st = _sessions.get(session_id)
    if st is None:
        st = SessionState(history=[], last_seen=now)
        _sessions[session_id] = st
    st.last_seen = now
    return st


def _append_turn(st: SessionState, role: str, content: str) -> None:
    st.history.append({"role": role, "content": content})
    # Keep a rolling window to bound prompt size
    max_msgs = max(4, _MAX_TURNS_PER_SESSION * 2)
    if len(st.history) > max_msgs:
        st.history = st.history[-max_msgs:]


def _require_api_key(x_api_key: str | None) -> None:
    expected = (os.getenv("API_KEY") or "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", (text or "").strip())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "item"


def _to_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _load_products_from_csv() -> tuple[list[Product], dict[str, Product]]:
    from holistic_rag.config import PRODUCTS_CSV

    if not PRODUCTS_CSV:
        raise HTTPException(status_code=500, detail="Missing PRODUCTS_CSV")
    if not PRODUCTS_CSV.is_file():
        raise HTTPException(status_code=500, detail=f"PRODUCTS_CSV not found: {PRODUCTS_CSV}")

    import pandas as pd

    df = pd.read_csv(PRODUCTS_CSV)
    # Expected columns (best effort):
    # Products, Category, Discounted Price (SGD $), Price (SGD $), Description, Product Link
    col_name = next((c for c in df.columns if c.strip().lower() in {"products", "product"}), None)
    if not col_name:
        raise HTTPException(status_code=500, detail=f"CSV missing product name column in {list(df.columns)}")

    col_category = next((c for c in df.columns if c.strip().lower() == "category"), None)
    col_promo = next((c for c in df.columns if "discount" in c.lower()), None)
    col_price = next((c for c in df.columns if c.strip().lower().startswith("price")), None)
    col_desc = next((c for c in df.columns if c.strip().lower() == "description"), None)
    col_link = next((c for c in df.columns if "link" in c.lower()), None)

    products: list[Product] = []
    by_id: dict[str, Product] = {}

    for _, row in df.iterrows():
        name = str(row.get(col_name) or "").strip()
        if not name:
            continue
        category = str(row.get(col_category) or "").strip() if col_category else ""
        description = str(row.get(col_desc) or "").strip() if col_desc else ""
        link = str(row.get(col_link) or "").strip() if col_link else ""

        price = _to_float(row.get(col_price)) if col_price else None
        promo = _to_float(row.get(col_promo)) if col_promo else None

        base = _slugify(name)
        # Avoid collisions when CSV has repeated names (e.g. bundles)
        suffix = _slugify(link)[:24] if link else ""
        pid = base if not suffix else f"{base}-{suffix}"
        # Ensure uniqueness
        if pid in by_id:
            i = 2
            while f"{pid}-{i}" in by_id:
                i += 1
            pid = f"{pid}-{i}"

        p = Product(
            id=pid,
            name=name,
            category=category or None,
            description=description or None,
            price_sgd=price,
            promo_price_sgd=promo,
            product_link=link or None,
        )
        products.append(p)
        by_id[p.id] = p

    # Stable default ordering: category then name
    products.sort(key=lambda x: ((x.category or "").lower(), x.name.lower()))
    return products, by_id


def _get_products_cached() -> tuple[list[Product], dict[str, Product]]:
    global _products_cache, _products_by_id, _products_mtime
    from holistic_rag.config import PRODUCTS_CSV

    if not PRODUCTS_CSV or not PRODUCTS_CSV.is_file():
        raise HTTPException(status_code=500, detail="PRODUCTS_CSV is not configured or missing")

    mtime = PRODUCTS_CSV.stat().st_mtime
    if _products_cache is None or _products_by_id is None or _products_mtime != mtime:
        products, by_id = _load_products_from_csv()
        _products_cache = products
        _products_by_id = by_id
        _products_mtime = mtime
    return _products_cache, _products_by_id


app = FastAPI(title="Holistic Way API", version="1.0.0")

allowed_origins_env = (os.getenv("CORS_ORIGINS") or "").strip()
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
if not allowed_origins:
    # Dev-friendly default; set CORS_ORIGINS in production.
    allowed_origins = ["*"]

# Starlette: allow_credentials=True với allow_origins=["*"] KHÔNG gửi Access-Control-Allow-Origin
# → trình duyệt chặn CORS. Chỉ bật credentials khi liệt kê origin cụ thể.
_use_wildcard_cors = allowed_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not _use_wildcard_cors,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/products", response_model=ProductListResponse)
def list_products(
    q: str | None = Query(default=None, description="Search query for product name/description"),
    category: str | None = Query(default=None, description="Filter by category (exact match)"),
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    limit: int = Query(default=60, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> ProductListResponse:
    _require_api_key(x_api_key)

    products, _ = _get_products_cached()
    items = products

    if category:
        cat_norm = category.strip().lower()
        items = [p for p in items if (p.category or "").strip().lower() == cat_norm]

    if q:
        qn = q.strip().lower()
        if qn:
            items = [
                p
                for p in items
                if (qn in p.name.lower())
                or (p.description is not None and qn in p.description.lower())
            ]

    def effective_price(p: Product) -> float | None:
        return p.promo_price_sgd if p.promo_price_sgd is not None else p.price_sgd

    if min_price is not None:
        items = [p for p in items if (effective_price(p) is not None and effective_price(p) >= min_price)]
    if max_price is not None:
        items = [p for p in items if (effective_price(p) is not None and effective_price(p) <= max_price)]

    total = len(items)
    page = items[offset : offset + limit]
    return ProductListResponse(total=total, items=page)


@app.get("/api/products/{product_id}", response_model=Product)
def get_product(
    product_id: str,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Product:
    _require_api_key(x_api_key)
    _, by_id = _get_products_cached()
    p = by_id.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return p


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_api_key: str | None = Header(default=None, alias="x-api-key")) -> ChatResponse:
    _require_api_key(x_api_key)

    msg = (req.message or "").strip()
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    now = time.time()
    _prune_sessions(now)
    st = _get_or_create_session(session_id, now)

    try:
        rag = get_rag()
        reply, sources = rag.answer(msg, st.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    _append_turn(st, "user", msg)
    _append_turn(st, "assistant", reply)

    return ChatResponse(reply=reply, sources=sources or None)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)

