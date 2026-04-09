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
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: str | None = None


@dataclass
class SessionState:
    history: list[dict[str, str]]
    last_seen: float


_rag = None
_sessions: dict[str, SessionState] = {}

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


app = FastAPI(title="Holistic Way API", version="1.0.0")

allowed_origins_env = (os.getenv("CORS_ORIGINS") or "").strip()
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
if not allowed_origins:
    # Dev-friendly default; set CORS_ORIGINS in production.
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


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

