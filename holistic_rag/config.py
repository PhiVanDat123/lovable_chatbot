import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PRODUCTS_CSV = Path(os.getenv("PRODUCTS_CSV", "")).expanduser()
POLICY_DOCX = Path(os.getenv("POLICY_DOCX", "")).expanduser()
NUTRITION_DOCX = Path(os.getenv("NUTRITION_DOCX", "")).expanduser()
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data_index")).expanduser()
# Mặc định dùng Flash-Lite (nhanh, rẻ). Có thể đổi gemini-2.5-flash trong .env.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash-lite")
# Giới hạn token đầu ra — thấp hơn = phản hồi nhanh hơn (ưu tiên <~3s)
CHAT_MAX_OUTPUT_TOKENS = max(
    128, min(2048, int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "384")))
)
# Gemini API (AI Studio): dùng gemini-embedding-001. text-embedding-004 là Vertex AI.
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# Số chunk RAG đưa vào prompt. Mặc định 10: cân bằng recall (câu gộp SP + chính sách) với độ dài prompt.
# Giảm xuống 6–8 nếu ưu tiên latency; tăng tới 12–15 nếu vẫn thiếu ngữ cảnh (tối đa 25).
TOP_K = max(1, min(25, int(os.getenv("TOP_K", "10"))))
# Ingest: batch nhỏ giúp tránh 429 (rate limit) khi embed nhiều chunk
EMBED_BATCH_SIZE = max(1, int(os.getenv("EMBED_BATCH_SIZE", "8")))
EMBED_BATCH_DELAY_SEC = float(os.getenv("EMBED_BATCH_DELAY_SEC", "0"))
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "memory").lower()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "holistic_rag_chunks")

INDEX_META = INDEX_DIR / "meta.jsonl"
INDEX_VECTORS = INDEX_DIR / "vectors.npy"
