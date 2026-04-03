import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PRODUCTS_CSV = Path(os.getenv("PRODUCTS_CSV", "")).expanduser()
POLICY_DOCX = Path(os.getenv("POLICY_DOCX", "")).expanduser()
NUTRITION_DOCX = Path(os.getenv("NUTRITION_DOCX", "")).expanduser()
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data_index")).expanduser()
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
TOP_K = int(os.getenv("TOP_K", "10"))
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "memory").lower()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "holistic_rag_chunks")

INDEX_META = INDEX_DIR / "meta.jsonl"
INDEX_VECTORS = INDEX_DIR / "vectors.npy"
