from __future__ import annotations

from google import genai
from google.genai import types

from holistic_rag.config import (
    CHAT_MODEL,
    EMBED_MODEL,
    GEMINI_API_KEY,
    TOP_K,
    VECTOR_BACKEND,
    FIRESTORE_COLLECTION,
    GOOGLE_CLOUD_PROJECT,
    INDEX_DIR,
)
from holistic_rag.embeddings import embed_texts, get_client
from holistic_rag.memory_index import MemoryVectorIndex


SYSTEM_PROMPT = """Bạn là trợ lý cửa hàng Holistic Way (Singapore). Trả lời ngắn gọn, rõ ràng, dựa CHỈ trên ngữ cảnh được cung cấp.
- Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ là không có trong dữ liệu nội bộ và đề nghị khách liên hệ holisticway.com.sg hoặc cửa hàng.
- Không bịa giá, chính sách hoặc thành phần không có trong ngữ cảnh.
- Có thể trả lời bằng tiếng Việt hoặc tiếng Anh tùy ngôn ngữ câu hỏi.
- Không đưa lời khuyên y khoa thay thế bác sĩ; nhắc tham vấn chuyên gia khi liên quan sức khỏe cá nhân."""


class RAGChat:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Thiếu GEMINI_API_KEY trong .env")
        self.client = get_client(GEMINI_API_KEY)
        self._index: MemoryVectorIndex | None = None
        self._firestore_ready = False

    def _load_memory_index(self) -> MemoryVectorIndex:
        if self._index is None:
            self._index = MemoryVectorIndex.load(INDEX_DIR)
        return self._index

    def _retrieve(self, query: str) -> list[tuple[dict, float]]:
        qv = embed_texts(self.client, EMBED_MODEL, [query], batch_size=1)[0]
        if VECTOR_BACKEND == "firestore":
            return self._retrieve_firestore(qv)
        idx = self._load_memory_index()
        return idx.search(qv, TOP_K)

    def _retrieve_firestore(self, query_vec: list[float]) -> list[tuple[dict, float]]:
        from google.cloud import firestore
        from google.cloud.firestore_v1.vector import Vector
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

        if not GOOGLE_CLOUD_PROJECT:
            raise ValueError("VECTOR_BACKEND=firestore cần GOOGLE_CLOUD_PROJECT")
        db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        coll = db.collection(FIRESTORE_COLLECTION)
        dist_field = "vector_distance"
        vq = coll.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_vec),
            distance_measure=DistanceMeasure.COSINE,
            limit=TOP_K,
            distance_result_field=dist_field,
        )
        out: list[tuple[dict, float]] = []
        for snap in vq.get():
            d = snap.to_dict() or {}
            text = d.get("text", "")
            src = d.get("source", "")
            dist = d.get(dist_field)
            sim = 1.0 - float(dist) if dist is not None else 0.0
            out.append(({"text": text, "source": src, "id": snap.id}, sim))
        return out

    def answer(self, user_message: str, history: list | None = None) -> tuple[str, str]:
        hits = self._retrieve(user_message)
        context_blocks = []
        for h, score in hits:
            context_blocks.append(f"---\n(relevance ~{score:.3f})\n{h.get('text', '')}")
        context = "\n".join(context_blocks) if context_blocks else "(Không có đoạn liên quan.)"

        user_parts = [
            "Ngữ cảnh từ cơ sở tri thức nội bộ:\n",
            context,
            "\n\nCâu hỏi khách:\n",
            user_message,
        ]
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text="".join(user_parts))],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1024,
        )
        resp = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
            config=config,
        )
        text = (resp.text or "").strip() or "Không tạo được câu trả lời. Vui lòng thử lại."
        sources_lines = []
        for h, sc in hits[:5]:
            src = h.get("source", "")
            tid = str(h.get("id", ""))[:48]
            sources_lines.append(f"- [{src}] {tid} (score {sc:.3f})")
        sources = "Nguồn tham chiếu:\n" + "\n".join(sources_lines) if sources_lines else ""
        return text, sources
