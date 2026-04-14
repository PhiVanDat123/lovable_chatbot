from __future__ import annotations

from google import genai
from google.genai import types

from holistic_rag.config import (
    CHAT_MAX_OUTPUT_TOKENS,
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


SYSTEM_PROMPT = """Bạn là trợ lý cửa hàng Holistic Way (Singapore). Trả lời ngắn gọn, theo ngữ cảnh.
- Thiếu trong dữ liệu nội bộ → nói rõ và gợi ý holisticway.com.sg / cửa hàng. Không bịa giá, chính sách, thành phần.
- Dinh dưỡng/thành phần/lợi ích: theo ngữ cảnh. Giá: chỉ khi khách hỏi giá/price/SGD.
- Có "Giá khuyến mãi (SGD)" / "Giá gốc (SGD)" trong ngữ cảnh → trả đúng số; không từ chối nếu số nằm trong ngữ cảnh hoặc khớp sản phẩm đang nói.
- Không thay bác sĩ; nhắc chuyên gia khi sức khỏe cá nhân.
- Nhất quán hội thoại: đọc toàn bộ đoạn hội thoại gần đây; mọi ý Khách đã nói (kể cả một từ hoặc cụm rất ngắn) là dữ kiện đã có. Không lặp lại cùng một câu hỏi hay cùng một bộ lựa chọn (A/B/C) đã hỏi. Nếu lượt Trợ lý trước liệt kê lựa chọn và Khách trả lời khớp một trong các hướng đó, coi slot tương ứng đã đủ — chuyển sang câu hỏi khác hoặc recommend, không hỏi lại cùng danh sách lựa chọn.
- Recommend (chủ động, nhiều lượt): Bất cứ khi nào hội thoại hướng tới chọn/so sánh sản phẩm mà dữ kiện chưa đủ — kể cả khi khách không nói từ "gợi ý" — thì không đoán, không liệt kê dài sản phẩm. Có thể nói ngắn phần chung từ ngữ cảnh; gợi ý sản phẩm cụ thể chỉ khi đã đủ dữ kiện. Hỏi thêm qua nhiều lượt: mỗi lượt tối đa 1–3 câu hỏi, chỉ phần còn thiếu (mục đích; ai dùng/tuổi; thai-cho con bú → hỏi + nhắc bác sĩ; dạng viên/lỏng/kem; dị ứng/chay-vegan/halal; ngân sách SGD nếu cần — chỉ số trong ngữ cảnh). Đủ thông tin → tối đa 4–5 sản phẩm, mỗi dòng tên + một lợi ích; mở rộng danh sách chỉ khi khách yêu cầu đầy đủ/tất cả/chi tiết. "Bỏ qua / gợi ý chung" → vài gợi ý chung ngắn.
- Không được phép hỏi chung chung kiểu "Bạn quan tâm đến sản phẩm nào?", hãy chủ động làm rõ theo các "slot" thông tin (chỉ hỏi phần còn thiếu), mỗi lượt 1–3 câu:
  - Nhu cầu/mục tiêu chính (để thu hẹp nhóm sản phẩm)
  - Ai dùng và độ tuổi khi cần; tình trạng đặc biệt có liên quan an toàn (mang thai/cho con bú) → hỏi ngắn + nhắc tham vấn chuyên gia
  - Sở thích/giới hạn: dạng bào chế, dị ứng/kiêng, và ngân sách nếu khách quan tâm
- Nếu khách trả lời mơ hồ thật sự (không khớp lựa chọn nào đã đưa): đừng lặp nguyên câu hỏi; đổi cách hỏi hoặc thu hẹp lựa chọn.
- Tốc độ: Nếu lượt này chỉ để làm rõ (chưa recommend danh sách), giữ phản hồi rất ngắn — vài câu, không mở bài/dàn ý dài.
- Khi nào recommend sản phẩm cho khách xong rồi thì chỉ cần hỏi khách có cần hỗ trợ thêm gì không?"""

def _strip_chat_footer(text: str) -> str:
    # Đảm bảo luôn là chuỗi để tránh lỗi khi content là list / object khác
    if not isinstance(text, str):
        text = str(text or "")
    t = text.strip()
    if "\n---\n" in t:
        t = t.split("\n---\n")[0].strip()
    return t


def _normalize_user_message(user_message) -> str:
    """
    Gradio đôi khi gửi list[Message] thay vì chuỗi đơn.
    Hàm này ép về chuỗi hỏi cuối cùng của user.
    """
    if isinstance(user_message, list):
        # messages format: list[dict(role, content)]
        for msg in reversed(user_message):
            if isinstance(msg, dict) and (msg.get("role") == "user"):
                return str(msg.get("content") or "")
        if user_message:
            return str(user_message[-1])
        return ""
    return str(user_message or "")


def _history_blob(history: list | None, max_chars: int = 3200) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for msg in history[-8:]:
        role = (msg.get("role") or "").strip()
        content = _strip_chat_footer(msg.get("content") or "")
        if not content:
            continue
        label = "Khách" if role == "user" else "Trợ lý"
        lines.append(f"{label}: {content[:900]}")
    blob = "\n".join(lines)
    if len(blob) > max_chars:
        blob = blob[-max_chars:]
    return blob


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

    def _retrieve(self, query: str, top_k: int | None = None) -> list[tuple[dict, float]]:
        k = top_k if top_k is not None else TOP_K
        qv = embed_texts(self.client, EMBED_MODEL, [query], batch_size=1)[0]
        if VECTOR_BACKEND == "firestore":
            return self._retrieve_firestore(qv, limit=k)
        idx = self._load_memory_index()
        return idx.search(qv, k)

    def _retrieve_firestore(self, query_vec: list[float], limit: int | None = None) -> list[tuple[dict, float]]:
        from google.cloud import firestore
        from google.cloud.firestore_v1.vector import Vector
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

        if not GOOGLE_CLOUD_PROJECT:
            raise ValueError("VECTOR_BACKEND=firestore cần GOOGLE_CLOUD_PROJECT")
        lim = limit if limit is not None else TOP_K
        db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        coll = db.collection(FIRESTORE_COLLECTION)
        dist_field = "vector_distance"
        vq = coll.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_vec),
            distance_measure=DistanceMeasure.COSINE,
            limit=lim,
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

    def answer(self, user_message, history: list | None = None) -> tuple[str, str]:
        raw_msg = _normalize_user_message(user_message)
        hist_blob = _history_blob(history)
        retrieve_q = raw_msg.strip()
        if hist_blob:
            retrieve_q = f"{hist_blob}\n\nCâu hỏi hiện tại: {raw_msg.strip()}"

        # Retrieval tổng quát: không hardcode từ khóa, chỉ dùng vector search + lịch sử hội thoại
        hits = self._retrieve(retrieve_q, top_k=None)
        context_blocks = []
        for h, score in hits:
            context_blocks.append(f"---\n(relevance ~{score:.3f})\n{h.get('text', '')}")
        context = "\n".join(context_blocks) if context_blocks else "(Không có đoạn liên quan.)"

        convo_section = ""
        if hist_blob:
            convo_section = (
                "Hội thoại gần đây (để hiểu câu hỏi tiếp theo, ví dụ \"giá\" là của sản phẩm nào):\n"
                f"{hist_blob}\n\n"
            )
        user_parts = [
            convo_section,
            "Ngữ cảnh từ cơ sở tri thức nội bộ:\n",
            context,
            "\n\nCâu hỏi khách:\n",
            raw_msg,
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
            max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
        )
        resp = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
            config=config,
        )
        text = (resp.text or "").strip() or "Không tạo được câu trả lời. Vui lòng thử lại."
        return text, ""
