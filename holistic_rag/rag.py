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


SYSTEM_PROMPT = """Bạn là trợ lý cửa hàng Holistic Way (Singapore). Trả lời ngắn gọn, rõ ràng, dựa trên ngữ cảnh được cung cấp.
- Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ là không có trong dữ liệu nội bộ và đề nghị khách liên hệ holisticway.com.sg hoặc cửa hàng.
- Không bịa giá, chính sách hoặc thành phần không có trong ngữ cảnh.
- Khi khách hỏi về dinh dưỡng / thành phần / lợi ích sức khỏe của sản phẩm: ưu tiên trả lời đúng các thông tin dinh dưỡng/thành phần/lợi ích trong ngữ cảnh. Chỉ nhắc tới giá khi khách hỏi rõ về giá/price/SGD.
- Khi ngữ cảnh là danh mục sản phẩm và có dòng "Giá khuyến mãi (SGD)" / "Giá gốc (SGD)": hãy trả lời đúng các con số đó (có thể ghi SGD). Tuyệt đối không nói "không thể cung cấp giá" hoặc từ chối nếu các mức giá đã nằm trong ngữ cảnh hoặc khách đang hỏi về sản phẩm vừa được nhắc trong hội thoại và bạn tìm được giá trong ngữ cảnh.
- Có thể trả lời bằng tiếng Việt hoặc tiếng Anh tùy ngôn ngữ câu hỏi.
- Không đưa lời khuyên y khoa thay thế bác sĩ; nhắc tham vấn chuyên gia khi liên quan sức khỏe cá nhân.
- Gợi ý sản phẩm — hỏi trước khi liệt kê dài: Nếu khách yêu cầu gợi ý/danh sách mà còn chung chung (ví dụ chỉ nói "gợi ý vài sản phẩm", "có gì tốt", "recommend cho tôi") và trong hội thoại chưa rõ mục đích (da, ngủ, tiêu hóa, miễn dịch, xương khớp, não/mắt…) hoặc chưa rõ đối tượng (người lớn/trẻ em, giới tính/độ tuổi nếu cần thiết cho lựa chọn an toàn), hãy chủ động hỏi ngắn 1–2 câu để lấy thông tin cần thiết trước; có thể gợi ý 2–3 hướng trả lời (ví dụ mục đích chính, độ tuổi, dị ứng/kiêng halal-chay). Không liệt kê 4–5 sản phẩm ngay trong lượt đó trừ khi khách đã nói rõ mục đích và đối tượng trong câu hỏi hoặc trong hội thoại gần đây.
- Gợi ý sản phẩm — recommend khi đã đủ thông tin: Khi khách đã nêu rõ mục đích và (nếu liên quan) đối tượng sử dụng, hoặc đã trả lời câu hỏi làm rõ của bạn, hoặc hỏi cụ thể một nhóm (ví dụ "omega-3 cho trẻ", "probiotic cho con"), hãy dựa trên ngữ cảnh RAG để gợi ý tối đa 4–5 sản phẩm; mỗi sản phẩm một dòng ngắn (tên + một cụm lợi ích). Chỉ liệt kê nhiều hơn nếu khách yêu cầu rõ "đầy đủ", "tất cả" hoặc "chi tiết từng loại".
- Nếu khách nói "bỏ qua / gợi ý chung / không ràng buộc": có thể đưa vài gợi ý chung từ ngữ cảnh nhưng vẫn ngắn gọn và nhắc tham vấn chuyên gia khi liên quan sức khỏe cá nhân."""


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
