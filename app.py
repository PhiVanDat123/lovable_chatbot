"""
Gradio UI — RAG Holistic Way (Gemini + vector search cục bộ hoặc Firestore).
Chạy: python app.py
"""
from __future__ import annotations

import os
import time

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

_rag = None


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


def respond(message: str, history: list | None):
    # Gradio 6+: Chatbot dùng list[dict] với keys role / content
    hist = list(history) if history is not None else []
    message = (message or "").strip()
    if not message:
        return hist, ""
    t0 = time.perf_counter()
    try:
        rag = get_rag()
        reply, sources = rag.answer(message, hist)
    except Exception as e:
        reply = f"Lỗi: {e}"
        sources = ""
    elapsed = time.perf_counter() - t0
    footer = f"\n\n---\n_Phản hồi trong ~{elapsed:.1f}s_"
    if sources:
        reply = f"{reply}\n\n{sources}{footer}"
    else:
        reply = f"{reply}{footer}"
    hist = hist + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return hist, ""


def build_ui():
    with gr.Blocks(title="Holistic Way — Trợ lý dữ liệu") as demo:
        gr.Markdown(
            """
            ## Holistic Way — Chatbot (Gemini + RAG)
            Trả lời dựa trên **danh mục sản phẩm (CSV)**, **chính sách (policy)** và **dinh dưỡng (nutrition)**.
            Đặt `GEMINI_API_KEY` trong `.env` và chạy `python ingest.py` trước lần đầu.
            """
        )
        # Lưu session chat ngay trên giao diện (ưu tiên lưu ở trình duyệt nếu Gradio hỗ trợ)
        SessionState = getattr(gr, "BrowserState", gr.State)
        session_history = SessionState([])

        chat = gr.Chatbot(label="Trò chuyện", height=480)
        with gr.Row():
            box = gr.Textbox(
                placeholder="Ví dụ: Giao hàng trong nước mất bao lâu? / Giá Rose Placenta Collagen Shot?",
                scale=5,
                show_label=False,
            )
            send = gr.Button("Gửi", variant="primary", scale=1)
            clear = gr.Button("Xóa chat", variant="secondary", scale=1)

        def on_load(hist):
            # Đồng bộ lịch sử từ state → Chatbot khi tải UI
            return list(hist or [])

        def on_msg(msg, hist, sess_hist):
            # Chatbot tự giữ state theo session, nhưng ta mirror thêm vào session_history
            new_hist, cleared = respond(msg, hist)
            return new_hist, cleared, new_hist

        def on_clear():
            return [], []

        demo.load(on_load, [session_history], [chat])

        send.click(on_msg, [box, chat, session_history], [chat, box, session_history])
        box.submit(on_msg, [box, chat, session_history], [chat, box, session_history])
        clear.click(on_clear, [], [chat, session_history])

    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(primary_hue="emerald"),
        share=True,
    )
