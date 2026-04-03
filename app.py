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


def respond(message: str, history: list):
    message = (message or "").strip()
    if not message:
        return history, ""
    t0 = time.perf_counter()
    try:
        rag = get_rag()
        reply, sources = rag.answer(message, history)
    except Exception as e:
        reply = f"Lỗi: {e}"
        sources = ""
    elapsed = time.perf_counter() - t0
    footer = f"\n\n---\n_Phản hồi trong ~{elapsed:.1f}s_"
    if sources:
        reply = f"{reply}\n\n{sources}{footer}"
    else:
        reply = f"{reply}{footer}"
    history = history + [(message, reply)]
    return history, ""


def build_ui():
    with gr.Blocks(
        title="Holistic Way — Trợ lý dữ liệu",
        theme=gr.themes.Soft(primary_hue="emerald"),
    ) as demo:
        gr.Markdown(
            """
            ## Holistic Way — Chatbot (Gemini + RAG)
            Trả lời dựa trên **danh mục sản phẩm (CSV)**, **chính sách (policy)** và **dinh dưỡng (nutrition)**.
            Đặt `GEMINI_API_KEY` trong `.env` và chạy `python ingest.py` trước lần đầu.
            """
        )
        chat = gr.Chatbot(label="Trò chuyện", height=480)
        with gr.Row():
            box = gr.Textbox(
                placeholder="Ví dụ: Giao hàng trong nước mất bao lâu? / Giá Rose Placenta Collagen Shot?",
                scale=5,
                show_label=False,
            )
            send = gr.Button("Gửi", variant="primary", scale=1)

        def on_msg(msg, hist):
            return respond(msg, hist)

        send.click(on_msg, [box, chat], [chat, box])
        box.submit(on_msg, [box, chat], [chat, box])

    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    build_ui().launch(server_name="0.0.0.0", server_port=port)
