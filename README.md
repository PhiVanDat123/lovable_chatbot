# Holistic Way RAG Chatbot

Chatbot Gradio trả lời dựa trên **dữ liệu nội bộ**: CSV sản phẩm + 2 file DOCX (chính sách, dinh dưỡng). Pipeline dùng **Google Gemini** (embedding + sinh câu trả lời) và **vector search** (mặc định: index cục bộ để phản hồi nhanh).

## Yêu cầu

- Python **3.10+** (khuyến nghị 3.11)
- Tài khoản Google AI Studio và **Gemini API key**: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Ba file dữ liệu:
  - `holistic_way_products.csv`
  - `policy_data.docx`
  - `nutrition_data.docx`

## Cài đặt nhanh

### 1. Clone / mở thư mục project

```bash
cd chatbot
```

### 2. Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài dependency

```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường

Sao chép file mẫu và chỉnh sửa:

```bash
copy .env.example .env
```

Trên macOS/Linux dùng: `cp .env.example .env`

Mở `.env` và điền tối thiểu:

| Biến | Mô tả |
|------|--------|
| `GEMINI_API_KEY` | **Bắt buộc** — API key Gemini |
| `PRODUCTS_CSV` | Đường dẫn tuyệt đối hoặc tương đối tới file CSV |
| `POLICY_DOCX` | Đường dẫn tới `policy_data.docx` |
| `NUTRITION_DOCX` | Đường dẫn tới `nutrition_data.docx` |

Các biến khác (model, `TOP_K`, `INDEX_DIR`, …) có thể giữ mặc định trong `.env.example`.

### 5. Tạo index vector (bắt buộc trước khi chat)

Lệnh này đọc 3 file, gọi API embedding Google, lưu vào thư mục `INDEX_DIR` (mặc định `./data_index`):

```bash
python ingest.py
```

Khi thành công sẽ có:

- `data_index/meta.jsonl` — metadata từng chunk
- `data_index/vectors.npy` — vector embedding

**Lưu ý:** Mỗi lần đổi nội dung CSV/DOCX, chạy lại `python ingest.py`.

### 6. Chạy giao diện Gradio

```bash
python app.py
```

Mở trình duyệt tại địa chỉ hiển thị trong terminal (mặc định **http://127.0.0.1:7860**).

Đổi cổng (tuỳ chọn): trong `.env` thêm `PORT=8080` rồi chạy lại `python app.py`.

---

## Cấu hình nâng cao

### Model & độ trễ

Trong `.env`:

- `CHAT_MODEL` — mặc định `gemini-2.0-flash` (nhanh, phù hợp chat)
- `EMBED_MODEL` — mặc định `gemini-embedding-001` (Gemini API; `text-embedding-004` là Vertex AI)
- `TOP_K` — số đoạn ngữ cảnh đưa vào prompt (mặc định `10`)
- `EMBED_BATCH_SIZE` — số chunk mỗi lần gọi embedding trong `ingest.py` (mặc định `8`; thử `4` nếu vẫn 429)
- `EMBED_BATCH_DELAY_SEC` — nghỉ (giây) giữa các batch (mặc định `0`; thử `0.5`–`1.5` nếu còn 429)

### Vector backend: Firestore

Mặc định `VECTOR_BACKEND=memory` (đọc `vectors.npy` — nhanh nhất).

Nếu muốn truy vấn vector trên **Firestore**:

1. Tạo project GCP, bật Firestore, cấu hình **Application Default Credentials** (ví dụ `gcloud auth application-default login` hoặc biến `GOOGLE_APPLICATION_CREDENTIALS`).
2. Tạo **vector index** trên collection (field embedding) theo hướng dẫn Google Cloud Console.
3. Trong `.env`: đặt `GOOGLE_CLOUD_PROJECT`, `FIRESTORE_COLLECTION`, `VECTOR_BACKEND=firestore`.
4. Đẩy dữ liệu lên Firestore (sau khi đã chạy `ingest.py` để có index cục bộ):

```bash
python scripts/firestore_ingest.py
```

### Upload file nguồn lên Google Cloud Storage (tuỳ chọn)

Dùng khi cần lưu trữ bản sao CSV/DOCX trên bucket:

1. Trong `.env`: `GCS_BUCKET=tên-bucket` (tuỳ chọn `GCS_PREFIX=thư_mục`).
2. Chạy:

```bash
python scripts/gcs_upload_sources.py
```

---

## Cấu trúc thư mục

```
chatbot/
├── app.py                 # Gradio UI
├── ingest.py              # Build index từ CSV + DOCX
├── requirements.txt
├── .env.example
├── holistic_rag/          # Code RAG (load, embed, retrieve, chat)
├── scripts/
│   ├── firestore_ingest.py
│   └── gcs_upload_sources.py
└── data_index/            # Sinh ra sau ingest.py (không cần commit)
```

---

## Xử lý sự cố thường gặp

| Triệu chứng | Gợi ý |
|-------------|--------|
| `Thiếu GEMINI_API_KEY` | Kiểm tra file `.env` cùng thư mục với `app.py` / `ingest.py`. |
| `Chưa có vector index` | Chạy `python ingest.py` trước `python app.py`. |
| `Không tìm thấy file` | Sửa `PRODUCTS_CSV`, `POLICY_DOCX`, `NUTRITION_DOCX` trong `.env`; Windows có thể dùng `C:/Users/...` hoặc `\\`. |
| `404` / `models/text-embedding-004 is not found` khi `ingest.py` | Bạn đang dùng **Gemini API** (AI Studio). Đặt `EMBED_MODEL=gemini-embedding-001` trong `.env`. `text-embedding-004` dành cho **Vertex AI**, không dùng với `GEMINI_API_KEY`. |
| Lỗi rate limit / quota API | Giảm tần suất gọi, hoặc chờ reset quota; `ingest.py` gọi batch embedding. |
| Firestore vector lỗi | Đảm bảo đã tạo vector index đúng field `embedding` và đã chạy `scripts/firestore_ingest.py`. |

---

## Giấy phép & dữ liệu

Dữ liệu mẫu (CSV, DOCX) thuộc nội dung nội bộ / website Holistic Way — chỉ dùng trong phạm vi được phép.
