# Buổi 07 - RAG với Gemini Embedding và ChromaDB

Buổi 07 xây dựng một pipeline RAG tối giản cho người mới học: đọc chunk JSON đã chuẩn bị, validate dữ liệu, tạo embedding bằng Gemini, lưu vào ChromaDB persistent, retrieval theo câu hỏi, confidence gate, generation có grounding và citation mapping.

> Demo này phục vụ học tập. Kết quả không phải tư vấn pháp lý, tài chính hoặc nghiệp vụ ngân hàng.

## 1. Mục tiêu

- Dùng lại dữ liệu chunk từ Buổi 05.
- Tạo semantic index bằng Gemini embedding thật.
- Lưu vector vào ChromaDB local persistent.
- Query bằng embedding cùng model/dimension với index.
- Chặn câu trả lời khi evidence yếu.
- Map citation từ metadata thật, không tin citation do LLM tự tạo.
- Cung cấp CLI, Streamlit UI và unittest offline.

## 2. Quan hệ với Buổi 05 và Buổi 06

- **Buổi 05** chuẩn bị dữ liệu: OCR/PDF parsing/chunking và ghi `rag_foundation/buoi_05/output/chunks/chunks.json` dạng JSON array.
- **Buổi 06** là demo RAG trước đó.
- **Buổi 07** không OCR lại, không parse PDF lại, không chunk lại. Buổi 07 chỉ đọc JSON chunks đã có từ Buổi 05.

## 3. Sơ đồ pipeline

```text
Buổi 05 chunks.json
        |
        v
Validate chunk contract
        |
        v
Gemini embedding: title/source + text
        |
        v
ChromaDB PersistentClient, cosine collection
        |
        v
Query embedding cùng model/dimension
        |
        v
Semantic retrieval top-k + distance
        |
        v
Confidence gate bằng RAG_MAX_DISTANCE
        |
        +-- không đủ evidence --> insufficient_evidence, không gọi generation
        |
        v
Gemini generation chỉ với accepted evidence
        |
        v
Code map [E1], [E2] sang citation metadata thật
```

## 4. Cấu trúc thư mục

```text
rag_foundation/buoi_07/
├── SPEC_buoi_07.md
├── buoi_07.md
├── rag.py
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   └── fixtures/
│       └── chunks_sample.json
└── storage/
    └── .gitkeep
```

`storage/chroma/` sẽ được tạo khi chạy index thật và được `.gitignore` bỏ qua.

## 5. Điều kiện đầu vào

Cần có file JSON chunk từ Buổi 05:

```text
rag_foundation/buoi_05/output/chunks/chunks.json
```

Mỗi chunk cần có các field:

- `chunk_id`
- `strategy`
- `source`
- `page_start`
- `page_end`
- `text`

File phải là JSON array hoặc object có field `chunks` là array.

## 6. Dùng `.venv` Buổi 05

Buổi 07 dùng lại virtual environment của Buổi 05. Không tạo `.venv` mới.

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python --version
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe --version
```

## 7. Cài requirements

Chỉ cài các package trực tiếp trong `requirements.txt`.

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python -m pip install -r rag_foundation/buoi_07/requirements.txt
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe -m pip install -r rag_foundation\buoi_07\requirements.txt
```

## 8. Tạo `.env` từ `.env.example`

Linux/macOS:

```bash
cp rag_foundation/buoi_07/.env.example rag_foundation/buoi_07/.env
```

Windows PowerShell:

```powershell
Copy-Item rag_foundation\buoi_07\.env.example rag_foundation\buoi_07\.env
```

Sau đó điền `GEMINI_API_KEY` trong `.env`. Không commit `.env`.

## 9. Biến môi trường

| Biến | Ý nghĩa |
|---|---|
| `GEMINI_API_KEY` | API key dùng để gọi Gemini embedding/generation. Không được in ra log/UI. |
| `GEMINI_EMBEDDING_MODEL` | Model tạo embedding, ví dụ `gemini-embedding-2`. |
| `GEMINI_EMBEDDING_DIM` | Số chiều vector. Code validate trong khoảng `128..3072`. Index/query phải cùng dimension. |
| `GEMINI_GENERATION_MODEL` | Model tạo câu trả lời tổng hợp. |
| `DEFAULT_TOP_K` | Top-k mặc định cho query CLI khi không truyền `--top-k`; trong khoảng `1..20`. |
| `RAG_MAX_DISTANCE` | Ngưỡng confidence gate cho cosine distance. Distance thấp hơn thường liên quan hơn. |

## 10. Validate dữ liệu

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python rag_foundation/buoi_07/rag.py validate --strategy hierarchical
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe rag_foundation\buoi_07\rag.py validate --strategy hierarchical
```

Có thể đổi strategy: `hierarchical`, `semantic`, `fixed-size`.

## 11. Status collection

`status` là read-only: không tạo collection rỗng, không gọi Gemini.

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python rag_foundation/buoi_07/rag.py status --strategy hierarchical
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe rag_foundation\buoi_07\rag.py status --strategy hierarchical
```

## 12. Index dữ liệu

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python rag_foundation/buoi_07/rag.py index --strategy hierarchical
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe rag_foundation\buoi_07\rag.py index --strategy hierarchical
```

Index dùng `upsert`, nên chạy lại cùng collection không tạo duplicate.

## 13. Reset đúng collection đích rồi index

`--reset` chỉ xóa collection đích theo strategy/model/dimension hiện tại, không xóa toàn bộ storage và không ảnh hưởng collection khác. Code tạo và validate toàn bộ embeddings trước khi reset collection cũ.

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python rag_foundation/buoi_07/rag.py index --strategy hierarchical --reset
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe rag_foundation\buoi_07\rag.py index --strategy hierarchical --reset
```

## 14. Query CLI

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python rag_foundation/buoi_07/rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe rag_foundation\buoi_07\rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"
```

## 15. Chạy test

Test dùng `unittest`, mock Gemini API, temporary Chroma storage và không cần Internet.

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python -m unittest discover -s rag_foundation/buoi_07/tests -v
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe -m unittest discover -s rag_foundation\buoi_07\tests -v
```

## 16. Chạy Streamlit

Linux/macOS:

```bash
rag_foundation/buoi_05/.venv/bin/python -m streamlit run rag_foundation/buoi_07/app.py
```

Windows PowerShell:

```powershell
rag_foundation\buoi_05\.venv\Scripts\python.exe -m streamlit run rag_foundation\buoi_07\app.py
```

## 17. Khái niệm chính

### Strategy

Chunk có thể thuộc một trong ba strategy: `hierarchical`, `semantic`, `fixed-size`. Mỗi strategy được index vào một collection riêng.

### Embedding model

Model dùng để biến text thành vector. Index và query phải dùng cùng model.

### Embedding dimension

Số chiều vector embedding. Index và query phải cùng dimension. Nếu đổi dimension, collection identity sẽ đổi.

### Collection identity

Tên collection có dạng:

```text
nhnn-<strategy>-<dimension>-<model_hash>
```

Tên này phân biệt strategy, embedding model và embedding dimension.

### Top-k

Số lượng evidence muốn retrieve. Code dùng `min(top_k, collection.count())` để không hỏi quá số record đang có.

### Cosine distance

Chroma collection dùng cosine distance. Distance thấp hơn thường liên quan hơn, nhưng không phải xác suất.

### RAG_MAX_DISTANCE

Ngưỡng demo để quyết định evidence có được đưa vào generation không:

```text
accepted = distance <= RAG_MAX_DISTANCE
```

### Confidence gate

Nếu không có evidence đạt ngưỡng, code trả `insufficient_evidence` và không gọi Gemini generation.

### Retrieval-only

Nếu retrieval có evidence đạt ngưỡng nhưng generation lỗi hoặc trả rỗng, code trả `retrieval_only`, vẫn giữ evidence để người học kiểm tra nguồn.

### Citation

Gemini chỉ được viết label như `[E1]`. Code map label đó sang metadata thật:

```text
[Nguồn: <source>, tr. <N hoặc N-M>, chunk: <chunk_id>]
```

Label không hợp lệ như `[E99]` bị loại và ghi warning.

## 18. Dừng Streamlit

Trong terminal đang chạy Streamlit, nhấn:

```text
Ctrl+C
```

Không nên mở nhiều tiến trình Streamlit cùng lúc cho cùng app.

## 19. Troubleshooting

### Thiếu package

Chạy lại cài requirements bằng đúng `.venv` Buổi 05:

```bash
rag_foundation/buoi_05/.venv/bin/python -m pip install -r rag_foundation/buoi_07/requirements.txt
```

### Sai interpreter

Nếu import lỗi dù đã cài package, kiểm tra đang dùng đúng interpreter Buổi 05:

```bash
rag_foundation/buoi_05/.venv/bin/python --version
```

### Thiếu API key

Nếu `status` báo API key `Thiếu`, điền `GEMINI_API_KEY` trong:

```text
rag_foundation/buoi_07/.env
```

Không dán API key vào chat/log.

### Collection rỗng hoặc chưa tồn tại

Chạy index trước:

```bash
rag_foundation/buoi_05/.venv/bin/python rag_foundation/buoi_07/rag.py index --strategy hierarchical
```

### Model/dimension mismatch

Nếu đổi `GEMINI_EMBEDDING_MODEL` hoặc `GEMINI_EMBEDDING_DIM`, cần index lại collection tương ứng. Không query nhầm collection cũ.

### JSON lỗi

Chạy validate để xem file/record lỗi:

```bash
rag_foundation/buoi_05/.venv/bin/python rag_foundation/buoi_07/rag.py validate --strategy hierarchical
```

### Embedding lỗi hoặc rate limit

Nếu Gemini trả `429`/`RESOURCE_EXHAUSTED`, code sẽ in log tiến độ, đợi 60 giây rồi thử lại tối đa vài lần. Trong lúc index, terminal sẽ hiển thị các dòng như `Embedding 10/293: ...`, `Gemini 429/quota...`, `Upsert ...` để theo dõi tiến độ. Nếu vẫn lỗi sau các lần retry, index sẽ dừng, không upsert một phần. Khi đó hãy thử lại sau, giảm số dữ liệu, hoặc kiểm tra quota/API key. Không dùng vector giả để thay thế.

## 20. Giới hạn của demo

- Không có reranker.
- Không có hybrid search.
- Không có OCR/PDF parsing ở Buổi 07.
- Không có RBAC, deployment hoặc monitoring production.
- Threshold cần hiệu chỉnh theo dữ liệu thật.
- Retrieval có thể bỏ sót thông tin hoặc false positive.
- Generation phụ thuộc evidence retrieve được và prompt grounding.

## 21. Cảnh báo quan trọng

- Đây không phải tư vấn pháp lý hoặc tài chính.
- `RAG_MAX_DISTANCE` là ngưỡng demo, không phải độ tin cậy tuyệt đối.
- Retrieval có thể bỏ sót thông tin hoặc retrieve nhầm thông tin gần nghĩa.
- Nội dung chunk được gửi tới Gemini khi embedding/generation; chỉ dùng dữ liệu mà người vận hành được phép gửi tới dịch vụ bên ngoài.

## 22. Manual test plan

Sau khi đã index dữ liệu thật, thử ba câu hỏi sau bằng CLI hoặc Streamlit.

### A. Có khả năng thuộc tài liệu

```text
Cơ cấu lại thời hạn trả nợ được quy định như thế nào?
```

Kỳ vọng: nếu dữ liệu thật và threshold phù hợp, hệ thống có thể retrieve evidence liên quan và tạo answer có citation. Không khẳng định chắc chắn có answer; phải xem evidence thực tế.

### B. Có khả năng thuộc tài liệu

```text
Việc phân loại nợ và trích lập dự phòng được thực hiện như thế nào?
```

Kỳ vọng: nếu dữ liệu thật và threshold phù hợp, hệ thống có thể retrieve evidence liên quan và tạo answer có citation. Không dùng kết quả như tư vấn pháp lý.

### C. Ngoài phạm vi

```text
Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?
```

Kỳ vọng mong muốn:

- evidence không đạt threshold thì không gọi generation
- trả: `Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp.`
- không bịa tên ngân hàng hoặc lãi suất

Đây không phải kết quả được bảo đảm trước khi hiệu chỉnh threshold. Nếu câu C vẫn đạt threshold, hãy ghi nhận là false positive của retrieval/gate, không đánh dấu PASS giả và không sửa câu trả lời thủ công để che lỗi.
