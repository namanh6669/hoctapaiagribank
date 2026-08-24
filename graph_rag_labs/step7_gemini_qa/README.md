# Bước 7 — Multi-hop + Gemini QA (End-to-end RAG)

Lắp ghép các bước trước thành pipeline hoàn chỉnh:

```
câu hỏi (vi)
  ↓ embed (thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5, CPU)
  ↓ vector search (kbhops_chunk_embedding, top-k)
  ↓ Multi-hop expansion (CAN_CU | THAY_THE | HOP_NHAT, 1..N hops)
  ↓
Context gồm:
  - chunks vector (body text đầy đủ từ chunk + children)
  - chunks hop:N (từ Document liên quan)
  ↓
Prompt: system (schema + VN legal structure) + user (query + context)
  ↓
Gemini API (gemini-flash-latest, with retry on 429/503)
  ↓
Answer + citations [N]
```

## Cấu hình

Thêm `GEMINI_API_KEY` vào `graph_rag_labs/.env.graph_rag`:

```bash
GEMINI_API_KEY=AIzaSyD4mLsmHxlncziZvLr1JEuMCnVVqVCB5Zk
GEMINI_MODEL=gemini-flash-latest
```

> Free tier: 20 request/ngày. Khi vượt quota, `GeminiClient` retry
> với backoff 2/4/8/16/32 s trước khi raise.

## Cài thêm

```bash
python3 -m pip install google-genai
```

## Chạy

```bash
cd graph_rag_labs/step7_gemini_qa
python -m src.run_demo
```

## System Prompt (cốt lõi)

`src/prompt_builder.py` xây dựng 3 phần:

1. **Schema description** — node Document / Chunk + 6 loại edge
   (PART_OF, PARENT_OF, NEXT, CAN_CU, THAY_THE, HOP_NHAT) với giải
   thích properties.

2. **Vietnamese legal structure** — Chương / Mục / Điều / Khoản / Điểm
   + heading_pattern.

3. **Quy tắc trả lời**:
   - **ĐỌC KỸ** `text` của từng chunk (title chỉ là tiêu đề)
   - Trả lời tiếng Việt, ngắn gọn, có cấu trúc
   - Trích dẫn `[N]` cho mỗi phát biểu
   - Nếu **ngữ cảnh không đủ thông tin** → nói rõ "Ngữ cảnh không có
     thông tin về …" — KHÔNG tự suy đoán, KHÔNG dùng kiến thức ngoài
   - Với văn bản liên quan (đi qua CAN_CU / THAY_THE) → ghi rõ "văn
     bản liên quan: …" thay vì khẳng định trực tiếp

## Demo console output

```
Q: Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ là gì?
Context: 6 doc(s), 4 chunk(s)
  [vector] score=+0.948 [article] len(text)=2181 'Tổ chức tín dụng...'
  [vector] score=+0.942 [article] len(text)=1836 'Thời gian cơ cấu lại...'
  [vector] score=+0.938 [article] len(text)=898 'a) Căn cứ quy định...'
  [vector] score=+0.929 [paragraph] len(text)=57 'Theo đề nghị của Vụ trưởng...'

Gemini (gemini-flash-latest) — 3902 ms, in=1629 out=103 tokens
---
| Dựa vào ngữ cảnh được cung cấp, ngữ cảnh chỉ chứa các tiêu đề điều khoản...
```

> Khi Gemini có đầy đủ text, nó sẽ trích dẫn điều khoản + khoản cụ
> thể. Khi ngữ cảnh không match (như Q3 về "Căn cứ pháp lý" — preamble
> không nằm trong top-k), model sẽ nói rõ "không có thông tin" thay vì
> bịa.

## Cấu trúc thư mục

```
step7_gemini_qa/
├── README.md
├── output/
│   └── qa_results.json     ← mỗi query: context, answer, tokens
└── src/
    ├── __init__.py
    ├── config.py            ← load .env.graph_rag
    ├── gemini_client.py     ← wrapper google-genai + retry
    ├── prompt_builder.py    ← system + user prompt
    └── run_demo.py          ← entrypoint end-to-end
```

## Cải tiến đáng chú ý

1. **Container-children fallback** (`step6_multi_hop/retriever.py`):
   `MATCH OPTIONAL (c)-[:PARENT_OF]->(child)` lấy text của các paragraph
   con khi chunk container (chapter/article) có `text` rỗng. Đảm bảo
   Gemini thấy body thực, không chỉ title.

2. **Retry với exponential backoff** cho 429 / 503 / 500 trong
   `GeminiClient.generate()`. Mặc định 4 retries (2/4/8/16/32 s).

3. **Max output tokens = 2000** đủ chứa câu trả lời dài kèm citations.

## Mở rộng

* Streaming response (`stream=True` trong `generate_content_stream`).
* Citation đi kèm offset (chunk_id + heading_path) cho UI.
* Re-rank: dùng cross-encoder (BAAI/bge-reranker-v2-m3) trước khi feed
  vào Gemini để tăng precision.
* RLHF: lưu feedback "chunk này hữu ích" để tune retriever.