# Buổi 08 - Advanced RAG

Buổi 08 xây trên semantic baseline của Buổi 07 để minh họa **Hybrid Search + Reranking + Comparison** cho workshop Advanced RAG. Đây là code học tập, dễ kiểm thử, không phải hệ thống tư vấn pháp lý.

## 1. Mục tiêu và khác biệt Buổi 07/08

- **Buổi 07**: semantic RAG cơ bản — Gemini embedding, Chroma, confidence gate bằng cosine distance, generation và citation.
- **Buổi 08**: thêm tầng nâng cao để người học nhìn thấy nhiều cách retrieve cùng một câu hỏi:
  - BM25 lexical retrieval.
  - Semantic retrieval từ baseline Buổi 07.
  - Reciprocal Rank Fusion (RRF) để hợp nhất rank BM25/semantic.
  - Cross-encoder reranker multilingual cho tập candidate nhỏ.
  - CLI/UI comparison để so sánh rank movement.
  - Offline evaluator cho Recall@K, MRR@K, nDCG@K.

Không dùng raw BM25 score và cosine distance để cộng tùy tiện vì hai thang đo khác nhau; Buổi 08 dùng **rank** qua RRF.

## 2. Sơ đồ pipeline

```text
Question
   │
   ├── BM25 tokenizer + BM25Okapi ─────┐
   │                                    ├── Union by chunk_id ── RRF ── Top fused candidates ── Reranker ── Final evidence
   └── Gemini query embedding + Chroma ┘
                                                                  │
                                                                  └── Grounded answer + real metadata citations
```

Các mode hỗ trợ:

- `bm25`
- `semantic`
- `hybrid`
- `hybrid_rerank` — mặc định cho Advanced RAG answer.

## 3. Cấu trúc project

```text
rag_advanced/buoi_08/
├── SPEC_buoi_08.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── rag.py                  # semantic baseline copy từ Buổi 07
├── advanced_rag.py         # BM25, semantic candidates, RRF, rerank, answer pipeline, CLI
├── evaluate.py             # offline retrieval evaluator
├── app.py                  # Streamlit comparison UI
├── eval/questions.json     # starter labels, needs_human_review=true
├── tests/
├── reports/
└── storage/
```

## 4. Setup `.venv`, requirements và `.env`

Dùng lại interpreter Buổi 05:

```bash
cd /Users/nana/Documents/GitHub/hoctapaiagribank/RAG
rag_foundation/buoi_05/.venv/bin/python -m pip install -r rag_advanced/buoi_08/requirements.txt
```

Tạo `.env` khi muốn chạy thật:

```bash
cp rag_advanced/buoi_08/.env.example rag_advanced/buoi_08/.env
```

Điền `GEMINI_API_KEY` thật trong `.env` nếu muốn chạy semantic prepare/query thật. Không commit key thật.

## 5. Cảnh báo tài nguyên reranker

Default reranker:

```text
BAAI/bge-reranker-v2-m3
```

Model này có thể lớn và cần Internet/disk/RAM trong lần tải đầu. Cache đặt tại:

```text
rag_advanced/buoi_08/storage/huggingface/
```

- `RERANK_DEVICE=auto`: dùng CUDA nếu khả dụng, ngược lại CPU.
- `RERANK_DEVICE=cpu`: ép CPU, có thể chậm.
- `RERANK_DEVICE=cuda`: fail rõ nếu CUDA không khả dụng.
- Không bật `trust_remote_code=True`.
- App/status/import không tự tải model; chỉ command/mode rerank khi người dùng chủ động mới có thể tải.

## 6. Lệnh vận hành chính

Tất cả dùng interpreter Buổi 05:

```bash
PY=rag_foundation/buoi_05/.venv/bin/python
```

### Status read-only

```bash
$PY rag_advanced/buoi_08/advanced_rag.py status --strategy hierarchical
```

Không tạo collection, không gọi Gemini, không tải reranker.

### Prepare semantic index

```bash
$PY rag_advanced/buoi_08/advanced_rag.py prepare-semantic --strategy hierarchical
```

Chỉ chạy khi có `GEMINI_API_KEY`. Dùng Chroma trong `rag_advanced/buoi_08/storage/chroma/`, không dùng storage Buổi 07.

### BM25 diagnostic

```bash
$PY rag_advanced/buoi_08/advanced_rag.py bm25 --strategy hierarchical --question "Điều 7 quy định gì?"
```

### Hybrid RRF diagnostic

```bash
$PY rag_advanced/buoi_08/advanced_rag.py hybrid --strategy hierarchical --question "Điều 7 quy định gì?"
```

Cần semantic index đã prepare.

### Hybrid + rerank diagnostic

```bash
$PY rag_advanced/buoi_08/advanced_rag.py rerank --strategy hierarchical --question "Điều 7 quy định gì?"
```

Có thể tải reranker model nếu cache chưa có.

### Query grounded answer

```bash
$PY rag_advanced/buoi_08/advanced_rag.py query --mode hybrid_rerank --strategy hierarchical --question "Điều 7 quy định gì?"
```

`query` là command duy nhất trong nhóm này gọi generation, và chỉ gọi tối đa một lần sau khi có accepted evidence.

### Compare retrieval modes

```bash
$PY rag_advanced/buoi_08/advanced_rag.py compare --strategy hierarchical --question "Điều 7 quy định gì?"
```

Compare chạy retrieval/rerank bốn mode nhưng không gọi generation.

## 7. Test, evaluate và Streamlit

### Unit test

```bash
cd rag_advanced/buoi_08
../buoi_05/.venv/bin/python -m unittest discover -s tests -v
```

Hoặc từ workspace root:

```bash
rag_foundation/buoi_05/.venv/bin/python -m unittest discover -s rag_advanced/buoi_08/tests -v
```

### Evaluate offline

Real retrieval evaluator:

```bash
$PY rag_advanced/buoi_08/evaluate.py --strategy hierarchical --k 5
```

Synthetic/mock evaluator cho test offline, không cần Gemini/Chroma/reranker:

```bash
$PY rag_advanced/buoi_08/evaluate.py --mock-synthetic --strategy hierarchical --k 5
```

Report lưu trong `rag_advanced/buoi_08/reports/`.

### Streamlit UI

```bash
cd /Users/nana/Documents/GitHub/hoctapaiagribank/RAG
$PY -m streamlit run rag_advanced/buoi_08/app.py
```

UI có bốn tab:

1. `Hỏi đáp Advanced RAG`
2. `So sánh Retrieval`
3. `Pipeline Trace`
4. `Đánh giá`

UI không tự index, không tự tải model và không tự chạy evaluation hàng loạt khi mở trang.

## 8. Giải thích score/rank

- **BM25 score**: cao hơn thường tốt hơn trong nhánh lexical, không phải xác suất.
- **Cosine distance**: thấp hơn tốt hơn trong semantic retrieval.
- **RRF score**: cao hơn tốt hơn, tính từ rank BM25/semantic; không dùng raw score/distance để cộng.
- **Rerank score**: `sigmoid(logit)` của cross-encoder, nằm trong `[0,1]`, **không phải xác suất câu trả lời đúng**.

## 9. Candidate K và Final K

- `BM25_CANDIDATES`: số candidate lấy từ BM25.
- `SEMANTIC_CANDIDATES`: số candidate lấy từ semantic retrieval.
- `RERANK_CANDIDATES`: số fused candidates tối đa đưa vào reranker.
- `FINAL_TOP_K`: số evidence cuối sau rerank/gating.

Nếu union ít hơn `RERANK_CANDIDATES`, hệ thống dùng `min(RERANK_CANDIDATES, union_count)`, không coi đó là lỗi cấu hình.

## 10. Evaluation metrics và giới hạn gold labels

Evaluator tính:

- Recall@K
- MRR@K
- nDCG@K với binary relevance
- latency mean và p50

`eval/questions.json` là starter set. Tất cả label ban đầu có `needs_human_review=true`, nên report phải cảnh báo và không tuyên bố mode thắng chính thức khi gold labels chưa được duyệt.

## 11. Manual comparison questions

Không khẳng định trước mode nào thắng; dùng ranking/metrics thực tế.

A. Exact legal reference:

```text
Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?
```

B. Paraphrase semantic:

```text
Khách hàng gặp khó khăn có thể được điều chỉnh kỳ hạn trả nợ ra sao?
```

C. Multi-concept:

```text
Phân loại nợ và trích lập dự phòng được thực hiện như thế nào?
```

D. Out-of-scope:

```text
Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?
```

## 12. Troubleshooting

### Thiếu semantic index

Chạy:

```bash
$PY rag_advanced/buoi_08/advanced_rag.py prepare-semantic --strategy hierarchical
```

Nếu thiếu API key, kiểm tra `.env` và biến `GEMINI_API_KEY`.

### Model download lỗi

- Kiểm tra Internet.
- Kiểm tra dung lượng trong `storage/huggingface/`.
- Chạy lại command `rerank` khi chủ động tải model.
- Nếu máy yếu, đặt `RERANK_DEVICE=cpu` nhưng chấp nhận chậm.

### CPU chậm / thiếu RAM

- Giảm `RERANK_CANDIDATES`.
- Giảm `RERANK_BATCH_SIZE`.
- Giảm `RERANKER_MAX_LENGTH` nhưng không dưới 64.

### API/model lỗi

- `generation_error` → hệ thống trả `retrieval_only` và vẫn giữ evidence.
- `reranker_unavailable` → không silent fallback sang RRF như thể rerank đã thành công.
- Status command không gọi Gemini/reranker nên dùng để kiểm tra an toàn trước.

## 13. Không phải tư vấn pháp lý

Project này phục vụ workshop kỹ thuật RAG. Fixture, starter gold labels và output không phải tư vấn pháp lý, không thay thế chuyên gia pháp lý và không nên dùng để ra quyết định nghiệp vụ thật nếu chưa được kiểm định độc lập.
