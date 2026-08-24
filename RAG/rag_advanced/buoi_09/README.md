# Buổi 09 - Multi-query & Parent–Child Retrieval

Buổi 09 mở rộng Advanced RAG của Buổi 08 theo hai hướng: **query fan-out có kiểm soát** và **retrieve child, return parent**. Đây là code workshop kỹ thuật, dễ kiểm thử, không phải hệ thống tư vấn pháp lý.

## 1. Mục tiêu và khác biệt Buổi 08/09

### Buổi 08

- BM25 lexical retrieval.
- Semantic retrieval với Gemini embedding + Chroma.
- Inner Reciprocal Rank Fusion (RRF) giữa BM25/semantic.
- Cross-encoder rerank child chunks.
- Grounded answer và citation theo metadata thật.
- Compare/evaluate baseline Advanced RAG.

### Buổi 09

- Giữ snapshot độc lập của Buổi 08 trong `rag.py` và `advanced_rag.py`.
- Thêm query expansion: Q0 gốc + Q1..Qn generated variants.
- Chạy hybrid retrieval cho từng query, rồi hợp nhất child bằng cross-query RRF.
- Map child hits sang parent documents từ hierarchy store.
- Aggregate/rerank parent, build context theo budget và sinh answer với citation `[P1]`.
- So sánh bốn mode để thấy trade-off recall, precision, latency, context size.

## 2. Pipeline hai tầng fusion và parent expansion

```text
Original question Q0
  ├─ Query generator → Q1..Qn
  │
  ├─ Per-query hybrid retrieval
  │    BM25 rank ┐
  │              ├─ Inner RRF per query → child fused rank
  │    Semantic ┘
  │
  ├─ Cross-query RRF → fused child hits
  │
  ├─ Child-to-parent lookup từ hierarchy store
  │
  ├─ Parent aggregation bằng top child ranks
  │
  ├─ Parent rerank bằng cross-encoder pair (Q0, parent_text)
  │
  └─ Evidence gate → grounded answer + citations
```

Không vector-search trực tiếp trên parent trong Buổi 09. Parent store là source of truth.

## 3. Bốn mode comparison

| Mode | Retrieval | Rerank | Evidence |
|---|---|---|---|
| `single_flat` | Q0 → hybrid child | child rerank bằng Q0 | child `[E1]` |
| `multi_flat` | Q0 + variants → per-query hybrid → MQ-RRF | child rerank bằng Q0 | child `[E1]` |
| `single_parent` | Q0 → hybrid child → parent aggregate | parent rerank bằng Q0 | parent `[P1]` |
| `multi_parent` | Q0 + variants → MQ child → parent aggregate | parent rerank bằng Q0 | parent `[P1]` |

`compare` chạy retrieval/rerank cho bốn mode nhưng không gọi answer generation.

## 4. Cấu trúc project và setup `.env`

```text
rag_advanced/buoi_09/
├── .env.example
├── requirements.txt
├── rag.py
├── advanced_rag.py
├── hierarchical_rag.py
├── evaluate.py
├── app.py
├── eval/questions.json
├── reports/
├── storage/chroma/
├── storage/hierarchy/
├── storage/huggingface/
└── tests/
```

Setup:

```bash
cd /Users/nana/Documents/GitHub/hoctapaiagribank/RAG
PY=rag_foundation/buoi_05/.venv/bin/python
$PY -m pip install -r rag_advanced/buoi_09/requirements.txt
cp rag_advanced/buoi_09/.env.example rag_advanced/buoi_09/.env
```

Điền `GEMINI_API_KEY` thật trong `.env` chỉ khi chủ động chạy semantic/query generation/answer generation. Không commit key thật.

## 5. Build hierarchy và warning/ambiguous

Status read-only:

```bash
$PY rag_advanced/buoi_09/hierarchical_rag.py hierarchy-status
```

Audit read-only:

```bash
$PY rag_advanced/buoi_09/hierarchical_rag.py hierarchy-audit --input rag_foundation/buoi_05/output/chunks
```

Build store:

```bash
$PY rag_advanced/buoi_09/hierarchical_rag.py build-hierarchy --input rag_foundation/buoi_05/output/chunks
```

Hierarchy resolution precedence:

1. `structure` metadata nếu có và nhất quán.
2. Heading pháp lý ở đầu chunk (`Điều`, `Chương`).
3. Carry-forward theo source-local order.
4. Document fallback.

Warnings thường gặp:

- `missing_structure_metadata`: chunk không có metadata structure dùng được.
- `heading_detected_from_text`: suy ra parent từ heading text.
- `inline_article_reference_ignored`: nhắc Điều trong thân đoạn, không coi là heading.
- `metadata_heading_conflict`: metadata và heading mâu thuẫn.
- `contains_ambiguous_children`: parent chứa child ambiguous.
- `oversized_single_child`: một child đã vượt `PARENT_MAX_CHARS`.

Parent-aware query yêu cầu store sẵn sàng. Store thiếu/stale trả `hierarchy_not_ready`, không tự build.

## 6. Query expansion contract và API call budget

`expand-query` giữ Q0 nguyên văn sau trim/NFC và chỉ nhờ Gemini sinh Q1..Qn:

```bash
$PY rag_advanced/buoi_09/hierarchical_rag.py expand-query --question "Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?"
```

Contract:

- Q0: `origin=original`, `focus=original_intent`.
- Q1..Qn: `origin=generated`, focus một trong `exact_legal_terms`, `paraphrase`, `missing_aspect`.
- Strict JSON schema từ model: `{"queries":[{"text":"...","focus":"..."}]}`.
- Deduplicate bằng NFC/casefold/whitespace-punctuation normalization.
- Không bịa số Điều/Khoản/Điểm/năm chắc chắn không có trong câu hỏi.
- Nếu có legal reference, ít nhất một variant phải giữ reference.
- Cache chỉ trong process/session; không ghi prompt/query người dùng xuống disk mặc định.

API call budget cho `multi_parent` hoàn chỉnh:

1. Một Gemini Generation call cho query variants.
2. Một Gemini Generation call cho answer nếu evidence đạt gate.

Gemini Embedding calls cho semantic retrieval được trace riêng, không tính vào giới hạn hai Generation calls.

## 7. Công thức score

### Inner RRF per query

Dùng trong snapshot Buổi 09 `advanced_rag.py`:

```text
inner_rrf_score(d) = BM25_WEIGHT / (RRF_K + bm25_rank(d))
                   + SEMANTIC_WEIGHT / (RRF_K + semantic_rank(d))
```

Không cộng raw BM25 score và cosine distance trực tiếp.

### Cross-query RRF

```text
multi_query_rrf_score(d) = Σ query_weight(q) / (MULTI_QUERY_RRF_K + inner_fused_rank_q(d))
```

- Q0 weight: `MULTI_QUERY_ORIGINAL_WEIGHT`.
- Generated query weight: `MULTI_QUERY_VARIANT_WEIGHT`.
- Candidate chỉ xuất hiện ở một query vẫn được giữ.
- Không dùng raw inner RRF score/rerank score trong công thức cross-query.

### Parent aggregation

```text
parent_rrf_score(p) = Σ 1 / (PARENT_RRF_K + multi_query_rank(child))
```

Chỉ top `PARENT_SCORE_CHILD_LIMIT` child tốt nhất theo `multi_query_rank` tham gia tính điểm. Giữ rõ:

- `scoring_child_ids`: child dùng tính parent score.
- `supporting_child_ids`: toàn bộ child hits map vào parent.
- `anchor_child_id`: child tốt nhất.

## 8. Child retrieval, parent return và parent rerank

Multi-child command:

```bash
$PY rag_advanced/buoi_09/hierarchical_rag.py multi-child --question "Điều kiện vay vốn và các trường hợp không được cho vay là gì?"
```

Parent retrieval command:

```bash
$PY rag_advanced/buoi_09/hierarchical_rag.py parent-retrieve --mode multi_parent --question "Điều kiện vay vốn và các trường hợp không được cho vay là gì?"
```

Answer pipeline command:

```bash
$PY rag_advanced/buoi_09/hierarchical_rag.py query --mode multi_parent --question "Điều kiện vay vốn và các nhu cầu vốn không được cho vay được quy định thế nào?"
```

Parent rerank:

- Cross-encoder pair là `(original_question, parent_text)`.
- Không rerank bằng generated query.
- Lazy-load/cache reranker theo contract Buổi 08.
- Score hiển thị là `sigmoid(logit)`, không phải xác suất đúng.
- Parent evidence gate: `parent_rerank_score >= RERANK_MIN_SCORE`.

Citation parent object gồm `evidence_id`, `parent_id`, `anchor_child_id`, `supporting_child_ids`, `source`, `page_start/page_end`, `structural_path`, `parent_rerank_score`, `ambiguous`, `warnings`.

## 9. Lệnh vận hành chính

```bash
# status/audit/build hierarchy
$PY rag_advanced/buoi_09/hierarchical_rag.py hierarchy-status
$PY rag_advanced/buoi_09/hierarchical_rag.py hierarchy-audit --input rag_foundation/buoi_05/output/chunks
$PY rag_advanced/buoi_09/hierarchical_rag.py build-hierarchy --input rag_foundation/buoi_05/output/chunks

# prepare semantic index khi có API key
$PY rag_advanced/buoi_09/advanced_rag.py prepare-semantic --strategy hierarchical

# query expansion / child fusion / parent retrieval
$PY rag_advanced/buoi_09/hierarchical_rag.py expand-query --question "..."
$PY rag_advanced/buoi_09/hierarchical_rag.py multi-child --question "..."
$PY rag_advanced/buoi_09/hierarchical_rag.py parent-retrieve --mode multi_parent --question "..."

# answer và compare
$PY rag_advanced/buoi_09/hierarchical_rag.py query --mode multi_parent --question "..."
$PY rag_advanced/buoi_09/hierarchical_rag.py compare --question "..."

# evaluate retrieval-only
$PY rag_advanced/buoi_09/evaluate.py --mock-synthetic --k 5
$PY rag_advanced/buoi_09/evaluate.py --k 5

# Streamlit
$PY -m streamlit run rag_advanced/buoi_09/app.py
```

## 10. Candidate K, parent K và context budget

- `BM25_CANDIDATES`, `SEMANTIC_CANDIDATES`: candidate mỗi branch trong inner hybrid.
- `PER_QUERY_CANDIDATES`: child hits giữ lại mỗi query sau inner RRF.
- `PARENT_SCORE_CHILD_LIMIT`: child tốt nhất dùng tính parent score.
- `PARENT_CANDIDATES`: số parent giữ trước parent rerank.
- `FINAL_PARENT_TOP_K`: số parent giữ sau rerank.
- `PARENT_MAX_CHARS`: parent builder split parent windows, không cắt giữa child.
- `TOTAL_CONTEXT_MAX_CHARS`: context budget tổng; chỉ thêm nguyên parent. Nếu parent đầu tiên vượt budget, vẫn giữ và warning.

## 11. Evaluation metrics và giới hạn gold labels

`eval/questions.json` chứa starter labels:

- `question_type`: `exact`, `paraphrase`, `multi_aspect`, `hierarchy_context`, `out_of_scope`.
- `relevant_child_ids`.
- `relevant_parent_ids` resolve từ hierarchy store hiện tại.
- `needs_human_review=true` cho toàn bộ starter labels.

Evaluator tính retrieval-only:

- Child Recall@K.
- Parent Recall@K.
- MRR@K.
- nDCG@K binary relevance.
- unique relevant parents/sources retrieved.
- query count, child union count.
- context chars, expansion factor.
- mean/p50 latency.
- query-generation call count và embedding call count riêng.

Report JSON ghi atomically vào `reports/`, rồi cập nhật `latest_report.json` sau khi report hợp lệ. Không tuyên bố `multi_parent` thắng nếu labels còn `needs_human_review=true`.

## 12. Troubleshooting

### `hierarchy_not_ready`

- Chạy `hierarchy-status` để xem thiếu/stale.
- Nếu thiếu store, chạy `build-hierarchy` bằng command/action riêng.
- Nếu stale config/fingerprint, rebuild hierarchy từ input hiện tại.

### `collection_not_ready` hoặc semantic lỗi

- Chạy `advanced_rag.py prepare-semantic --strategy hierarchical` khi có `GEMINI_API_KEY`.
- Kiểm tra Chroma storage trong `rag_advanced/buoi_09/storage/chroma/`.

### `query_generation_unavailable`

- Kiểm tra `GEMINI_API_KEY` và schema JSON trả về.
- Single modes vẫn chạy được nếu không cần query variants.

### `reranker_unavailable`

- Reranker có thể cần Internet/disk/RAM lần tải đầu.
- Cache: `rag_advanced/buoi_09/storage/huggingface/`.
- Có thể đặt `RERANK_DEVICE=cpu` nếu máy không có CUDA.

### Latency/context lớn

- Giảm `MULTI_QUERY_COUNT`, `PER_QUERY_CANDIDATES`, `PARENT_CANDIDATES`.
- Giảm `TOTAL_CONTEXT_MAX_CHARS` để quan sát budget behavior.
- Rerank parent chỉ chạy sau action query/compare, không chạy khi import/status.

## 13. Không phải tư vấn pháp lý

Buổi 09 phục vụ workshop kỹ thuật RAG. Dữ liệu, starter labels, retrieval result và generated answer không phải tư vấn pháp lý, không thay thế chuyên gia pháp lý và không nên dùng để ra quyết định nghiệp vụ thật nếu chưa được kiểm định độc lập.
