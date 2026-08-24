# Buổi 14 — Evaluation Report

_Sinh tự động bằng `scripts/compare_retrieval.py`, top_k=5._

- Số câu hỏi: **8** (EXACT_KEYWORD, MIXED, SEMANTIC)
- Methods so sánh: bm25, dense, hybrid, rerank
- Corpus: `data/processed/chunks_normalized.csv` (chunks=1463)

- Rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3` (is_fallback=False)


## Metric tổng hợp

| Method | Hit@1 | Hit@3 | Hit@5 | MRR | n |
|---|---:|---:|---:|---:|---:|
| BM25-only | 0.12 | 0.50 | 0.62 | 0.30 | 8 |
| Dense-only | 0.50 | 0.88 | 0.88 | 0.65 | 8 |
| Hybrid (RRF) | 0.25 | 0.38 | 0.75 | 0.39 | 8 |
| Hybrid + Rerank (BGE-reranker-v2-m3) | 0.62 | 0.75 | 1.00 | 0.74 | 8 |

## Metric theo query_type

| Method | EXACT_KEYWORD | SEMANTIC | MIXED |
|---|---:|---:|---:|
| BM25-only | H@5=1.00 MRR=0.34 (n=3) | H@5=0.00 MRR=0.00 (n=3) | H@5=1.00 MRR=0.67 (n=2) |
| Dense-only | H@5=0.67 MRR=0.50 (n=3) | H@5=1.00 MRR=0.78 (n=3) | H@5=1.00 MRR=0.67 (n=2) |
| Hybrid (RRF) | H@5=0.67 MRR=0.42 (n=3) | H@5=0.67 MRR=0.19 (n=3) | H@5=1.00 MRR=0.62 (n=2) |
| Hybrid + Rerank (BGE-reranker-v2-m3) | H@5=1.00 MRR=1.00 (n=3) | H@5=1.00 MRR=0.58 (n=3) | H@5=1.00 MRR=0.60 (n=2) |

**Cột trên 1 method 1 dòng trống — bảng dưới tách riêng từng metric để dễ đọc:**


### EXACT_KEYWORD

| Method | Hit@1 | Hit@3 | Hit@5 | MRR | n |
|---|---:|---:|---:|---:|---:|
| BM25-only | 0.00 | 0.67 | 1.00 | 0.34 | 3 |
| Dense-only | 0.33 | 0.67 | 0.67 | 0.50 | 3 |
| Hybrid (RRF) | 0.33 | 0.33 | 0.67 | 0.42 | 3 |
| Hybrid + Rerank (BGE-reranker-v2-m3) | 1.00 | 1.00 | 1.00 | 1.00 | 3 |

### SEMANTIC

| Method | Hit@1 | Hit@3 | Hit@5 | MRR | n |
|---|---:|---:|---:|---:|---:|
| BM25-only | 0.00 | 0.00 | 0.00 | 0.00 | 3 |
| Dense-only | 0.67 | 1.00 | 1.00 | 0.78 | 3 |
| Hybrid (RRF) | 0.00 | 0.33 | 0.67 | 0.19 | 3 |
| Hybrid + Rerank (BGE-reranker-v2-m3) | 0.33 | 0.67 | 1.00 | 0.58 | 3 |

### MIXED

| Method | Hit@1 | Hit@3 | Hit@5 | MRR | n |
|---|---:|---:|---:|---:|---:|
| BM25-only | 0.50 | 1.00 | 1.00 | 0.67 | 2 |
| Dense-only | 0.50 | 1.00 | 1.00 | 0.67 | 2 |
| Hybrid (RRF) | 0.50 | 0.50 | 1.00 | 0.62 | 2 |
| Hybrid + Rerank (BGE-reranker-v2-m3) | 0.50 | 0.50 | 1.00 | 0.60 | 2 |

## Per-question (rank của gold trong top-5)

| question_id | type | bm25 | dense | hybrid | rerank |
|---|---|---:|---:|---:|---:|
| Q01 | EXACT_KEYWORD | 3 | — | — | 1 |
| Q02 | EXACT_KEYWORD | 5 | 2 | 4 | 1 |
| Q03 | EXACT_KEYWORD | 2 | 1 | 1 | 1 |
| Q04 | SEMANTIC | — | 1 | — | 4 |
| Q05 | SEMANTIC | — | 1 | 4 | 1 |
| Q06 | SEMANTIC | — | 3 | 3 | 2 |
| Q07 | MIXED | 3 | 3 | 4 | 5 |
| Q08 | MIXED | 1 | 1 | 1 | 1 |

### BM25 mạnh ở đâu?

- EXACT_KEYWORD : Hit@5 = 1.00 · MRR = 0.34 (n=3)
- SEMANTIC      : Hit@5 = 0.00 · MRR = 0.00 (n=3)
- MIXED         : Hit@5 = 1.00 · MRR = 0.67 (n=2)

### Dense mạnh ở đâu?

- EXACT_KEYWORD : Hit@5 = 0.67 · MRR = 0.50 (n=3)
- SEMANTIC      : Hit@5 = 1.00 · MRR = 0.78 (n=3)
- MIXED         : Hit@5 = 1.00 · MRR = 0.67 (n=2)

### Hybrid (BM25 + Dense) có giúp không?

So sánh metric Hybrid với max(BM25, Dense) điểm MRR theo từng câu:
- Hybrid tốt hơn best-of-single: 0/8
- Hybrid ngang best-of-single : 3/8
- Hybrid thua best-of-single  : 5/8

### Rerank có đổi ranking không?

- Rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3`
- Số câu mà rerank đổi vị trí gold (so với hybrid): 6/8
- Số câu mà rerank thay đổi top-1 (so với hybrid)   : 5/8

### Failure cases

- Câu mà cả 4 method đều không tìm thấy gold trong top-5: 0/8

Chi tiết từng câu:

- `Q01` (EXACT_KEYWORD): bm25=3, dense=—, hybrid=—, rerank=1
- `Q02` (EXACT_KEYWORD): bm25=5, dense=2, hybrid=4, rerank=1
- `Q03` (EXACT_KEYWORD): bm25=2, dense=1, hybrid=1, rerank=1
- `Q04` (SEMANTIC): bm25=—, dense=1, hybrid=—, rerank=4
- `Q05` (SEMANTIC): bm25=—, dense=1, hybrid=4, rerank=1
- `Q06` (SEMANTIC): bm25=—, dense=3, hybrid=3, rerank=2
- `Q07` (MIXED): bm25=3, dense=3, hybrid=4, rerank=5
- `Q08` (MIXED): bm25=1, dense=1, hybrid=1, rerank=1

### Lỗi trong quá trình chạy

- Không có lỗi nào.

## Kết luận có giới hạn

- Bộ câu hỏi chỉ có **8 câu** — sai số thống kê lớn. Không nên kết luận quá mạnh.
- Gold là chunk cụ thể, không phải toàn bộ document liên quan — một số câu có nhiều chunk đúng (ví dụ Q03 có thể ghi nhận preamble 41/2016 hoặc các chunk khác cùng doc). Đánh giá này đo *chunk-level exact hit*, không đo *document-level recall*.
- Không đo latency/throughput; chỉ đo chất lượng ranking.
- Reranker chỉ rerank top-candidate_k của Hybrid — nếu Hybrid miss, Rerank cũng miss (không thể phục hồi).
- Văn bản pháp luật VN dài (max chunk ~58 KB); truncate ở 512 token khi encode dense/rerank có thể bỏ phần đầu/cuối chunk.
- Rerank nặng (Cross-Encoder ~2.3 GB), nếu OOM hoặc torch lỗi sẽ tự động rơi vào FALLBACK — phần rerank trong bảng trên sẽ không phản ánh đúng neural rerank, mà chỉ là hybrid ordering.
