# Bước 6 — Multi-hop Vector + Graph Retrieval

Kết hợp **tìm kiếm vector** (Bước 2) với **mở rộng đa bước (multi-hop)**
trên đồ thị `kb-hops` (Bước 4) — cho phép lấy thêm ngữ cảnh từ các tài
liệu luật có liên quan qua các quan hệ `CAN_CU`, `THAY_THE`, `HOP_NHAT`.

## Bước 1 — Kết nối Neo4j

Đã chuẩn bị từ Bước 4. File `.env.graph_rag` (cùng thư mục `graph_rag_labs/`)
chứa:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YourSecurePassword123
NEO4J_DATABASE=neo4j
NEO4J_VECTOR_INDEX_KBHOPS=kbhops_chunk_embedding
```

> Community Edition không tạo được database `kb-hops` riêng → script dùng
> database `neo4j` mặc định và cô lập dữ liệu qua label `kbhopsDocument` /
> `kbhopsChunk`.

## Bước 2 — Multi-hop retrieval

Pipeline:

```
câu hỏi → embed tiếng Việt → vector search (top-k)
                       ↓
                seed chunks + seed Documents
                       ↓
       duyệt CAN_CU | THAY_THE | HOP_NHAT (1..N bước)
                       ↓
                related Documents + chunks
                       ↓
            gộp + xếp hạng theo score
```

### API

```python
from src.retriever import MultiHopRetriever

with MultiHopRetriever() as r:
    result = r.search(
        "Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ",
        top_k=3,
        num_hops=2,                  # 0 = vector-only, 1 = expand 1 hop, ...
        max_extra_chunks_per_doc=3,  # mỗi doc sau hop lấy tối đa 3 chunk
    )

for c in result.chunks:
    print(c.source, c.score, c.text[:80])
for d in result.documents:
    print(d.hops, d.via_relationship, d.path)
```

### Kết quả mẫu

Với query `"Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ"`:

```
Query: 'Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ'
num_hops=0
  → 2 document(s) | 3 chunk(s)
  Top chunks:
    [vector] score=+0.932 [paragraph] KT. THỐNG ĐỐC PHÓ THỐNG ĐỐC…
    [vector] score=+0.930 [article]  Điều 6 - Trích lập dự phòng rủi ro
    [vector] score=+0.924 [article]  Điều 4 - Điều khoản thi hành

num_hops=1
  → 8 document(s) | 6 chunk(s)
  Documents theo graph path:
    [ROOT] c0001-9d3d51  hops=0 score=0.937  → Thông tư 02/2023/TT-NHNN
           doc-luat-cac-to-chuc-tin-dung  hops=1 score=0.850 via CAN_CU
           doc-nghi-quyet-so-50-nq-cp-...  hops=1 score=0.850 via CAN_CU
           102/2022/NĐ-CP  hops=1 score=0.850 via CAN_CU
           ...
```

## Chạy demo

```bash
cd graph_rag_labs/step6_multi_hop
python -m src.run_demo
```

Demo chạy 4 câu truy vấn tiếng Việt với 3 cấu hình hops (0, 1, 2) → in
đường đi + chunks + ghi `output/multi_hop_results.json`.

## Cấu trúc thư mục

```
step6_multi_hop/
├── README.md
├── output/
│   └── multi_hop_results.json
└── src/
    ├── __init__.py
    ├── config.py            ← load .env.graph_rag
    ├── neo4j_client.py      ← driver + session_scope
    ├── embedder.py          ← CPU Vietnamese encoder
    ├── retriever.py         ← MultiHopRetriever + Result classes
    └── run_demo.py          ← entrypoint + console report
```

## Cài thêm

```bash
python3 -m pip install neo4j python-dotenv sentence-transformers
```

## Score semantics

| Source | Score |
| --- | --- |
| `vector` | cosine similarity (0-1) |
| `hop:1` | 0.85 (= seed × 0.85) |
| `hop:2` | 0.85² = 0.7225 |

Tất cả chunks `hop:*` mang từ các Document mở rộng, đảm bảo độ liên quan
giảm dần theo số bước nhảy. Caller có thể lọc `score >= threshold` hoặc
`hops <= max_hops` để kiểm soát ngữ cảnh.

## Mở rộng (sẽ làm ở Bước 7+)

* Nhồi `chunks` + `heading_path` vào prompt LLM (Gemini) để generate câu
  trả lời grounded.
* Thêm `PARENT_OF` walk theo chiều ngược (từ chunk đến Chương → Điều →
  Document) để enrichment context cho từng hit.
* Bổ sung `co-citation` (2 doc cùng được cite bởi 1 doc thứ 3) làm
  edge `CO_CITED_WITH` mới.