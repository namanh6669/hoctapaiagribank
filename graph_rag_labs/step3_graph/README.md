# Bước 3 — Nạp Graph + Truy vấn Cypher

Đưa 91 chunks + vectors (Bước 1 + Bước 2) vào Neo4j, tạo vector index, chạy
các truy vấn Cypher minh họa retrieval ngữ nghĩa kết hợp đồ thị.

## Cấu hình

1. Copy file mẫu và điền mật khẩu thật:

    ```bash
    cp ../.env.graph_rag.template ../.env.graph_rag
    # Sửa NEO4J_PASSWORD trong file vừa copy
    ```

2. Đảm bảo Neo4j đang chạy (mặc định `bolt://localhost:7687`).

## Cài thêm

```bash
python3 -m pip install neo4j python-dotenv
```

## Chạy demo

```bash
cd graph_rag_labs/step3_graph
python -m src.run_demo
```

## Schema đồ thị

| Loại | Label | Ghi chú |
| ---- | ----- | ------- |
| Root | `:Chunk:Document` | tiêu đề thông tư |
| Container | `:Chunk:Chapter`, `:Chunk:Section`, `:Chunk:Article`, `:Chunk:Subarticle` | có `HAS_CHILD` xuống con |
| Leaf | `:Chunk:Paragraph`, `:Chunk:List`, `:Chunk:Table` | có `HAS_VECTOR` → `:Embedding` |

### Quan hệ

| Edge | Ý nghĩa |
| ---- | ------- |
| `(:Chunk)-[:HAS_CHILD]->(:Chunk)` | cây chứa (cha → con) |
| `(:Chunk)-[:NEXT]->(:Chunk)` | luồng đọc (sibling) |
| `(:Chunk)-[:HAS_VECTOR]->(:Embedding)` | liên kết sang node `Embedding` chứa vector |

### Index

* `chunk_id` (unique trên `Chunk.id`)
* `embedding_id` (unique trên `Embedding.id`)
* `chunk_kind` (BTREE trên `Chunk.kind`)
* `chunk_embedding` (vector index trên `Embedding.vec`, dim=384, cosine)

## Cấu trúc thư mục

```
step3_graph/
├── README.md
├── output/
│   └── graph_report.json         ← tóm tắt sau khi nạp
├── cypher/                       ← (dành cho Cypher snippet tái sử dụng)
└── src/
    ├── __init__.py
    ├── config.py                 ← load .env.graph_rag
    ├── neo4j_client.py           ← driver + session helpers
    ├── loader.py                 ← schema, upsert chunks + embeddings
    └── run_demo.py               ← entrypoint + console demo
```

## Demo Cypher

Sau khi nạp, demo chạy 5 truy vấn:

1. `MATCH (d:Document)-[:HAS_CHILD]->(c:Chunk)` — in root + 3 chunk đầu.
2. **NEXT walk** từ Document, đi 6 bước qua các `NEXT` edge.
3. Một `:Article` + các node con của nó.
4. Leo lên cây cha từ một `:Paragraph` đến `:Document`.
5. **Vector search** qua `db.index.vector.queryNodes` với 2 câu truy vấn
   tiếng Việt — kết quả trả về top-3 chunks cùng đường dẫn cha.

Ví dụ output:

```
[Q5] Vector search — top-3
  Query: 'Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ'
    #1 score=+0.9362 [article] 'Điều 5 - Giữ nguyên nhóm nợ và phân loại nợ'
    #2 score=+0.9332 [article] 'Điều 4 - Cơ cấu lại thời hạn trả nợ'
    #3 score=+0.9317 [article] 'Điều 6 - Trích lập dự phòng rủi ro'
```

## Re-run

Demo wipe DB trước khi nạp lại (`wipe_db=True`), nên có thể chạy đi chạy
lại nhiều lần mà không lo trùng lặp. Tắt wipe bằng cách sửa flag nếu
muốn append.

## Mở rộng (sẽ làm ở Bước 4+)

* Gộp câu truy vấn tiếng Việt → vector → vector index → lấy top-K chunks.
* Với mỗi chunk, leo lên cha (`:HAS_CHILD` đảo ngược hoặc truy vấn qua
  `heading_path`) → lấy ngữ cảnh Chương/Điều để nhồi vào prompt cho LLM.
* Thêm `BEFORE/AFTER` edge để mô tả quan hệ logic giữa các Điều (ví dụ
  "Điều 5 áp dụng cho các trường hợp của Điều 4").