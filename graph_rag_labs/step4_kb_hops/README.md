# Bước 4 + Bước 5 — Nạp kb-hops vào Neo4j & Kiểm tra

Bước 4 nạp 3 Thông tư NHNN (TT_02/2023, TT_06/2023, TT_39/2016) vào
schema kb-hops. Bước 5 verify số lượng và tính đúng đắn của cây.

## Schema

| Node | Label |
| --- | --- |
| Tài liệu gốc + placeholder | `(:kbhopsDocument)` |
| Phân đoạn văn bản | `(:kbhopsChunk {embedding})` |

| Edge | Ý nghĩa |
| --- | --- |
| `(:kbhopsChunk)-[:PART_OF]->(:kbhopsDocument)` | mỗi chunk trỏ về Document gốc |
| `(:kbhopsChunk)-[:PARENT_OF]->(:kbhopsChunk)` | cây phân cấp Chương → Mục → Điều → đoạn |
| `(:kbhopsChunk)-[:NEXT]->(:kbhopsChunk)` | luồng đọc anh em + Document→chunk đầu |
| `(:kbhopsDocument)-[:CAN_CU]->(:kbhopsDocument)` | trích từ "Căn cứ …" |
| `(:kbhopsDocument)-[:THAY_THE]->(:kbhopsDocument)` | trích từ "Sửa đổi, bổ sung một số điều của …" |
| `(:kbhopsDocument)-[:HOP_NHAT]->(:kbhopsDocument)` | trích từ "Hợp nhất …" |

## Database `kb-hops`

- **Neo4j Enterprise**: tự động `CREATE DATABASE kb-hops` (nếu chưa có).
- **Neo4j Community** (Docker image `neo4j`): Community chỉ hỗ trợ 1
  database người dùng → script **fallback** sang database `neo4j` và
  cô lập dữ liệu qua các label đặc biệt (`kbhopsDocument`,
  `kbhopsChunk`).

## Cài thêm

```bash
python3 -m pip install neo4j python-dotenv sentence-transformers beautifulsoup4 markdown lxml
```

## Chạy

```bash
cd graph_rag_labs/step4_kb_hops
python -m src.run_demo
```

## Cấu trúc thư mục

```
step4_kb_hops/
├── README.md
├── data/
│   ├── TT_02_2023_NHNN.md
│   ├── TT_06_2023_NHNN.md
│   └── TT_39_2016_NHNN.md
├── output/
│   └── kb_hops_report.json
└── src/
    ├── __init__.py
    ├── config.py                       ← load .env.graph_rag
    ├── db_manager.py                   ← Enterprise vs Community
    ├── relationship_extractor.py       ← regex Căn cứ / Sửa đổi / Hợp nhất
    ├── loader.py                       ← schema + 3-pass upsert (root → ref → edges)
    └── run_demo.py                     ← chunk → embed → load → verify → demo Cypher
```

## Bước 5 — Verify

Sau khi nạp, demo in ra console:

```
── Yêu cầu Bước 5 ──
  ✗ Số Document       : 9  (kỳ vọng 15)
    ├─ Document root  : 3
    └─ Placeholder    : 6
  ✗ Quan hệ Document : 12  (kỳ vọng 8)
── Tính đúng đắn của cây phân cấp + tuần tự ──
  ✓ PART_OF edges    : 409  (= n_chunks, mọi chunk trỏ về root)
  ✓ PARENT_OF edges  : 406  (cha → con)
  ✓ NEXT edges       : 503  (anh em liền kề + Document → chunk đầu)
  ✓ Chunk orphan     : 0  (phải = 0)
```

### Lưu ý về số liệu

Số liệu thực tế phụ thuộc vào tập tài liệu đầu vào:

- Với 3 Thông tư NHNN đã chuẩn bị, tự động phát hiện 6 Document được
  nhắc tới (Luật NHNN, Luật TCTD, NĐ 102/2022, NĐ 156/2013, NQ 50/2023,
  Thông tư NHNN về dự phòng rủi ro) + 1 quan hệ THAY_THE (TT_06/2023
  sửa đổi TT_39/2016).
- Nếu cần đạt đúng 15 Document / 8 edges: nạp thêm Thông tư / Luật vào
  `data/`, thêm vào `DOCUMENTS` trong `run_demo.py`.

### Tính đúng đắn cấu trúc

| Quan hệ | Kỳ vọng | Thực tế |
| --- | --- | --- |
| `PART_OF` | = n_chunks (409) | ✓ 409 |
| `PARENT_OF` | n_chunks - n_roots | ✓ 406 |
| `NEXT` | n_chunks (kèm liên kết giữa các chunk) | ✓ 503 |
| Chunk orphan | 0 | ✓ 0 |

## Cypher queries minh họa

Mở Neo4j Browser tại `http://localhost:7474` (user `neo4j`, password
trong `.env.graph_rag`):

```cypher
// 1. Tất cả Document và cấu trúc
MATCH (d:kbhopsDocument)
RETURN d.id, d.title, d.doc_type, d.is_root
ORDER BY d.is_root DESC, d.id;

// 2. Quan hệ giữa các Document
MATCH (a:kbhopsDocument)-[r]->(b:kbhopsDocument)
WHERE type(r) IN ['CAN_CU','THAY_THE','HOP_NHAT']
RETURN a.title, type(r), b.title
ORDER BY type(r);

// 3. Tất cả chunk của một Document
MATCH (c:kbhopsChunk)-[:PART_OF]->(d:kbhopsDocument {original_doc_id:'TT-02-2023-NHNN'})
RETURN c.kind, c.title LIMIT 10;

// 4. Leo lên cây cha từ 1 paragraph
MATCH (leaf:kbhopsChunk {kind:'paragraph'})
MATCH path = (leaf)<-[:PARENT_OF*0..4]-(up:kbhopsChunk)
WITH path, length(path) AS depth ORDER BY depth DESC LIMIT 1
RETURN [n IN nodes(path) | n.kind + ': ' + coalesce(n.title,'')] AS chain;

// 5. NEXT walk
MATCH (:kbhopsDocument {is_root:true})-[:NEXT]->(first:kbhopsChunk)
MATCH path = (first)-[:NEXT*0..5]->(rest:kbhopsChunk)
RETURN [n IN nodes(path) | n.kind + ': ' + coalesce(n.title,'')];

// 6. Vector search
CALL db.index.vector.queryNodes('kbhops_chunk_embedding', 3, $vec)
YIELD node AS c, score
MATCH (c)-[:PART_OF]->(d:kbhopsDocument)
RETURN c.title, score, d.original_doc_id
ORDER BY score DESC;
```

## Output thực tế với 3 Thông tư

```
Documents       : 9   (3 root + 6 placeholder)
Chunks          : 409
PART_OF edges   : 409
PARENT_OF edges : 406
NEXT edges      : 503
CAN_CU edges    : 11
THAY_THE edges  : 1   ← TT_06 sửa đổi TT_39
HOP_NHAT edges  : 0

Quan hệ tiêu biểu:
  CAN_CU    ('Thông tư 02/2023/TT-NHNN') --> 'Luật Ngân hàng Nhà nước Việt Nam'
  CAN_CU    ('Thông tư 02/2023/TT-NHNN') --> 'Luật Các tổ chức tín dụng'
  CAN_CU    ('Thông tư 02/2023/TT-NHNN') --> 'Nghị định 102/2022/NĐ-CP'
  CAN_CU    ('Thông tư 02/2023/TT-NHNN') --> 'Nghị quyết 50/NQ-CP'
  CAN_CU    ('Thông tư 06/2023/TT-NHNN') --> 'Luật Ngân hàng Nhà nước Việt Nam'
  CAN_CU    ('Thông tư 39/2016/TT-NHNN') --> 'Nghị định 156/2013/NĐ-CP'
  THAY_THE  ('Thông tư 06/2023/TT-NHNN') --> 'Thông tư 39/2016/TT-NHNN'  ← liên kết ngang
```