# Buổi 14 — Hybrid Search + Reranking + Mini Knowledge Graph

Chuẩn bị môi trường và corpus cho pipeline retrieval trên 30 văn bản pháp luật
Việt Nam (Luật, Nghị định, Thông tư NHNN/BTC, văn bản hợp nhất…).

## Cấu trúc dự kiến

```
buoi_14/
├── data/
│   └── processed/
│       └── chunks_normalized.csv        # chunk corpus phục vụ retrieval
├── scripts/
│   └── prepare_corpus.py                # chuẩn hoá content.csv + metadata.csv
├── src/                                 # (bước sau) bm25, dense, hybrid, reranker
├── cypher/                              # (bước sau) schema & demo queries cho Neo4j
├── outputs/
│   └── inspection_report.md             # báo cáo pre-check môi trường & dữ liệu
├── tests/                               # (bước sau) test_retrieval.py
├── .env.example                         # (bước sau) Neo4j creds + HuggingFace cache
├── requirements.txt                     # danh sách dependency tối thiểu
└── README.md
```

## Quy tắc chung

- Dữ liệu nguồn `kb+hops/` là **read-only**. Mọi script chỉ `open(..., "r")` trên
  đường dẫn đó; output ghi vào `buoi_14/{data,outputs,src,scripts,cypher,tests}/`.
- **Không** cài LangChain / LlamaIndex / FlagEmbedding. Stack tối thiểu: pandas,
  rank-bm25, sentence-transformers (+ torch), neo4j (driver), beautifulsoup4,
  lxml, tqdm, python-dotenv.

## Cài đặt (đã thực hiện)

```bash
# 1. Tạo virtualenv
python3 -m venv buoi_14/.venv

# 2. Cài dependencies
buoi_14/.venv/bin/pip install -r buoi_14/requirements.txt
```

## Chuẩn bị corpus

```bash
buoi_14/.venv/bin/python buoi_14/scripts/prepare_corpus.py
```

Script:

1. Đọc ` kb+hops/metadata.csv` và ` kb+hops/content.csv`.
2. Strip HTML (`bs4` + `lxml`), normalize UTF-8, collapse whitespace.
3. Tách chunk theo marker `Điều \d+` (article). Mỗi `Điều` → 1 chunk.
   Đoạn preamble (header văn bản trước Điều đầu) → 1 chunk với `article = ""`.
4. Nối metadata để giữ citation (`title`, `so_ky_hieu`, `ngay_ban_hanh`,
   `co_quan_ban_hanh`, `effective_date`, `status`).
5. `chunk_id` sinh bằng `uuid5(namespace, doc_id|article|pos|length)` — ổn định
   qua nhiều lần chạy.
6. Ghi `buoi_14/data/processed/chunks_normalized.csv`, in thống kê và 3 record
   mẫu.

Cờ tuỳ chọn:

```bash
buoi_14/.venv/bin/python buoi_14/scripts/prepare_corpus.py --out /path/to/other.csv
```

## Schema `chunks_normalized.csv`

| Cột | Nguồn | Ghi chú |
|---|---|---|
| `chunk_id` | deterministic uuid5 | unique, ổn định |
| `document_id` | `metadata.id` / `content.id` | string |
| `text` | `content.content_html` đã strip tag | UTF-8 |
| `source_file` | đường dẫn tương đối tới content.csv | |
| `title` | `metadata.title` | không bịa |
| `document_type` | `metadata.loai_van_ban` | |
| `chapter` | `Chương X` gần nhất (Roman/Arabic) | có thể trống |
| `section` | — | để trống (chưa tách) |
| `article` | số trong `Điều N.` | trống cho preamble |
| `article_title` | phần heading sau `Điều N.` | |
| `clause` | — | để trống (regex `Khoản` không đáng tin) |
| `effective_date` | `metadata.ngay_co_hieu_luc` | |
| `status` | `metadata.tinh_trang_hieu_luc` | |
| `so_ky_hieu` | `metadata.so_ky_hieu` | giữ cho citation |
| `ngay_ban_hanh` | `metadata.ngay_ban_hanh` | |
| `co_quan_ban_hanh` | `metadata.co_quan_ban_hanh` | |
| `nguoi_ky` | `metadata.nguoi_ky` | |

## Trạng thái sau bước này

Sau khi chạy `prepare_corpus.py`:

- Tổng chunk: **1463**
- Documents: **30**
- Chunk thiếu text: **0**
- `chunk_id` trùng: **0**
- `metadata.csv / content.csv / relationships.csv` SHA-256 **không đổi** so
  với baseline pre-check (xem `outputs/inspection_report.md`).

## Bước tiếp theo (chưa thực hiện)

- `scripts/inspect_project.py` — sinh báo cáo Markdown về corpus (chunks / doc /
  cỡ text, phân bố `article` / `chapter`, sanity check schema).
- `src/bm25_retriever.py`, `src/dense_retriever.py`, `src/hybrid_retriever.py`,
  `src/reranker.py`, `src/citation.py`.
- `scripts/baseline_retrieval.py`, `scripts/hybrid_search.py`, `scripts/rerank.py`,
  `scripts/compare_retrieval.py`.
- `scripts/load_mini_kg.py` + `cypher/schema.cypher` +
  `cypher/demo_queries.cypher` (Neo4j driver, không kèm server).
- `tests/test_retrieval.py`, `.env.example`.

Mọi script sau sẽ tiếp tục quy tắc read-only trên ` kb+hops/` và chỉ ghi vào
`buoi_14/`.

---

## Streamlit Demo

App: `buoi_14/app.py` — giao diện web cho hệ thống retrieval đã hoàn thiện,
**tái sử dụng trực tiếp pipeline của Buổi 14** (`src.bm25_retriever`,
`src.dense_retriever`, `src.hybrid_retriever`, `src.reranker`,
`scripts.query_demo.retrieve`). Không có pipeline song song chỉ dành cho UI.

### Cài

```bash
buoi_14/.venv/bin/pip install -r buoi_14/requirements.txt   # đã có streamlit>=1.30
```

### Chạy

```bash
buoi_14/.venv/bin/streamlit run buoi_14/app.py
```

Streamlit in URL thực tế ra terminal (ví dụ):
```
Local URL: http://localhost:8501
Network URL: http://<lan-ip>:8501
External URL: http://<wan-ip>:8501
```
→ Mở Local URL trên trình duyệt. Nếu muốn đổi port: `streamlit run ... --server.port=8765`.

**Nếu thấy nhiều log `ModuleNotFoundError: torchvision` ở console** (do `transformers` v5+ lazy-scan các vision submodule — ta KHÔNG dùng vision), chạy thêm cờ tắt watcher:
```bash
buoi_14/.venv/bin/streamlit run buoi_14/app.py \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false
```
App vẫn reload khi `app.py` đổi (chỉ thay đổi là watcher không còn scan mọi import để dò file), nhưng console gọn hơn rất nhiều.

**Cách triệt để:** `pip install torchvision` (~100 MB, đã có sẵn trong `requirements.txt`).
Gói này KHÔNG chứa model vision nào được load — chỉ giúp `transformers` thoả mãn import-check cho các vision submodule trong `transformers.models.*` mà ta không dùng tới.

### Dừng

- Trong terminal đang chạy, nhấn **Ctrl+C**.
- Hoặc `pkill -f "streamlit run app.py"`.

### Cách dùng

1. Nhập **câu hỏi** vào ô bên trái (tiếng Việt).
2. Chọn **method** trong dropdown:
   - **BM25** — lexical, mạnh với mã văn bản / số điều.
   - **Dense** — `intfloat/multilingual-e5-base`, mạnh với semantic.
   - **Hybrid** — RRF (k=60) trộn BM25 + Dense.
   - **Hybrid + Rerank** — Hybrid rồi Cross-Encoder `BAAI/bge-reranker-v2-m3`.
3. Kéo **Top-k** (1–20) và **Candidate-k** (chỉ Hybrid: 10–50).
4. Nhấn **Tìm kiếm**.

### Đọc kết quả

Mỗi card hiển thị:
- `rank` — thứ tự 1..top_k.
- `chunk` — `chunk_id` rút gọc.
- `doc` — `document_id` (UUID hoặc số).
- `score` — điểm chính của method đang dùng (BM25 raw · cosine · RRF · cross-encoder).
- `retrieval_method` — đúng method đã chọn.

Với **Hybrid** còn kèm `bm25_rank`, `dense_rank`, `rrf` để bạn thấy vì sao RRF sắp xếp như vậy.
Với **Hybrid + Rerank** còn có bảng **Before / After Rerank** ở trên:

```
                      BEFORE                AFTER
#  chunk     rrf             #  chunk     hybrid-score   rerank-score   Δ
1  aaaa…     +0.0241         1  bcad…     +0.0312       +0.987        +2
2  bcad…     +0.0312         2  efgh…     +0.0149       +0.872        +2
3  efgh…     +0.0249         3  aaaa…     +0.0241       +0.654        -2
```

- `Δ` chênh lệch rank trước-sau (số dương = promoted).
- Nếu rerank là FALLBACK, score có nhãn `FALLBACK-reranker`.

### Graph hints

Phần cuối trang — không phải Graph RAG đầy đủ, chỉ là dữ liệu tham khảo
cho buổi sau:

- `Document IDs` và `Chunk IDs` của top-k.
- Nếu Neo4j sẵn sàng và `lab_session='buoi_14'` đã nạp qua
  `scripts/load_mini_kg.py`: in thêm các edge 1-hop từ VanBan →
  `(VanBan/Entity) -[:THAM_CHIEU|...|AP_DUNG_CHO]-> ...` kèm `confidence`.
- Nếu Neo4j KHÔNG chạy: ghi rõ "Neo4j chưa sẵn sàng — thiếu NEO4J_URI/PASSWORD trong .env".

KG đầy đủ vẫn xem bằng Neo4j Browser (chạy `:source cypher/demo_queries.cypher`).

### Bảo mật

- App **không** hardcode password / API key.
- Đọc credentials qua `python-dotenv.load_dotenv('.env')` rồi `os.environ`.
- Nếu `.env` không tồn tại / thiếu biến, Graph hints sẽ tự thông báo và bỏ qua
  truy vấn — retrieval vẫn chạy bình thường.

