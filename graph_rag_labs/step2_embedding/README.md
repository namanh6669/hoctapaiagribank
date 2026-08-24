# Bước 2 — Tạo Vector Nhúng (Embedding)

Mục tiêu: chuyển các chunk văn bản (Bước 1) thành các vector dense, dùng cho
truy vấn ngữ nghĩa và xây dựng đồ thị RAG.

## Model

`thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5` (HuggingFace):

- SBERT distilled, fine-tuned trên mmarco + tiếng Việt.
- 384 chiều, ~90 MB.
- Đã train với `cosine-similarity` nên L2-normalise trước khi tính
  cosine.

## Cài đặt (CPU-only)

```bash
# PyTorch CPU-only (không có CUDA):
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch

# Sentence-Transformers + NumPy:
python3 -m pip install sentence-transformers numpy
```

Không cần GPU. Module đã pin `device="cpu"` và `torch.set_num_threads(4)` để
không chiếm hết CPU laptop.

## Chạy demo

```bash
cd graph_rag_labs/step2_embedding
python -m src.run_demo
```

Mặc định đọc `../step1_chunking/output/chunks.json`. Có thể truyền file
khác:

```bash
python -m src.run_demo /path/to/chunks.json
```

## Cấu trúc thư mục

```
step2_embedding/
├── README.md
├── data/                          # (trống — input nằm ở step1)
├── output/
│   ├── embeddings.npz             # vectors (N, 384) float32 + ids
│   ├── embeddings_meta.json       # metadata cho mỗi vector
│   ├── embeddings_report.json     # tóm tắt (model, dim, ms/chunk, …)
│   └── sample/
│       └── embeddings_sample.json # 3 vector đầu + metadata
└── src/
    ├── __init__.py
    ├── embedder.py                # load model + encode + persist
    └── run_demo.py                # console demo
```

## Kết quả in ra console

1. Thông tin môi trường (`torch`, `cuda is_available`, threads).
2. Thông tin model (`dim`, `max_seq_length`).
3. Batch stats: `N`, `dim`, `dtype`, `elapsed`, `ms / chunk`.
4. 5 vector đầu (8 chiều đầu + L2-norm + chunk info).
5. **Cosine similarity**: chunk đầu tiên ↔ 5 chunk giống nhất (tự kiểm tra).
6. **Semantic search**: 3 câu truy vấn tiếng Việt ↔ top-3 chunk giống nhất.
7. Vị trí file output.

## Dùng cho Bước 3+

- `embeddings.npz` — load nhanh cho retrieval.
- `embeddings_meta.json` — payload cho cypher/upsert vào Neo4j.
- `cosine_similarity(a, b)` — hàm dùng cho cả query-side lẫn cross-chunk
  edge weighting.