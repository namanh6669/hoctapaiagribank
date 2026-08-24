# Bước 1 — Làm sạch HTML & Chunking phân cấp

Mục tiêu của bước này là chuyển văn bản pháp lý (Markdown/HTML) sang dạng
**graph-ready**: mỗi đoạn văn bản trở thành một node trong cây phân cấp,
giữ nguyên cấu trúc đọc và bỏ đi HTML thừa.

## Cấu trúc thư mục

```
step1_chunking/
├── README.md
├── data/
│   └── TT_02_2023_NHNN.md        # tài liệu thô (markdown)
├── output/
│   ├── chunks.json               # toàn bộ chunks sau khi xử lý
│   └── sample/
│       └── chunks_sample.json    # 8 chunk đầu tiên
└── src/
    ├── __init__.py
    ├── cleaner.py                # HTML/MD -> HTML sạch (chỉ giữ cấu trúc)
    ├── chunker.py                # HTML sạch -> cây chunks (Cha-Con + NEXT)
    └── run_demo.py               # entrypoint, in kết quả mẫu ra console
```

## Chạy thử

```bash
cd graph_rag_labs/step1_chunking
python -m src.run_demo
```

## Kết quả mong đợi

Script sẽ in ra console:

1. **Kích thước trước/sau khi làm sạch** (giảm bao nhiêu %).
2. **Trước/sau HTML fragment** (600 ký tự đầu).
3. **Tóm tắt cấu trúc chunk**: tổng số chunk, số lượng theo `kind`,
   trung bình ký tự / leaf chunk.
4. **Cây 3 lớp đầu** (Document → Chương → Mục/Điều).
5. **3 chunk mẫu chi tiết** (Chương, Điều, Đoạn văn) với đầy đủ:
   `id`, `kind`, `depth`, `heading_path`, `parent_id`, `children_ids`,
   `next_id`, `char_count`, `text`.

## Mô hình dữ liệu

```
Document (root, kind=document)
  └─ Chương I, II, …        (kind=chapter)
       └─ Mục 1, 2, …        (kind=section)  [optional]
            └─ Điều 1, 2, …  (kind=article)
                 ├─ Paragraph (kind=paragraph)
                 ├─ List     (kind=list)
                 └─ Table    (kind=table)
```

Mỗi node giữ **2 loại quan hệ**:

- **Cha → Con**: `parent_id` / `children_ids` (cây chứa).
- **NEXT (anh em liền kề)**: `next_id` của node này trỏ tới node kế tiếp
  trong luồng đọc. Node cuối cùng trỏ về root.

Vì đã loại bỏ các trường HTML cồng kềnh (`class`, `style`, `data-*`, ...),
mỗi chunk chỉ chứa `text` sạch và metadata ngắn gọn.

## Khi tích hợp Graph-RAG

- `parent_id` → quan hệ `HAS_CHILD` (Document → Chương → Điều).
- `next_id` → quan hệ `NEXT` giữa các node anh em.
- `heading_path` → dùng cho embedding + LLM context (giúp LLM biết đang ở
  chương/điều nào).