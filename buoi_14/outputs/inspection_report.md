# Buổi 14 — Project Pre-Check Inspection Report

> Phạm vi: chỉ đọc, **không sửa/ghi/xoá** 3 file trong `kb+hops/`.
> Không chạy retrieval, không tạo Knowledge Graph ở bước này.

---

## 1. Cấu trúc `buoi_14/` hiện tại

```
buoi_14/
├── .venv/                         (Python 3.12.6 virtualenv, tạo ở bước trước)
└──  kb+hops/                      (nguồn, chỉ đọc)
    ├── content.csv
    ├── metadata.csv
    └── relationships.csv
```

**File theo extension mục tiêu (`.py / .md / .csv / .json / requirements*.txt / .env*`):**

| Đường dẫn | Loại | Ghi chú |
|---|---|---|
| ` kb+hops/content.csv` | CSV | nguồn (read-only) |
| ` kb+hops/metadata.csv` | CSV | nguồn (read-only) |
| ` kb+hops/relationships.csv` | CSV | nguồn (read-only) |

- Không có file `.py`, `.md`, `.json`, `requirements.txt`, `.env` nào trong project root.
- Thư mục `outputs/` (chứa báo cáo này) được tạo mới bằng lệnh `mkdir -p` sau khi pre-check hoàn tất.

---

## 2. Hồ sơ 3 file nguồn (đọc thật, không suy diễn)

### 2.1 `metadata.csv`

| Tiêu chí | Giá trị |
|---|---|
| Encoding | `utf-8-sig` (có BOM, đã strip khi đọc) |
| Dung lượng | 11,575 bytes |
| Tổng dòng (non-empty) | 31 = 1 header + 30 dòng dữ liệu |
| Số dòng dữ liệu | **30** |
| Số cột | **17** |
| Tên cột | `id, title, so_ky_hieu, ngay_ban_hanh, loai_van_ban, ngay_co_hieu_luc, ngay_het_hieu_luc, nguon_thu_thap, ngay_dang_cong_bao, nganh, linh_vuc, co_quan_ban_hanh, chuc_danh, nguoi_ky, pham_vi, thong_tin_ap_dung, tinh_trang_hieu_luc` |
| Dòng trùng lặp (hash exact) | **0** |
| Null/empty theo cột đáng chú ý | `thong_tin_ap_dung` 30/30 rỗng; `ngay_het_hieu_luc` 28/30 rỗng; `ngay_dang_cong_bao` 22/30 rỗng |
| Khoá ứng viên (unique & non-null) | `id`, `title`, `so_ky_hieu` |
| Trường text phù hợp retrieval | `title` (avg 117, max 237 ký tự) |
| Metadata phù hợp citation | `so_ky_hieu`, `ngay_ban_hanh`, `loai_van_ban`, `ngay_co_hieu_luc`, `co_quan_ban_hanh`, `nguoi_ky`, `pham_vi`, `tinh_trang_hieu_luc`, `linh_vuc`, `id` |

**Phân bố loại văn bản (low cardinality, tốt cho filter):**
- Thông tư: 19 — Nghị định: 5 — Luật: 4 — Văn bản hợp nhất: 2

---

### 2.2 `relationships.csv`

| Tiêu chí | Giá trị |
|---|---|
| Encoding | `utf-8-sig` |
| Dung lượng | 34,080 bytes |
| Tổng dòng (non-empty) | 174 = 1 header + 173 dòng |
| Số dòng dữ liệu | **173** |
| Số cột | **10** |
| Tên cột | `source_kind, source_id, source_name, target_kind, target_id, target_name, relationship_type, method, confidence, evidence` |
| Dòng trùng lặp | **0** |
| Null/empty | 0 ở mọi cột (đầy đủ) |
| Khoá ứng viên | **Không có** (chỉ có unique thoáng qua, không chuẩn khoá tổ hợp) |

**Cardinality & phân bố đáng lưu ý:**
- `source_kind`: 1 giá trị — toàn bộ là `Document` (173/173).
- `target_kind`: 2 giá trị — `Entity` (144), `Document` (29).
- `source_id`: 33 giá trị unique; `source_name` cũng 30 giá trị (một số doc xuất hiện nhiều lần).
- `target_id`: 76 giá trị unique — gồm mã `E0001..E0076` (Entity) và `xx/yyyy/QH15`, `xx/yyyy/TT-…` (Document).
- `relationship_type` (7 loại):
  - `AP_DUNG_CHO` 61 — `BAN_HANH_BOI` 30 — `KY_BOI` 30
  - `THUOC_LINH_VUC` 28 — `THAM_CHIEU` 7 — `SUA_DOI_BO_SUNG` 10 — `THAY_THE_BOI` 7
- `method`: `gemini` 74 — `metadata` 70 — `rule` 29
- `confidence`: range 0.75 → 1.0 (rule/metadata = 0.9/1.0, gemini = 0.75)

**Trường text phù hợp retrieval:** `evidence` (avg 86, max 175 ký tự — snippet trích từ văn bản).
**Metadata phù hợp citation (để truy ngược về nguồn):** `source_id` + `source_name`, `target_id` + `target_name`, `relationship_type`, `method`, `confidence`.

---

### 2.3 `content.csv`

| Tiêu chí | Giá trị |
|---|---|
| Encoding | `utf-8-sig` |
| Dung lượng | 5,748,937 bytes (~5.5 MB) |
| Tổng dòng | 19,822 (đa số là wrap của HTML trong ô) |
| Số dòng dữ liệu (CSV row-level) | **30** (mỗi văn bản 1 dòng, HTML nằm trong 1 ô) |
| Số cột | **2** |
| Tên cột | `id`, `content_html` |
| Dòng trùng lặp | **0** |
| Số `id` unique | 30/30 (khớp 1-1 với `metadata.csv`) |
| Dòng rỗng | 0/30 |
| Độ dài cột `content_html` | min 5,574 — **avg 161,162** — max **557,135** ký tự |

**Cảnh báo kỹ thuật:** `csv.field_size_limit()` mặc định (131,072) **không đủ** để parse trực tiếp — đã nâng lên `sys.maxsize` để đọc thành công. Khi viết `prepare_corpus.py` phải nâng giới hạn này (hoặc dùng `pandas.read_csv` vì pandas tự xử lý kích thước lớn).

**Cấu trúc HTML mẫu (lấy 3 văn bản đầu):**
- Thẻ phổ biến: `span` 2,219 — `p` 1,878 — `strong` 179 — `br` 45 — `div` 19 — `td` 12 — `em` 12 — `tr/table/tbody` 6/4/4 — `html/head/body` 3/3/3.
- Tài liệu có `table` ở cuối (phụ lục), nội dung chính ở `p`/`span`.
- Marker tiếng Việt trong plain text: `Điều \d+` xuất hiện **431** lần, `Khoản \d+` **25** lần (chỉ trong 3 doc) → đủ dày để chunk theo `Điều`.

**Trường text phù hợp retrieval:** `content_html` (sau khi strip tag thành plain text).
**Metadata phù hợp citation:** `id` (join ngược về `metadata.csv` để lấy `so_ky_hieu, title, …`).

---

## 3. Quan hệ khoá giữa 3 file

```
content.csv.id   ──┐                   ┌── relationships.source_id (Document)
                   │                   │
metadata.csv.id  ──┼── 1-1 (cùng 30 ID) ┤
                   │                   │
                   │                   └── relationships.target_id (Document side; Entity side trỏ tới E0001..)
```

- `metadata.id` và `content.id` cùng tập 30 giá trị, **liên kết 1-1** và dùng làm canonical doc id.
- `relationships.source_id` (Document) tham chiếu 33 doc → có 3 doc không nằm trong top 30 ở `metadata.csv` (cần đối chiếu kỹ trước khi load KG).
- 12 `source_id` có dạng UUID (`f69936f0-…`, `6e689cd0-…`) trùng với `metadata.id` UUID — dùng làm khoá nối an toàn.
- `relationships.source_name` chỉ mang tính hiển thị; vẫn ưu tiên nối bằng `id`.

---

## 4. Quét code `buoi_14/` — pattern phá dữ liệu

Lệnh quét: `grep -rE "os\.remove|shutil\.rmtree|rm -rf|os\.unlink|open\([^)]*'[^']*'[^)]*,'w'|DELETE FROM|DROP TABLE|DETACH DELETE"`

Kết quả:

- **Trong `buoi_14/` project code:** **0 match** — chưa có file `.py/.cypher/.sh` nào được viết.
- **Ngoài project:** các match duy nhất nằm trong `.venv/lib/python3.12/site-packages/pip/...` — đây là thư viện hệ thống, **không phải code do dự án** này viết; venv là môi trường cô lập, không ảnh hưởng dữ liệu nguồn.

Kết luận: không có code dự án nào đọc/ghi/xoá `kb+hops/`; rủi ro xoá dữ liệu nguồn ở bước này bằng **0**.

---

## 5. Môi trường

| Hạng mục | Trạng thái |
|---|---|
| Working root | `/Users/nana/Documents/GitHub/hoctapaiagribank/buoi_14` |
| Python hệ thống | `/opt/homebrew/bin/python3` 3.12.6 |
| `.venv` | đã tạo tại `buoi_14/.venv`, interpreter chạy OK (3.12.6, base tách hẳn hệ thống, macOS arm64) |
| `pip` trong venv | 24.2 (mặc định, chưa upgrade) |
| `import pandas` trong venv | **FAIL — chưa cài** (`ModuleNotFoundError: No module named 'pandas'`) |
| `requirements.txt` | **không tồn tại** (chưa tạo theo yêu cầu trước) |
| `.env` / `.env.example` | **không tồn tại** |
| File `.py/.cypher/.sh` trong project | **không có** |

> Bước tiếp theo nếu muốn chạy code: tạo `requirements.txt` + `pip install` các gói đề xuất (pandas, rank-bm25, sentence-transformers, neo4j, tqdm, python-dotenv). KHÔNG cài LangChain/LlamaIndex.

---

## 6. Rủi ro & điểm cần lưu ý khi vào bước tiếp theo

| # | Rủi ro | Mức | Hành động đề xuất |
|---|---|---|---|
| R1 | `content.csv` ô HTML vượt `csv.field_size_limit` mặc định | Trung bình | `csv.field_size_limit(sys.maxsize)` hoặc dùng `pandas.read_csv` (tự xử lý) trong `prepare_corpus.py` |
| R2 | Lệch giữa 30 doc trong `metadata/content` và 33 doc xuất hiện trong `relationships.source_id` | Thấp | Đối chiếu & quyết định: (a) bỏ qua edge nguồn trỏ tới doc không có trong tập 30, (b) yêu cầu bổ sung metadata. **Chưa cần xử lý ở pre-check.** |
| R3 | HTML chứa `&nbsp;` và các ký tự Unicode đặc biệt → tokenizer tiếng Việt cần normalize | Thấp | Khi chunk, decode HTML entities trước (`html.unescape`) rồi strip tag |
| R4 | `relationships.source_id` có cả số lẫn UUID (chuỗi) → cần join 2 kiểu | Thấp | Dùng so khớp kiểu lỏng (string) sau khi cast sang str |
| R5 | Một số trường metadata rỗng trên diện rộng (`thong_tin_ap_dung`, `ngay_het_hieu_luc`) — không nên bịa khi citation | Trung bình | Citation chỉ dùng trường có giá trị; nếu thiếu thì ghi `[không có]` chứ không suy diễn. Khớp với rule "không bịa tên từ mã" trong memory `wiki-risk-graph-mvp-scope.md` |
| R6 | Văn bản ở content.csv rất dài (max ~557 KB ≈ hơn 100 trang A4) → chunk bắt buộc | Trung bình | Chunk theo `Điều \d+` (regex đã đếm 431 marker chỉ trong 3 doc); cộng thêm chunk-size cap và overlap |
| R7 | Quét destructive pattern về sau | Thấp | Khi viết code, giữ nguyên tắc **chỉ ghi vào `buoi_14/data/processed/`, `buoi_14/outputs/`**; mọi thao tác mở file phải dùng `open(path, "r")` hoặc chỉ rõ path nằm trong project. Không bao giờ mở `kb+hops/...csv` với mode `"w"/"a"/"x"`. |
| R8 | Cài nhầm LangChain/LlamaIndex làm nặng môi trường | Thấp | Theo rule prompt: KHÔNG cài 2 framework này trừ khi có lý do bắt buộc. |

---

## 7. Khối dữ liệu phù hợp từng mục tiêu

| Mục tiêu | Trường nguồn |
|---|---|
| Corpus retrieval (BM25 / dense) | `content.content_html` → strip tag → chunk theo `Điều` |
| Tài liệu tham chiếu khi trả lời | `metadata.{id, so_ky_hieu, title, ngay_ban_hanh, loai_van_ban, co_quan_ban_hanh, tinh_trang_hieu_luc}` join bằng `content.id = metadata.id` |
| Mini Knowledge Graph | `relationships.{source_*, target_*, relationship_type, method, confidence, evidence}` |
| Citation enrichment (khi bước sau) | `metadata.*` (rút gọn còn trường có giá trị) — KHÔNG bịa |

---

## 8. Đề xuất dependency tối thiểu (chưa cài)

| Nhóm | Gói | Ghi chú |
|---|---|---|
| CSV / bảng | `pandas`, `numpy` | `pandas.read_csv` tự xử lý ô lớn |
| BM25 | `rank-bm25` | pure-Python, đủ cho ~30 văn bản |
| Dense embed | `sentence-transformers` (+ `torch`) | mô hình đa ngôn ngữ, ví dụ `intfloat/multilingual-e5-base` hoặc `BAAI/bge-m3` |
| Rerank | `sentence-transformers` CrossEncoder | `BAAI/bge-reranker-v2-m3` (đa ngôn ngữ) |
| Neo4j | `neo4j` (driver thuần, không kèm server) | |
| Tiện ích | `tqdm`, `python-dotenv` | |

**Loại bỏ chủ động:** LangChain, LlamaIndex, FlagEmbedding, Chroma/Qdrant client, Neo4j server.

---

## 9. Kết luận pre-check

- ✅ 3 file nguồn đã đọc thật, schema xác nhận, không có dòng trùng.
- ✅ Buổi này chưa có code `.py`/`.cypher`/`.sh` nào → bề mặt xoá/sửa dữ liệu = 0.
- ✅ `kb+hops/` tuyệt đối read-only; mọi file mới chỉ ghi vào `buoi_14/{data/processed, outputs, scripts, src, cypher, tests}`.
- ⚠️ Thiếu `requirements.txt`, `pandas`, model embedding — chưa thể chạy retrieval ở bước này.
- ⚠️ `content.csv` cần nâng `csv.field_size_limit` khi đọc.

**Safe to continue: NO** (vì `pandas` chưa cài và `requirements.txt` chưa có — cần 1 bước setup nhỏ trước khi vào `inspect_project.py` / `prepare_corpus.py` thực sự).
