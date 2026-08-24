# Wiki Risk Graph — Buoi 13 (MVP)

Wiki Risk Graph phục vụ đào tạo. Bộ dữ liệu hiện tại là dữ liệu **mô phỏng** (12 RuiRo, 10 KiemSoat, 12 SuKienRuiRo, 22 edge) — KHÔNG phải dữ liệu nghiệp vụ thực tế.

---

## Quy tắc MVP (đọc trước khi thay đổi gì)

| Quy tắc | Tuân thủ |
|---|---|
| Chỉ 3 nhãn node: `RuiRo`, `KiemSoat`, `SuKienRuiRo` | ✅ Toàn bộ script + Cypher đều giới hạn |
| Chỉ 2 loại quan hệ: `MITIGATES`, `OBSERVED_AS` | ✅ Không tự sinh thêm |
| `id` là khóa duy nhất | ✅ Dùng `id` cho constraint (Neo4j) và wikilink (Obsidian) |
| `MERGE` thay cho `CREATE` | ✅ Idempotent, chạy lại không tạo duplicate |
| Cypher parameterized | ✅ `sess.run(cypher, rows=rows)` — không string-interpolate |
| Không hard-code password | ✅ Đọc từ `.env` |
| Không bịa tên đơn vị từ `owner_unit_id` | ✅ Hiển thị mã gốc (`DV-OPS`, …) |
| Không bịa tên vai trò từ `owner_role_id` | ✅ Hiển thị mã gốc (`VT-OPS-CONTROL`, …) |
| Không đổi `verification_status` | ✅ Giữ nguyên giá trị gốc |

**Dữ liệu CHƯA CÓ (không tự bịa):** bảng master `DonVi` / `VaiTro`, các node `VanBan`, `DieuKhoan`, `QuyTrinh`, `BangChung`. Đây là phần Graph RAG nâng cao, không thuộc MVP.

---

## Cấu trúc thư mục

```
buoi_13/
├── data/                              # Dữ liệu gốc (KHÔNG sửa)
│   ├── risk_profiles_seed.csv
│   ├── controls_seed.csv
│   ├── risk_events_seed.csv
│   ├── relationships_seed.csv
│   ├── README.md
│   └── SOURCE.md
│
├── scripts/                           # Pipeline chuẩn hoá + load
│   ├── inspect_data.py                # Bước 1
│   ├── build_entities.py              # Bước 2
│   ├── build_wiki.py                  # Bước 3
│   ├── validate_wiki.py               # Bước 4
│   └── load_neo4j.py                  # Bước 5 (tuỳ chọn)
│
├── outputs/
│   ├── entities.csv                   ← Bước 2
│   ├── relations.csv                  ← Bước 2
│   └── wiki_validation_report.md     ← Bước 4
│
├── wiki/
│   ├── Home.md                        ← Bước 3
│   ├── risks/      (RR-001.md … RR-012.md + README.md)
│   ├── controls/   (KS-001.md … KS-010.md + README.md)
│   └── events/     (SK-001.md … SK-012.md + README.md)
│
├── cypher/
│   ├── schema.cypher                  ← Bước 5 (Neo4j schema MVP)
│   ├── demo_queries.cypher            ← Bước 5 (Cypher minh hoạ)
│   └── ho_so_rui_ro_schema.cypher     # Schema rộng (Graph RAG nâng cao) — KHÔNG dùng cho MVP
│
├── requirements.txt
├── .env.example
└── README.md                          ← file này
```

---

## Thứ tự chạy project

```bash
# 0. Cài Python (>= 3.9). Các script Wiki dùng stdlib — không cần pip install gì.

# 1. Kiểm tra 4 file CSV gốc (row count, cột, FK, null, duplicates).
python3 scripts/inspect_data.py

# 2. Chuẩn hoá → outputs/entities.csv + outputs/relations.csv.
python3 scripts/build_entities.py

# 3. Sinh wiki/ từ outputs/*.csv (mở Home.md trong Obsidian để duyệt).
python3 scripts/build_wiki.py

# 4. Kiểm thử wiki → outputs/wiki_validation_report.md.
python3 scripts/validate_wiki.py

# 5. (Tuỳ chọn) Load dữ liệu vào Neo4j — KHÔNG bắt buộc cho Wiki.
pip install -r requirements.txt
cp .env.example .env          # rồi sửa .env với thông tin thật
python3 scripts/load_neo4j.py
```

> **Tất cả script đều idempotent** — chạy lại nhiều lần vẫn ra cùng kết quả; chỉ `build_wiki.py` và `validate_wiki.py` ghi đè file đầu ra.

---

## Mô tả từng script

### `scripts/inspect_data.py` — Bước 1
Đọc 4 file CSV, in ra:
- Số dòng, tên cột, khóa chính/duy nhất
- Khóa ngoại `risk_id → risk_profile.id`
- Phân bố `relationship_type` + tiền tố `source_id`/`target_id`
- Đếm null/empty theo cột
- Phát hiện duplicate khóa hoặc duplicate dòng
- Tính toàn vẹn tham chiếu (FK check)

### `scripts/build_entities.py` — Bước 2
Chuẩn hoá 4 file CSV thành **2 file wide**:
- `outputs/entities.csv` (34 dòng, 23 cột): schema tối thiểu `{id, type, name, description, source_file, data_origin, verification_status}` + các cột nghiệp vụ riêng từng loại. Không suy đoán tên đơn vị/vai trò.
- `outputs/relations.csv` (22 dòng, 8 cột): giữ nguyên schema từ `relationships_seed.csv`.

### `scripts/build_wiki.py` — Bước 3
Sinh Obsidian-flavored wiki:
- 1 `Home.md` (thống kê + đường đi minh hoạ)
- 12 `risks/RR-XXX.md`, 10 `controls/KS-XXX.md`, 12 `events/SK-XXX.md`
- 3 `README.md` index (danh sách từng loại)
- Wikilink dạng `[[ID|Friendly Name]]` — Obsidian resolve theo basename (mặc định)
- YAML frontmatter tối thiểu `id/type/verification_status/data_origin`, có thêm các trường nghiệp vụ
- Không tự bịa quan hệ — mọi wikilink đều dựng từ `relations.csv`
- Trang "rủi ro trần" (RR-011, RR-012) hiển thị blockquote trung thực thay vì tự thêm kiểm soát

### `scripts/validate_wiki.py` — Bước 4
Sinh `outputs/wiki_validation_report.md` kiểm tra 9 điều kiện:
1. Tổng file Markdown
2. Tổng wikilink
3. Wikilink trỏ tới trang không tồn tại
4. Entity trùng ID
5. Trang có ID không có trong `entities.csv`
6. Relation có source/target không tồn tại
7. RuiRo không có KiemSoat
8. RuiRo không có SuKienRuiRo
9. Trang orphan (không có wikilink ra trang khác)

Cuối báo cáo có **phân loại lỗi**: lỗi chương trình (vá được) vs lỗi / đặc điểm dữ liệu (phản ánh đúng nguồn).

### `scripts/load_neo4j.py` — Bước 5 (tuỳ chọn)
Load `outputs/*.csv` vào Neo4j:
- Đọc `.env` (URI / USER / PASSWORD / DATABASE) — không hard-code
- Áp dụng `cypher/schema.cypher` (3 constraint + 6 index)
- `MERGE` theo `id` để idempotent, parameterized Cypher (không string-interpolate)
- Nếu Neo4j chưa cài / chưa chạy / thiếu `.env`: báo lý do rõ ràng và **không đụng** tới `wiki/` hay `outputs/`

---

## Cấu hình Neo4j (`.env`)

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j
```

Lấy từ Neo4j Browser khi tạo project. **Không commit `.env` vào git.**

---

## Truy vấn Cypher kiểm tra nhanh (sau khi load)

Xem chi tiết trong `cypher/demo_queries.cypher`. Một số truy vấn hay dùng:

```cypher
-- Đếm node/edge theo nhãn/loại
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS so_node ORDER BY so_node DESC;
MATCH ()-[r]->() RETURN type(r) AS loai, count(*) AS so_edge ORDER BY so_edge DESC;

-- Rủi ro không có kiểm soát (sẽ trả về RR-011, RR-012 trong dữ liệu hiện tại)
MATCH (rr:RuiRo)
WHERE NOT EXISTS { MATCH (:KiemSoat)-[:MITIGATES]->(rr) }
RETURN rr.id, rr.name, rr.category ORDER BY rr.id;
```

---

## Quan sát dữ liệu hiện tại (từ inspect + validate)

- 12 RuiRo nhưng chỉ **10 edge `MITIGATES`** → `RR-011` và `RR-012` hiện **chưa có kiểm soát** trong bộ mô phỏng (phản ánh đúng `relationships_seed.csv`).
- 22 edge đều có `verification_status = 'VERIFIED'` (do dữ liệu mô phỏng) — truy vấn F (chưa VERIFIED) sẽ trả về 0 dòng.
- `loss_amount_vnd` = 0 ở 6/12 sự kiện — KHÔNG dùng giá trị này cho báo cáo nghiệp vụ / kiểm toán.

---

## Nguồn gốc dữ liệu mô phỏng

Xem `data/README.md` và `data/SOURCE.md`. Tất cả bản ghi có `data_origin = 'SYNTHETIC'` chỉ trong phạm vi bài lab này; dữ liệu thật cần được xác minh (`verification_status = 'VERIFIED'`) trước khi dùng cho nghiệp vụ.
