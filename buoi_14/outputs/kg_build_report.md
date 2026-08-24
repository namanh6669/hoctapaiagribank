# Buổi 14 — Mini Knowledge Graph Build Report

_Sinh tự động bằng `scripts/load_mini_kg.py`._

## Thống kê (in-memory + graph sau push nếu có)

- VanBan      : **30**
- DieuKhoan   : **1463**
- Entity      : **63**
- Edges (all) : **3062** (CONTAINS=1463 · NEXT=1433 · rels=166)
- Skipped do kind/rtype mismatch: **0**
- Remap `so_ky_hieu → metadata.id`: **29** lần


### Edges theo type (Source CSV = trước collapse; Graph = sau MERGE/dedup)

| Type | Source CSV rows | Graph in-memory (post-collapse) |
|---|---:|---:|
| THAM_CHIEU | 17 | 10 |
| THAY_THE_BOI | 3 | 3 |
| SUA_DOI_BO_SUNG | 9 | 9 |
| BAN_HANH_BOI | 30 | 30 |
| KY_BOI | 30 | 30 |
| THUOC_LINH_VUC | 23 | 23 |
| AP_DUNG_CHO | 61 | 61 |

### Orphan check

- VanBan không có CONTAINS outgoing : **0** → []
- DieuKhoan không có VanBan cha : **0**


### Entity có nhiều biến thể tên (`name_alt` được lưu)

| Entity id | name | name_alt |
|---|---|---|
| E0036 | Tổ chức tín dụng | tổ chức tín dụng |
| E0028 | Kiểm toán viên hành nghề | kiểm toán viên hành nghề |
| E0012 | Ngân hàng hợp tác xã | ngân hàng hợp tác xã |
| E0032 | Quỹ tín dụng nhân dân | quỹ tín dụng nhân dân |
| E0049 | Kiểm toán | kiểm toán |
| E0019 | Cơ quan nhà nước | cơ quan nhà nước |
| E0020 | Doanh nghiệp | doanh nghiệp |
| E0031 | Ngân hàng thương mại | ngân hàng thương mại |


## Cypher files

- `buoi_14/cypher/schema.cypher` — viết tay (ontology VanBan/DieuKhoan)
- `buoi_14/cypher/load_data.cypher` — auto, MERGE + UNWIND, idempotent
- `buoi_14/cypher/load_data_params.json` — params cho Cypher Browser `:param`
- `buoi_14/cypher/demo_queries.cypher` — viết tay ($params)


## Push status

- **SKIP — không có NEO4J_URI env, không có NEO4J_PASSWORD env**


## An toàn Neo4j

- KHÔNG chạy `MATCH (n) DETACH DELETE n` trong script.
- Mọi node/edge mang `lab_session = 'buoi_14'` để phạm vi hoá.
- Cleanup nếu cần chỉ chạy khi `--clean-previous` (hoặc `CLEAN_PREVIOUS=1`), giới hạn ở `lab_session='buoi_14'`.
- Tất cả Cypher dùng `$params` — KHÔNG string-interpolate, KHÔNG hardcode password.
- Password chỉ đọc qua `os.getenv('NEO4J_PASSWORD')` sau `python-dotenv` load `.env`.


## Hạn chế & ghi chú

- `Chunks có text rỗng` bị loại (không tạo :DieuKhoan cho chúng).
- `:NEXT` chỉ chain trong CÙNG VanBan, theo thứ tự rows trong CSV.
- Quan hệ Document↔Document từ relationships.csv giữ NGUYÊN HƯỚNG như CSV ghi.
- 1 source_id có thể có nhiều target qua nhiều method khác nhau; chỉ merge nếu `(u, v, type)` trùng — KHÔNG trộn evidence các lần.
