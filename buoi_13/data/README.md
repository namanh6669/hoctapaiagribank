# Bộ dữ liệu khởi động cho Buổi 13

## Phân loại dữ liệu

| Thư mục | Nội dung | Trạng thái sử dụng |
| --- | --- | --- |
| `raw/` | Tệp public gốc khi cần bổ sung trong tương lai | Hiện để trống để tránh lưu dữ liệu không dùng. |
| `processed/` | Tệp chuẩn hoá từ `raw/` | Có CSV phản hồi ngân hàng tiếng Việt UTS2017_Bank. |
| `synthetic/` | Dữ liệu mô phỏng phục vụ lab | Có thể nạp ngay vào Neo4j; không phải dữ liệu của Agribank hay sự kiện thực tế. |

## Dữ liệu chạy ngay

`synthetic/` chứa 12 hồ sơ rủi ro, 12 sự kiện mô phỏng, 10 kiểm soát và các cạnh liên kết. Mọi bản ghi đều có `data_origin=SYNTHETIC` và `verification_status=VERIFIED` **chỉ trong phạm vi bài lab**.

Thứ tự dùng đề xuất:

1. Nạp `risk_profiles_seed.csv`, `controls_seed.csv`, `risk_events_seed.csv` thành node.
2. Nạp `relationships_seed.csv` thành cạnh theo cột `relationship_type`.
3. Dùng các tệp này để kiểm tra Bước 5–7 trong `../buoi_13.md`.
4. Khi đã xử lý nguồn public hoặc có dữ liệu nội bộ được phê duyệt, thay thế từng phần dữ liệu mô phỏng bằng dữ liệu có bằng chứng.

Không dùng trường `loss_amount_vnd` mô phỏng cho báo cáo nghiệp vụ hay kết luận kiểm toán.

## Dữ liệu tiếng Việt: UTS2017_Bank

- `processed/uts2017_bank_classification_train_vi.csv`: 1.977 phản hồi ngân hàng tiếng Việt.
- `processed/uts2017_bank_classification_test_vi.csv`: 494 phản hồi kiểm thử.
- Cột: `text`, `label`. `label` là nhóm khía cạnh dịch vụ ngân hàng, không phải nhóm rủi ro Basel.

Dùng `text` để làm nguồn `BangChung`/khiếu nại và dùng LLM để đề xuất `RuiRo` hoặc `QuyTrinh` liên quan. Không tạo `SuKienRuiRo` ở trạng thái `VERIFIED` chỉ từ một phản hồi khách hàng; phải để `PROPOSED` cho đến khi có nguồn kiểm chứng khác.
