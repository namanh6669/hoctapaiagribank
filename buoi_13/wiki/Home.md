---
title: Wiki Risk Graph — Trang chủ
type: Home
---

# Wiki Risk Graph

Wiki tri thức cho **Wiki Risk Graph MVP** — được sinh tự động từ `outputs/entities.csv` + `outputs/relations.csv`.

## Thống kê

- **Tổng node:** 34
  - `RuiRo`: 12
  - `KiemSoat`: 10
  - `SuKienRuiRo`: 12
- **Tổng edge:** 22
  - `OBSERVED_AS`: 12
  - `MITIGATES`: 10

## Danh sách

- [[risks/README|Danh sách rủi ro (12)]]
- [[controls/README|Danh sách kiểm soát (10)]]
- [[events/README|Danh sách sự kiện rủi ro (12)]]

## Đường đi minh họa

Một đường đi đầy đủ **KiemSoat → RuiRo → SuKienRuiRo** có trong dữ liệu:

[[KS-001]] → [[RR-001]] → [[SK-001]]

- **KS-001** — kiểm soát liên quan (mở trang để xem tên).
- **RR-001** — Giao dịch chuyển tiền bị hạch toán sai
- **SK-001** — sự kiện quan sát được.

## Cam kết dữ liệu

- Wiki này được sinh tự động — không tự bịa quan hệ.
- Không suy đoán tên đơn vị/vai trò từ mã (`owner_unit_id`, `owner_role_id`).
- Mọi `verification_status` giữ nguyên giá trị gốc.

