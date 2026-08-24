# Wiki Risk Graph — Validation Report

_Báo cáo này được sinh tự động bởi `scripts/validate_wiki.py`, đọc `outputs/entities.csv`, `outputs/relations.csv` và `wiki/**/*.md`._

## Tóm tắt

- Tổng file Markdown: **38**
- Tổng entity trong CSV: **34** (12 RuiRo, 10 KiemSoat, 12 SuKienRuiRo)
- Tổng relation trong CSV: **22**
- Tổng wikilink trong body các trang: **84**

### 1. Tổng số file Markdown

✅ **OK**
- Đếm được **38** file (bao gồm `Home.md`, README và các trang entity).

### 2. Tổng số wikilink

✅ **OK**
- Tổng `[[...]]` trong **body** các trang: **84**.

### 3. Wikilink trỏ tới trang không tồn tại

✅ **OK**
- Mọi `[[target]]` đều resolve được trong `wiki/`.

### 4. Entity bị trùng ID trong `entities.csv`

✅ **OK**

### 5. Trang wiki có ID không tồn tại trong `entities.csv`

✅ **OK**
- Đã kiểm tra **34** trang có frontmatter `id`.

### 6. Relation có source/target không tồn tại

✅ **OK**

### 7. RuiRo không có KiemSoat nào

⚠️  **Dữ liệu:** Có **2** RuiRo chưa có kiểm soát trong `relations.csv` (phản ánh đúng nguồn — không tự vá):

- `RR-011`
- `RR-012`

### 8. RuiRo không có SuKienRuiRo nào

✅ **OK**

### 9. Trang không có liên kết ra trang khác (orphan)

✅ **OK**

## Phân loại lỗi

### Lỗi chương trình (có thể vá bằng cách sửa `build_wiki.py`)

- _(không có)_

### Lỗi / đặc điểm dữ liệu (phản ánh đúng `data/*.csv`, KHÔNG tự vá)

- #7 RuiRo không có kiểm soát — **2**: RR-011, RR-012.
