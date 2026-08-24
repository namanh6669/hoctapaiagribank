# Bu�i 15 — Security Audit Report (RBAC)

_Sinh tự động bởi `scripts/security_audit.py` vào 2026-08-24 19:47:21._

## Tổng quan
- Tổng số test case: **6**
- ✅ PASS (không leak): **5**
- � FAIL (rò rỉ dữ liệu): **0**
- ⚠️ WARN (authorized không tìm thấy): **1**

## Bảng tóm tắt

| # | Test case | Category | Status | Leak chunks | Auth hit? |
|---:|---|---|---|---:|:---:|
| 1 | `T01_HR_pure_via_Guest` | HR | ✅ PASS | 0 | ✓ |
| 2 | `T02_Risk_via_Guest` | Risk | ✅ PASS | 0 | ✓ |
| 3 | `T03_Risk_an_toan_von_via_Guest` | Risk | ✅ PASS | 0 | ✓ |
| 4 | `T04_HR_via_Employee_blocked` | HR | ✅ PASS | 0 | ✓ |
| 5 | `T05_Mixed_doc_Admin_only` | Mixed | ✅ PASS | 0 | ✓ |
| 6 | `T06_Multiselect_union_can_see_HR` | HR | ⚠️ WARN | 0 | ✗ |

## Chi tiết từng test case

### 1. `T01_HR_pure_via_Guest`

- **Query:** `bổ nhiệm thành viên hội đồng quản trị`
- **Target doc:** `166170` — _Luật Các tổ chức tín dụng số 32/2024/QH15_
- **Category:** HR
- **Unauthorized roles:** `['Guest']`
- **Authorized roles:** `['Admin', 'HR_Manager']`

#### 🔒 Search với unauthorized roles → 10 hits

✅ **Bằng chứng không leak:** trong top-K không có chunk nào thuộc target doc `166170`.

Top-3 hits:

| rank | chunk_id (8 chars) | doc | label |
|---:|---|---|---|
| 1 | `c0ab003b…` | `30402` | General |
| 2 | `36c3843c…` | `30402` | General |
| 3 | `4846c4b0…` | `30402` | General |

#### 🔓 Search với authorized roles → 10 hits

✅ **Authorized tìm thấy target doc** với `rrf_score=0.0164`.

**Notes:**
- ✓ Không leak: doc 166170 bị ẩn khỏi role ['Guest']; authorized roles ['Admin', 'HR_Manager'] thấy doc với rrf_score=0.0164

### 2. `T02_Risk_via_Guest`

- **Query:** `giao nhận bảo quản vận chuyển tiền mặt tài sản quý`
- **Target doc:** `44209` — _Thông tư 01/2014/TT-NHNN Quy định giao nhận, bảo quản, vận chuyển_
- **Category:** Risk
- **Unauthorized roles:** `['Guest']`
- **Authorized roles:** `['Admin', 'Risk_Officer', 'Employee']`

#### 🔒 Search với unauthorized roles → 10 hits

✅ **Bằng chứng không leak:** trong top-K không có chunk nào thuộc target doc `44209`.

Top-3 hits:

| rank | chunk_id (8 chars) | doc | label |
|---:|---|---|---|
| 1 | `1214ab6d…` | `30402` | General |
| 2 | `36c3843c…` | `30402` | General |
| 3 | `00aa1333…` | `30402` | General |

#### 🔓 Search với authorized roles → 10 hits

✅ **Authorized tìm thấy target doc** với `rrf_score=0.0320`.

**Notes:**
- ✓ Không leak: doc 44209 bị ẩn khỏi role ['Guest']; authorized roles ['Admin', 'Risk_Officer', 'Employee'] thấy doc với rrf_score=0.0320

### 3. `T03_Risk_an_toan_von_via_Guest`

- **Query:** `tỷ lệ an toàn vốn ngân hàng`
- **Target doc:** `117310` — _Thông tư 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn_
- **Category:** Risk
- **Unauthorized roles:** `['Guest']`
- **Authorized roles:** `['Admin', 'Risk_Officer']`

#### 🔒 Search với unauthorized roles → 10 hits

✅ **Bằng chứng không leak:** trong top-K không có chunk nào thuộc target doc `117310`.

Top-3 hits:

| rank | chunk_id (8 chars) | doc | label |
|---:|---|---|---|
| 1 | `36c3843c…` | `30402` | General |
| 2 | `4846c4b0…` | `30402` | General |
| 3 | `1214ab6d…` | `30402` | General |

#### 🔓 Search với authorized roles → 10 hits

✅ **Authorized tìm thấy target doc** với `rrf_score=0.0295`.

**Notes:**
- ✓ Không leak: doc 117310 bị ẩn khỏi role ['Guest']; authorized roles ['Admin', 'Risk_Officer'] thấy doc với rrf_score=0.0295

### 4. `T04_HR_via_Employee_blocked`

- **Query:** `chuẩn mực kiểm toán nội bộ`
- **Target doc:** `150974` — _Thông tư 08/2021/TT-BTC chuẩn mực kiểm toán nội bộ_
- **Category:** HR
- **Unauthorized roles:** `['Employee']`
- **Authorized roles:** `['Admin', 'HR_Manager']`

#### 🔒 Search với unauthorized roles → 10 hits

✅ **Bằng chứng không leak:** trong top-K không có chunk nào thuộc target doc `150974`.

Top-3 hits:

| rank | chunk_id (8 chars) | doc | label |
|---:|---|---|---|
| 1 | `4964e2e0…` | `30402` | General |
| 2 | `cbac09de…` | `38128` | General |
| 3 | `8a6b38b0…` | `38128` | General |

#### 🔓 Search với authorized roles → 10 hits

✅ **Authorized tìm thấy target doc** với `rrf_score=0.0320`.

**Notes:**
- ✓ Không leak: doc 150974 bị ẩn khỏi role ['Employee']; authorized roles ['Admin', 'HR_Manager'] thấy doc với rrf_score=0.0320

### 5. `T05_Mixed_doc_Admin_only`

- **Query:** `luật các tổ chức tín dụng`
- **Target doc:** `166170` — _Luật Các tổ chức tín dụng số 32/2024/QH15 (mixed HR+Risk)_
- **Category:** Mixed
- **Unauthorized roles:** `['HR_Manager', 'Risk_Officer', 'Employee', 'Guest']`
- **Authorized roles:** `['Admin']`

#### 🔒 Search với unauthorized roles → 10 hits

✅ **Bằng chứng không leak:** trong top-K không có chunk nào thuộc target doc `166170`.

Top-3 hits:

| rank | chunk_id (8 chars) | doc | label |
|---:|---|---|---|
| 1 | `96edab3e…` | `164719` | Risk |
| 2 | `c353b5aa…` | `44209` | Risk |
| 3 | `1a3d9079…` | `117310` | Risk |

#### 🔓 Search với authorized roles → 10 hits

✅ **Authorized tìm thấy target doc** với `rrf_score=0.0308`.

**Notes:**
- ✓ Không leak: doc 166170 bị ẩn khỏi role ['HR_Manager', 'Risk_Officer', 'Employee', 'Guest']; authorized roles ['Admin'] thấy doc với rrf_score=0.0308

### 6. `T06_Multiselect_union_can_see_HR`

- **Query:** `bổ nhiệm`
- **Target doc:** `166170` — _Luật Các tổ chức tín dụng (HR chunks về bổ nhiệm HĐQT)_
- **Category:** HR
- **Unauthorized roles:** `['Guest']`
- **Authorized roles:** `['HR_Manager', 'Risk_Officer']`

#### 🔒 Search với unauthorized roles → 8 hits

✅ **Bằng chứng không leak:** trong top-K không có chunk nào thuộc target doc `166170`.

Top-3 hits:

| rank | chunk_id (8 chars) | doc | label |
|---:|---|---|---|
| 1 | `276c371b…` | `30402` | General |
| 2 | `4964e2e0…` | `30402` | General |
| 3 | `85f34c7a…` | `30402` | General |

#### 🔓 Search với authorized roles → 10 hits

⚠️ Authorized roles **không** tìm thấy target doc trong top-K — không phải leak, nhưng đáng kiểm tra query.

**Notes:**
- ⚠️ Authorized roles ['HR_Manager', 'Risk_Officer'] không tìm thấy doc 166170 trong top-10 — kiểm tra query hoặc điểm tương đồng quá thấp.

## Kết luận

⚠️ **ĐẠT có điều kiện.** Không có rò rỉ (0 FAIL), nhưng 1 test case có authorized roles không tìm thấy doc (có thể do query quá cụ thể hoặc điểm thấp — cần xem xét query).

---

Cách tái-chạy: `python ../buoi_14/.venv/bin/python3.12 scripts/security_audit.py`. Sửa query hoặc target_document_id trong `TEST_CASES` để mở rộng phạm vi test.