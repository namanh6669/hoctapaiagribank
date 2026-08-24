# Buổi 14 — Retrieval so sánh: BM25 · Dense · Hybrid (RRF) · Rerank

_Sinh tự động bằng `scripts/run_examples.py`, top_k=5, candidate_k=20_.

- **BM25** — lexical trên toàn chunk corpus (rank_bm25).
- **Dense** — `intfloat/multilingual-e5-base`, L2-normalize, query/passage theo prefix E5.
- **Hybrid** — Reciprocal Rank Fusion (k=60) của BM25 + Dense trên cùng corpus.
- **Rerank** — Cross-Encoder `BAAI/bge-reranker-v2-m3` rerank top-20 candidate của Hybrid → top-5.

Citation lấy từ metadata thật (`title`, `so_ky_hieu`, `article`, `chunk_id`); không bịa ở bất kỳ method nào.


## Câu có mã/số hiệu cụ thể

**Query:** `Quy định tại Điều 5 Thông tư 01/2014/TT-NHNN về giao nhận, bảo quản tiền mặt`

### BM25 RESULTS

| rank | score | chunk_id | citation | text (≤240 chars) |
|---:|---:|---|---|---|
| 1 | 24.4757 | `c32bb8d7-b490-56d0-937a-bb1ad606ace8` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | c32bb8d7-b490-56d0-937a-bb1ad606ace8] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM ------- CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc --------------- Số: 01/2014/TT-NHNN Hà Nội, ngày 06 tháng 01 năm 2014 THÔNG TƯ Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản … |
| 2 | 20.8934 | `a76e9382-e564-501c-a5e0-daad6a05da62` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 26 | a76e9382-e564-501c-a5e0-daad6a05da62] | Điều 26. Quy định ủy quyền của các thành viên tham gia quản lý tiền mặt, tài sản quý, giấy tờ có giá và kho tiền 1. Quy định ủy quyền của Giám đốc: a) Giám đốc được ủy quyền bằng văn bản cho một Phó Giám đốc thực hiện nhiệm vụ quản lý tiền … |
| 3 | 20.4311 | `13ae2f57-5927-5d2e-9b83-c26fb19abcde` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 14 | 13ae2f57-5927-5d2e-9b83-c26fb19abcde] | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng 1. Việc giao nhận tiền mặt giữa Sở Giao dịch, Ngân hàng Nhà nước chi nhánh, tổ chức tín dụng, chi nhánh ngân hàng nước ngoài với Kho bạc Nhà … |
| 4 | 20.0163 | `c5eb1cfa-0d46-5e54-966e-b3e7a55d2563` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 11 | c5eb1cfa-0d46-5e54-966e-b3e7a55d2563] | Điều 11. Giao nhận tiền mặt trong ngành Ngân hàng 1. Giao nhận tiền mặt theo bó tiền đủ 10 thếp, nguyên niêm phong hoặc túi tiền nguyên niêm phong kẹp chì trong các trường hợp: a) Giao nhận tiền mặt trong nội bộ Sở Giao dịch, Ngân hàng Nhà … |
| 5 | 19.3769 | `d0d0fc6f-3725-589e-98ba-c5cee4ed831b` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 12 | d0d0fc6f-3725-589e-98ba-c5cee4ed831b] | Điều 12. Kiểm đếm tiền mặt giao nhận trong ngành Ngân hàng 1. Sở Giao dịch, Ngân hàng Nhà nước chi nhánh nhận tiền trong trường hợp quy định tại điểm b Khoản 1 Điều 11 tổ chức kiểm đếm tờ (miếng) số tiền đã nhận phải thành lập Hội đồng kiểm… |

### DENSE RESULTS

| rank | score | chunk_id | citation | text (≤240 chars) |
|---:|---:|---|---|---|
| 1 | 0.9015 | `c32bb8d7-b490-56d0-937a-bb1ad606ace8` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | c32bb8d7-b490-56d0-937a-bb1ad606ace8] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM ------- CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc --------------- Số: 01/2014/TT-NHNN Hà Nội, ngày 06 tháng 01 năm 2014 THÔNG TƯ Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản … |
| 2 | 0.8828 | `2f82ddc3-b9f9-5f4e-8520-f466001cd970` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 72 | 2f82ddc3-b9f9-5f4e-8520-f466001cd970] | Điều 72. Hiệu lực thi hành 1. Thông tư này có hiệu lực thi hành kể từ ngày 20/02/2014. 2. Kể từ ngày Thông tư này có hiệu lực, các văn bản sau hết hiệu lực thi hành: a) Quyết định số 60/2006/QĐ-NHNN ngày 27/12/2006 của Thống đốc Ngân hàng N… |
| 3 | 0.8776 | `bf62e928-2798-5ff0-b4e1-48839f790f85` | [Thông tư số 37/2014/TT-NHNN Quy định việc thiết kế mẫu tiền, chế bản và quản lý in, đúc tiền Việt Nam | bf62e928-2798-5ff0-b4e1-48839f790f85] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 37/2014/TT-NHNN Hà Nội, Ngày 26 tháng 11 năm 2014 THÔNG TƯ Q uy định việc thiết kế mẫu tiền, chế bản và quản lý in, đúc tiền Việt Nam ___________… |
| 4 | 0.8741 | `13ae2f57-5927-5d2e-9b83-c26fb19abcde` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 14 | 13ae2f57-5927-5d2e-9b83-c26fb19abcde] | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng 1. Việc giao nhận tiền mặt giữa Sở Giao dịch, Ngân hàng Nhà nước chi nhánh, tổ chức tín dụng, chi nhánh ngân hàng nước ngoài với Kho bạc Nhà … |
| 5 | 0.8704 | `87d1ddce-2d52-557f-8e27-24a6f4e8f91a` | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 15 | 87d1ddce-2d52-557f-8e27-24a6f4e8f91a] | Điều 15. Sắp xếp, bảo quản tài sản tại quầy giao dịch và trong kho tiền 1. Hết giờ làm việc hàng ngày, toàn bộ tiền mặt, tài sản quý, giấy tờ có giá phải được bảo quản trong kho tiền. Giám đốc Sở Giao dịch, Giám đốc Ngân hàng Nhà nước chi n… |

### HYBRID RESULTS (RRF)

| final_rank | chunk_id | bm25_rank | dense_rank | rrf_score | citation | text (≤240 chars) |
|---:|---|---:|---:|---:|---|---|
| 1 | `c32bb8d7-b490-56d0-937a-bb1ad606ace8` | 1 | 1 | 0.032787 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | c32bb8d7-b490-56d0-937a-bb1ad606ace8] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM ------- CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc --------------- Số: 01/2014/TT-NHNN Hà Nội, ngày 06 tháng 01 năm 2014 THÔNG TƯ Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản … |
| 2 | `13ae2f57-5927-5d2e-9b83-c26fb19abcde` | 3 | 4 | 0.031498 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 14 | 13ae2f57-5927-5d2e-9b83-c26fb19abcde] | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng 1. Việc giao nhận tiền mặt giữa Sở Giao dịch, Ngân hàng Nhà nước chi nhánh, tổ chức tín dụng, chi nhánh ngân hàng nước ngoài với Kho bạc Nhà … |
| 3 | `c5eb1cfa-0d46-5e54-966e-b3e7a55d2563` | 4 | 8 | 0.030331 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 11 | c5eb1cfa-0d46-5e54-966e-b3e7a55d2563] | Điều 11. Giao nhận tiền mặt trong ngành Ngân hàng 1. Giao nhận tiền mặt theo bó tiền đủ 10 thếp, nguyên niêm phong hoặc túi tiền nguyên niêm phong kẹp chì trong các trường hợp: a) Giao nhận tiền mặt trong nội bộ Sở Giao dịch, Ngân hàng Nhà … |
| 4 | `d0d0fc6f-3725-589e-98ba-c5cee4ed831b` | 5 | 15 | 0.028718 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 12 | d0d0fc6f-3725-589e-98ba-c5cee4ed831b] | Điều 12. Kiểm đếm tiền mặt giao nhận trong ngành Ngân hàng 1. Sở Giao dịch, Ngân hàng Nhà nước chi nhánh nhận tiền trong trường hợp quy định tại điểm b Khoản 1 Điều 11 tổ chức kiểm đếm tờ (miếng) số tiền đã nhận phải thành lập Hội đồng kiểm… |
| 5 | `87d1ddce-2d52-557f-8e27-24a6f4e8f91a` | 16 | 5 | 0.028543 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 15 | 87d1ddce-2d52-557f-8e27-24a6f4e8f91a] | Điều 15. Sắp xếp, bảo quản tài sản tại quầy giao dịch và trong kho tiền 1. Hết giờ làm việc hàng ngày, toàn bộ tiền mặt, tài sản quý, giấy tờ có giá phải được bảo quản trong kho tiền. Giám đốc Sở Giao dịch, Giám đốc Ngân hàng Nhà nước chi n… |

### HYBRID → RERANK RESULTS  _(rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3`)_

| final_rank | chunk_id | hybrid_rank | rerank_score | Δ vs hybrid | citation | text (≤240 chars) |
|---:|---|---:|---:|---:|---|---|
| 1 | `c32bb8d7-b490-56d0-937a-bb1ad606ace8` | 1 | +0.9973 | +0 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | c32bb8d7-b490-56d0-937a-bb1ad606ace8] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM ------- CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc --------------- Số: 01/2014/TT-NHNN Hà Nội, ngày 06 tháng 01 năm 2014 THÔNG TƯ Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản … |
| 2 | `ee9261eb-6544-56a8-973c-e40996622b68` | 12 | +0.9831 | +10 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 1 | ee9261eb-6544-56a8-973c-e40996622b68] | Điều 1. Phạm vi điều chỉnh 1. Thông tư này quy định việc giao nhận, bảo quản, vận chuyển; kiểm tra, kiểm kê, bàn giao, xử lý thừa thiếu tiền mặt, tài sản quý, giấy tờ có giá trong ngành Ngân hàng; việc thu, chi tiền mặt giữa Ngân hàng Nhà n… |
| 3 | `2f82ddc3-b9f9-5f4e-8520-f466001cd970` | 7 | +0.9593 | +4 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 72 | 2f82ddc3-b9f9-5f4e-8520-f466001cd970] | Điều 72. Hiệu lực thi hành 1. Thông tư này có hiệu lực thi hành kể từ ngày 20/02/2014. 2. Kể từ ngày Thông tư này có hiệu lực, các văn bản sau hết hiệu lực thi hành: a) Quyết định số 60/2006/QĐ-NHNN ngày 27/12/2006 của Thống đốc Ngân hàng N… |
| 4 | `13ae2f57-5927-5d2e-9b83-c26fb19abcde` | 2 | +0.9379 | -2 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 14 | 13ae2f57-5927-5d2e-9b83-c26fb19abcde] | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng 1. Việc giao nhận tiền mặt giữa Sở Giao dịch, Ngân hàng Nhà nước chi nhánh, tổ chức tín dụng, chi nhánh ngân hàng nước ngoài với Kho bạc Nhà … |
| 5 | `87d1ddce-2d52-557f-8e27-24a6f4e8f91a` | 5 | +0.9273 | +0 | [Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 15 | 87d1ddce-2d52-557f-8e27-24a6f4e8f91a] | Điều 15. Sắp xếp, bảo quản tài sản tại quầy giao dịch và trong kho tiền 1. Hết giờ làm việc hàng ngày, toàn bộ tiền mặt, tài sản quý, giấy tờ có giá phải được bảo quản trong kho tiền. Giám đốc Sở Giao dịch, Giám đốc Ngân hàng Nhà nước chi n… |

**Phân tích nhanh:**

- Rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3`
- BEFORE top-5 → AFTER top-5: 2 promoted, 1 demoted, 2 mới vào top, 2 bị đẩy ra.


## Câu diễn đạt semantic (không mã)

**Query:** `Điều kiện để cấp giấy phép cho ngân hàng thương mại`

### BM25 RESULTS

| rank | score | chunk_id | citation | text (≤240 chars) |
|---:|---:|---|---|---|
| 1 | 22.9553 | `9bfcf0c0-eb6e-5f73-ac04-ddd90bcbdcb3` | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 22 | 9bfcf0c0-eb6e-5f73-ac04-ddd90bcbdcb3] | Điều 22. Trách nhiệm của Đơn vị đầu mối xử lý hồ sơ đề nghị cấp Giấy phép 1. Làm đầu mối tiếp nhận và thẩm định hồ sơ đề nghị cấp Giấy phép thành lập và hoạt động của ngân hàng thương mại, hồ sơ đề nghị cấp Giấy phép thành lập chi nhánh ngâ… |
| 2 | 22.9349 | `f0ae7d50-ad5b-5b37-aaa3-41d74bfaa728` | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 23 | f0ae7d50-ad5b-5b37-aaa3-41d74bfaa728] | Điều 23. Trách nhiệm của các đơn vị khác thuộc Ngân hàng Nhà nước 1. Ngân hàng Nhà nước chi nhánh: a) Trong thời hạn 30 ngày, kể từ ngày Đơn vị đầu mối xử lý hồ sơ đề nghị cấp Giấy phép quy định tại khoản 7 Điều 3 Thông tư này có văn bản đề… |
| 3 | 22.9349 | `4662e51c-2f6d-5b87-92aa-95d6cbf4e9f8` | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 23 | 4662e51c-2f6d-5b87-92aa-95d6cbf4e9f8] | Điều 23. Trách nhiệm của các đơn vị khác thuộc Ngân hàng Nhà nước 1. Ngân hàng Nhà nước chi nhánh: a) Trong thời hạn 30 ngày, kể từ ngày Đơn vị đầu mối xử lý hồ sơ đề nghị cấp Giấy phép quy định tại khoản 7 Điều 3 Thông tư này có văn bản đề… |
| 4 | 22.4592 | `50c10577-d466-5d91-9364-f35226162f66` | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 22 | 50c10577-d466-5d91-9364-f35226162f66] | Điều 22. Trách nhiệm của Đơn vị đầu mối xử lý hồ sơ đề nghị cấp Giấy phép 1. Làm đầu mối tiếp nhận và thẩm định hồ sơ đề nghị cấp Giấy phép thành lập và hoạt động của ngân hàng thương mại, hồ sơ đề nghị cấp Giấy phép thành lập chi nhánh ngâ… |
| 5 | 22.0423 | `89c2c6ac-044e-5cf8-84d0-0ceaacce0322` | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 25 | 89c2c6ac-044e-5cf8-84d0-0ceaacce0322] | Điều 25. Tổ chức thực hiện Thủ trưởng các đơn vị thuộc Ngân hàng Nhà nước Việt Nam, ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài và các tổ chức, cá nhân có liên quan chịu trách nhiệm thực hiện Thông tư… |

### DENSE RESULTS

| rank | score | chunk_id | citation | text (≤240 chars) |
|---:|---:|---|---|---|
| 1 | 0.8895 | `d2f44828-1df2-5fd2-be37-30c5bd60f7f0` | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 7 | d2f44828-1df2-5fd2-be37-30c5bd60f7f0] | Điều 7. Thủ tục cấp Giấy phép 1. Thủ tục cấp Giấy phép thành lập và hoạt động của ngân hàng thương mại, Giấy phép thành lập chi nhánh ngân hàng nước ngoài như sau: a) Ban trù bị lập hồ sơ đề nghị cấp Giấy phép theo quy định tại Điều 10, Điề… |
| 2 | 0.8862 | `f93f44e9-3b76-5571-8914-40b7ef9ffa65` | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 7 | f93f44e9-3b76-5571-8914-40b7ef9ffa65] | Điều 7. Thủ tục cấp Giấy phép 1. Thủ tục cấp Giấy phép thành lập và hoạt động của ngân hàng thương mại, Giấy phép thành lập chi nhánh ngân hàng nước ngoài như sau: a) Ban trù bị lập hồ sơ đề nghị cấp Giấy phép theo quy định tại Điều 10, Điề… |
| 3 | 0.8836 | `54415e4b-4c82-5822-b409-a162c31ddb15` | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 1 | 54415e4b-4c82-5822-b409-a162c31ddb15] | Điều 1. Phạm vi điều chỉnh Thông tư này quy định về hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài.… |
| 4 | 0.8836 | `fae8c300-675a-53da-94a3-370101aa1da3` | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 1 | fae8c300-675a-53da-94a3-370101aa1da3] | Điều 1. Phạm vi điều chỉnh Thông tư này quy định về hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài.… |
| 5 | 0.8825 | `b261597a-db9e-5a19-b8a3-8cc2061866c4` | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 20 | b261597a-db9e-5a19-b8a3-8cc2061866c4] | Điều 20. Phối hợp cấp Giấy phép 1. Sau khi có văn bản xác nhận hồ sơ đầy đủ và hợp lệ, Ngân hàng Nhà nước có văn bản gửi lấy ý kiến của: a) Ủy ban nhân dân tỉnh, thành phố trực thuộc Trung ương nơi dự kiến đặt trụ sở chính của ngân hàng thư… |

### HYBRID RESULTS (RRF)

| final_rank | chunk_id | bm25_rank | dense_rank | rrf_score | citation | text (≤240 chars) |
|---:|---|---:|---:|---:|---|---|
| 1 | `1b4ade42-0188-5c94-a1db-a579e2ea0f66` | 6 | 7 | 0.030077 | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 19 | 1b4ade42-0188-5c94-a1db-a579e2ea0f66] | Điều 19. Thông báo thông tin về cấp Giấy phép, thông tin về người đại diện pháp luật của ngân hàng thương mại, thông tin về Tổng giám đốc (Giám đốc) chi nhánh ngân hàng nước ngoài, Trưởng văn phòng đại diện nước ngoài cho cơ quan đăng ký ki… |
| 2 | `06cac849-c8a0-5749-aa50-1f20ab77b1a9` | 7 | 8 | 0.029631 | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 19 | 06cac849-c8a0-5749-aa50-1f20ab77b1a9] | Điều 19. Thông báo thông tin về cấp Giấy phép, thông tin về người đại diện pháp luật của ngân hàng thương mại, thông tin về Tổng giám đốc (Giám đốc) chi nhánh ngân hàng nước ngoài, Trưởng văn phòng đại diện nước ngoài cho cơ quan đăng ký ki… |
| 3 | `9bfcf0c0-eb6e-5f73-ac04-ddd90bcbdcb3` | 1 | 19 | 0.029052 | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 22 | 9bfcf0c0-eb6e-5f73-ac04-ddd90bcbdcb3] | Điều 22. Trách nhiệm của Đơn vị đầu mối xử lý hồ sơ đề nghị cấp Giấy phép 1. Làm đầu mối tiếp nhận và thẩm định hồ sơ đề nghị cấp Giấy phép thành lập và hoạt động của ngân hàng thương mại, hồ sơ đề nghị cấp Giấy phép thành lập chi nhánh ngâ… |
| 4 | `17527db5-95b2-561a-87de-414500aa2408` | 10 | 10 | 0.028571 | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 6 | 17527db5-95b2-561a-87de-414500aa2408] | Điều 6. Giấy phép 1. Ngân hàng Nhà nước quy định cụ thể nội dung hoạt động ngân hàng, hoạt động kinh doanh khác của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, nội dung hoạt động của văn phòng đại diện nước ngoài theo mẫu Giấy phé… |
| 5 | `8b240e3c-d57f-5f03-ace6-517961c35aca` | 11 | 11 | 0.028169 | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 6 | 8b240e3c-d57f-5f03-ace6-517961c35aca] | Điều 6. Giấy phép 1. Ngân hàng Nhà nước quy định cụ thể nội dung hoạt động ngân hàng, hoạt động kinh doanh khác của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, nội dung hoạt động của văn phòng đại diện nước ngoài theo mẫu Giấy phé… |

### HYBRID → RERANK RESULTS  _(rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3`)_

| final_rank | chunk_id | hybrid_rank | rerank_score | Δ vs hybrid | citation | text (≤240 chars) |
|---:|---|---:|---:|---:|---|---|
| 1 | `0d796f60-b9e2-52f4-9de1-c524affc7ae0` | 7 | +0.9918 | +6 | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | 0d796f60-b9e2-52f4-9de1-c524affc7ae0] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 56/2024/TT-NHNN Hà Nội, ngày 24 tháng 12 năm 2024 THÔNG TƯ Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân… |
| 2 | `89c2c6ac-044e-5cf8-84d0-0ceaacce0322` | 14 | +0.9802 | +12 | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 25 | 89c2c6ac-044e-5cf8-84d0-0ceaacce0322] | Điều 25. Tổ chức thực hiện Thủ trưởng các đơn vị thuộc Ngân hàng Nhà nước Việt Nam, ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài và các tổ chức, cá nhân có liên quan chịu trách nhiệm thực hiện Thông tư… |
| 3 | `f93f44e9-3b76-5571-8914-40b7ef9ffa65` | 10 | +0.9783 | +7 | [Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (52/V… | Điều 7 | f93f44e9-3b76-5571-8914-40b7ef9ffa65] | Điều 7. Thủ tục cấp Giấy phép 1. Thủ tục cấp Giấy phép thành lập và hoạt động của ngân hàng thương mại, Giấy phép thành lập chi nhánh ngân hàng nước ngoài như sau: a) Ban trù bị lập hồ sơ đề nghị cấp Giấy phép theo quy định tại Điều 10, Điề… |
| 4 | `d2f44828-1df2-5fd2-be37-30c5bd60f7f0` | 8 | +0.9739 | +4 | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 7 | d2f44828-1df2-5fd2-be37-30c5bd60f7f0] | Điều 7. Thủ tục cấp Giấy phép 1. Thủ tục cấp Giấy phép thành lập và hoạt động của ngân hàng thương mại, Giấy phép thành lập chi nhánh ngân hàng nước ngoài như sau: a) Ban trù bị lập hồ sơ đề nghị cấp Giấy phép theo quy định tại Điều 10, Điề… |
| 5 | `17527db5-95b2-561a-87de-414500aa2408` | 4 | +0.9705 | -1 | [Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phò… | Điều 6 | 17527db5-95b2-561a-87de-414500aa2408] | Điều 6. Giấy phép 1. Ngân hàng Nhà nước quy định cụ thể nội dung hoạt động ngân hàng, hoạt động kinh doanh khác của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, nội dung hoạt động của văn phòng đại diện nước ngoài theo mẫu Giấy phé… |

**Phân tích nhanh:**

- Rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3`
- BEFORE top-5 → AFTER top-5: 4 promoted, 1 demoted, 4 mới vào top, 4 bị đẩy ra.
- Top-1 đã đổi: `1b4ade42` (hybrid) → `0d796f60` (rerank).


## Câu kết hợp cả hai

**Query:** `Tỷ lệ an toàn vốn tối thiểu theo Thông tư 41/2016/TT-NHNN được sửa đổi bởi 22/2023/TT-NHNN`

### BM25 RESULTS

| rank | score | chunk_id | citation | text (≤240 chars) |
|---:|---:|---|---|---|
| 1 | 34.9410 | `96edab3e-3305-5b10-a3e3-908a8be49736` | [Thông tư số 22/2023/TT-NHNN Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016 của Thống đốc Ngân hàng… | 96edab3e-3305-5b10-a3e3-908a8be49736] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 22/2023/TT-NHNN Hà Nội, ngày 29 tháng 12 năm 2023 THÔNG TƯ Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016… |
| 2 | 23.2955 | `b65a40cd-3aac-53e5-9be0-97aac512db9d` | [Luật Các tổ chức tín dụng số 32/2024/QH15 | Điều 138 | b65a40cd-3aac-53e5-9be0-97aac512db9d] | Điều 138. Tỷ lệ bảo đảm an toàn 1. Tổ chức tín dụng, chi nhánh ngân hàng nước ngoài phải duy trì các tỷ lệ bảo đảm an toàn sau đây: a) Tỷ lệ khả năng chi trả; b) Tỷ lệ an toàn vốn tối thiểu 08% hoặc tỷ lệ cao hơn theo quy định của Thống đốc… |
| 3 | 22.9108 | `81297ec7-68b0-5e09-a1f9-071d3c3de9f3` | [Thông tư số 62/2025/TT-NHNN Quy định về hệ thống kiểm soát nội bộ của tổ chức tín dụng là hợp tác xã, tổ chức tài chính vi mô | Điều 20 | 81297ec7-68b0-5e09-a1f9-071d3c3de9f3] | Điều 20. Chính sách quản lý rủi ro 1. Chính sách quản lý rủi ro của tổ chức tín dụng do Hội đồng quản trị, Hội đồng thành viên ban hành, sửa đổi, bổ sung. 2. Chính sách quản lý rủi ro bao gồm tối thiểu các nội dung sau đây: a) Khẩu vị rủi r… |
| 4 | 22.7027 | `f4924965-73f4-5fd2-8a24-d634d56ebdb1` | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | Điều 6 | f4924965-73f4-5fd2-8a24-d634d56ebdb1] | Điều 6. Tỷ lệ an toàn vốn 1. Tỷ lệ an toàn vốn (CAR) tính theo đơn vị phần trăm (%) được xác định bằng công thức: Trong đó: - C : Vốn tự có; - RWA : Tổng tài sản tính theo rủi ro tín dụng; - KOR : Vốn yêu cầu cho rủi ro hoạt động; - KMR : V… |
| 5 | 21.1105 | `fb3faec3-dc03-57ef-ac77-3696a3235f2e` | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | Điều 4 | fb3faec3-dc03-57ef-ac77-3696a3235f2e] | Điều 4. Dữ liệu và hệ thống công nghệ thông tin 1. Ngân hàng, chi nhánh ngân hàng nước ngoài phải có dữ liệu đầy đủ và hệ thống công nghệ thông tin phù hợp để tính tỷ lệ an toàn vốn theo quy định tại Thông tư này. 2. Ngân hàng, chi nhánh ng… |

### DENSE RESULTS

| rank | score | chunk_id | citation | text (≤240 chars) |
|---:|---:|---|---|---|
| 1 | 0.8981 | `96edab3e-3305-5b10-a3e3-908a8be49736` | [Thông tư số 22/2023/TT-NHNN Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016 của Thống đốc Ngân hàng… | 96edab3e-3305-5b10-a3e3-908a8be49736] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 22/2023/TT-NHNN Hà Nội, ngày 29 tháng 12 năm 2023 THÔNG TƯ Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016… |
| 2 | 0.8909 | `5b92a756-3b0e-5c54-aaa1-8867edd4b049` | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | 5b92a756-3b0e-5c54-aaa1-8867edd4b049] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 41/2016/TT-NHNN Hà Nội, Ngày 30 tháng 12 năm 2016 THÔNG TƯ Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài Căn cứ Lu… |
| 3 | 0.8699 | `f4924965-73f4-5fd2-8a24-d634d56ebdb1` | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | Điều 6 | f4924965-73f4-5fd2-8a24-d634d56ebdb1] | Điều 6. Tỷ lệ an toàn vốn 1. Tỷ lệ an toàn vốn (CAR) tính theo đơn vị phần trăm (%) được xác định bằng công thức: Trong đó: - C : Vốn tự có; - RWA : Tổng tài sản tính theo rủi ro tín dụng; - KOR : Vốn yêu cầu cho rủi ro hoạt động; - KMR : V… |
| 4 | 0.8685 | `066a1aec-cb12-52f3-b0a4-498caea9c300` | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | Điều 23 | 066a1aec-cb12-52f3-b0a4-498caea9c300] | Điều 23. Hiệu lực thi hành 1. Thông tư này có hiệu lực thi hành kể từ ngày 01 tháng 01 năm 2020, trừ trường hợp quy định tại khoản 2 Điều này. 2. Các quy định tại Thông tư này được áp dụng sớm hơn thời điểm quy định tại khoản 1 Điều này đối… |
| 5 | 0.8682 | `b65a40cd-3aac-53e5-9be0-97aac512db9d` | [Luật Các tổ chức tín dụng số 32/2024/QH15 | Điều 138 | b65a40cd-3aac-53e5-9be0-97aac512db9d] | Điều 138. Tỷ lệ bảo đảm an toàn 1. Tổ chức tín dụng, chi nhánh ngân hàng nước ngoài phải duy trì các tỷ lệ bảo đảm an toàn sau đây: a) Tỷ lệ khả năng chi trả; b) Tỷ lệ an toàn vốn tối thiểu 08% hoặc tỷ lệ cao hơn theo quy định của Thống đốc… |

### HYBRID RESULTS (RRF)

| final_rank | chunk_id | bm25_rank | dense_rank | rrf_score | citation | text (≤240 chars) |
|---:|---|---:|---:|---:|---|---|
| 1 | `96edab3e-3305-5b10-a3e3-908a8be49736` | 1 | 1 | 0.032787 | [Thông tư số 22/2023/TT-NHNN Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016 của Thống đốc Ngân hàng… | 96edab3e-3305-5b10-a3e3-908a8be49736] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 22/2023/TT-NHNN Hà Nội, ngày 29 tháng 12 năm 2023 THÔNG TƯ Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016… |
| 2 | `b65a40cd-3aac-53e5-9be0-97aac512db9d` | 2 | 5 | 0.031514 | [Luật Các tổ chức tín dụng số 32/2024/QH15 | Điều 138 | b65a40cd-3aac-53e5-9be0-97aac512db9d] | Điều 138. Tỷ lệ bảo đảm an toàn 1. Tổ chức tín dụng, chi nhánh ngân hàng nước ngoài phải duy trì các tỷ lệ bảo đảm an toàn sau đây: a) Tỷ lệ khả năng chi trả; b) Tỷ lệ an toàn vốn tối thiểu 08% hoặc tỷ lệ cao hơn theo quy định của Thống đốc… |
| 3 | `f4924965-73f4-5fd2-8a24-d634d56ebdb1` | 4 | 3 | 0.031498 | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | Điều 6 | f4924965-73f4-5fd2-8a24-d634d56ebdb1] | Điều 6. Tỷ lệ an toàn vốn 1. Tỷ lệ an toàn vốn (CAR) tính theo đơn vị phần trăm (%) được xác định bằng công thức: Trong đó: - C : Vốn tự có; - RWA : Tổng tài sản tính theo rủi ro tín dụng; - KOR : Vốn yêu cầu cho rủi ro hoạt động; - KMR : V… |
| 4 | `5b92a756-3b0e-5c54-aaa1-8867edd4b049` | 6 | 2 | 0.031281 | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | 5b92a756-3b0e-5c54-aaa1-8867edd4b049] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 41/2016/TT-NHNN Hà Nội, Ngày 30 tháng 12 năm 2016 THÔNG TƯ Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài Căn cứ Lu… |
| 5 | `bf51e0a4-1216-5e53-ac26-33e8222b7873` | 10 | 8 | 0.028992 | [Thông tư số 22/2023/TT-NHNN Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016 của Thống đốc Ngân hàng… | Điều 1 | bf51e0a4-1216-5e53-ac26-33e8222b7873] | Điều 1. Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN 1. Sửa đổi, bổ sung khoản 11 Điều 2 như sau: "11. Khoản cho vay thế chấp nhà là khoản cho vay bảo đảm bằng bất động sản đối với cá nhân để mua nhà, bao gồm: a) Khoản cho v… |

### HYBRID → RERANK RESULTS  _(rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3`)_

| final_rank | chunk_id | hybrid_rank | rerank_score | Δ vs hybrid | citation | text (≤240 chars) |
|---:|---|---:|---:|---:|---|---|
| 1 | `96edab3e-3305-5b10-a3e3-908a8be49736` | 1 | +0.9992 | +0 | [Thông tư số 22/2023/TT-NHNN Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016 của Thống đốc Ngân hàng… | 96edab3e-3305-5b10-a3e3-908a8be49736] | NGÂN HÀNG NHÀ NƯỚC VIỆT NAM CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 22/2023/TT-NHNN Hà Nội, ngày 29 tháng 12 năm 2023 THÔNG TƯ Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN ngày 30 tháng 12 năm 2016… |
| 2 | `b65a40cd-3aac-53e5-9be0-97aac512db9d` | 2 | +0.9170 | +0 | [Luật Các tổ chức tín dụng số 32/2024/QH15 | Điều 138 | b65a40cd-3aac-53e5-9be0-97aac512db9d] | Điều 138. Tỷ lệ bảo đảm an toàn 1. Tổ chức tín dụng, chi nhánh ngân hàng nước ngoài phải duy trì các tỷ lệ bảo đảm an toàn sau đây: a) Tỷ lệ khả năng chi trả; b) Tỷ lệ an toàn vốn tối thiểu 08% hoặc tỷ lệ cao hơn theo quy định của Thống đốc… |
| 3 | `f4924965-73f4-5fd2-8a24-d634d56ebdb1` | 3 | +0.9041 | +0 | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | Điều 6 | f4924965-73f4-5fd2-8a24-d634d56ebdb1] | Điều 6. Tỷ lệ an toàn vốn 1. Tỷ lệ an toàn vốn (CAR) tính theo đơn vị phần trăm (%) được xác định bằng công thức: Trong đó: - C : Vốn tự có; - RWA : Tổng tài sản tính theo rủi ro tín dụng; - KOR : Vốn yêu cầu cho rủi ro hoạt động; - KMR : V… |
| 4 | `060c5e64-80f1-589d-87b8-7181acfee92e` | 14 | +0.8497 | +10 | [Thông tư số 27/2024/TT-NHNN Quy định về việc ngân hàng hợp tác xã, việc trích nộp, quản lý và sử dụng Quỹ bảo đảm an toàn hệ thống quỹ tín… | Điều 33 | 060c5e64-80f1-589d-87b8-7181acfee92e] | Điều 33. Hiệu lực thi hành 1. Thông tư này có hiệu lực từ ngày 01 tháng 7 năm 2024. 2. Thông tư này bãi bỏ: a) Thông tư số 31/2012/TT-NHNN ngày 26 tháng 11 năm 2012 của Thống đốc Ngân hàng Nhà nước quy định về ngân hàng hợp tác xã; b) Thông… |
| 5 | `066a1aec-cb12-52f3-b0a4-498caea9c300` | 9 | +0.8452 | +4 | [Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài | Điều 23 | 066a1aec-cb12-52f3-b0a4-498caea9c300] | Điều 23. Hiệu lực thi hành 1. Thông tư này có hiệu lực thi hành kể từ ngày 01 tháng 01 năm 2020, trừ trường hợp quy định tại khoản 2 Điều này. 2. Các quy định tại Thông tư này được áp dụng sớm hơn thời điểm quy định tại khoản 1 Điều này đối… |

**Phân tích nhanh:**

- Rerank method: `CrossEncoder:BAAI/bge-reranker-v2-m3`
- BEFORE top-5 → AFTER top-5: 2 promoted, 0 demoted, 2 mới vào top, 2 bị đẩy ra.


## Tổng quan

- **Q1 (có mã cụ thể)** — BM25 + Dense đều nhắm đúng `01/2014/TT-NHNN`. Hybrid thừa hưởng thế mạnh đó; rerank bám top-1 hoặc đẩy canonical preamble lên đầu nếu hữu ích.
- **Q2 (semantic thuần)** — Hybrid trộn tốt; Cross-Encoder rerank có xu hướng đẩy điều khoản có chữ 'điều kiện / thủ tục' lên ngay top-1.
- **Q3 (mã + semantic)** — Hybrid đã thấy cả `41/2016` và `22/2023`; rerank có thể đẩy bản sửa đổi (`22/2023`) lên #1 nếu cross-encoder đánh giá mức độ liên quan với câu hỏi 'sửa đổi bởi' cao hơn.
- Nếu rớt mạng / OOM / torch lỗi, hệ thống chuyển sang FALLBACK identity. Trong trường hợp đó ordering sẽ không đổi — bạn vẫn thấy Hybrid, mất phần rerank. Báo cáo này ghi rõ nếu FALLBACK được dùng.
