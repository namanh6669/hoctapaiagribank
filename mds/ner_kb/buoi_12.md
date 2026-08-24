# Bài thực hành 3: Chuẩn hóa, Làm giàu Metadata và Xây dựng Đồ thị Tri thức

## 1. Mục tiêu

Chuẩn hóa và làm giàu metadata từ tập 30 văn bản pháp luật:

```text
ner_kb/metadata.csv
ner_kb/content.csv
```

Học viên sử dụng kết hợp:

- Rule-based extraction.
- Entity Extraction / NER.
- LLM với Gemini API.
- Neo4j.

Sau bài thực hành, hệ thống cần thực hiện được pipeline:

```text
Raw Data
   ↓
Data Validation
   ↓
HTML Cleaning
   ↓
Rule-based Candidate Extraction
   ↓
Gemini Entity Extraction
   ↓
Metadata Enrichment
   ↓
Entity Normalization
   ↓
Relationship Extraction
   ↓
Relationship Validation
   ↓
Knowledge Graph
   ↓
Neo4j
```

> **Nguyên tắc quan trọng**
>
> - Không sửa trực tiếp `metadata.csv` và `content.csv`.
> - Không đưa trực tiếp output chưa kiểm tra của LLM vào Neo4j.
> - Mỗi bước phải chạy thành công và kiểm tra output trước khi sang bước tiếp theo.
> - Không yêu cầu Coding Agent làm toàn bộ bài trong một lần.

---

# 2. Dữ liệu đầu vào

Ban đầu thư mục cần có:

```text
ner_kb/
├── metadata.csv
└── content.csv
```

Trong đó:

- `metadata.csv`: metadata của 30 văn bản pháp luật.
- `content.csv`: nội dung HTML của 30 văn bản.

Không tự ý thay đổi hai file này.

---

# 3. Quy tắc làm việc với Coding Agent

Ở mỗi bước:

```text
Gửi yêu cầu
   ↓
Agent kiểm tra code hiện tại
   ↓
Agent chỉ thực hiện bước được yêu cầu
   ↓
Chạy chương trình
   ↓
Kiểm tra output
   ↓
PASS → sang bước tiếp theo
FAIL → sửa bước hiện tại
```

Cuối mỗi prompt nên thêm:

```text
Chưa thực hiện bước tiếp theo.
Sau khi hoàn thành, hãy chạy thử, báo cáo kết quả và dừng lại để tôi kiểm tra.
```

Điều này giúp tránh Agent tự thay đổi toàn bộ project.

---

# BƯỚC 0 — Kiểm tra môi trường

## Mục tiêu

Đảm bảo môi trường có thể chạy trước khi viết logic xử lý dữ liệu.

## Agent cần kiểm tra

- Python.
- Virtual environment.
- Hai file dữ liệu đầu vào.
- Các thư viện cần thiết.
- Gemini API key.
- Neo4j.
- Cấu trúc project.

## Thư viện Python

Các package chính:

```text
pandas
beautifulsoup4
python-dotenv
google-genai
neo4j
```

Google hiện khuyến nghị sử dụng **Google Gen AI SDK** cho Gemini API.

Neo4j sử dụng package Python chính thức:

```text
neo4j
```

## Cấu hình `.env`

Không hard-code khóa bí mật trong code.

Ví dụ:

```text
GEMINI_API_KEY=YOUR_KEY_HERE

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_PASSWORD_HERE
NEO4J_DATABASE=neo4j
```

Không commit `.env` lên Git.

Có thể tạo:

```text
.env.example
```

nhưng không chứa khóa thật.

## Prompt cho Agent

```text
Hãy thực hiện BƯỚC 0: kiểm tra môi trường project hiện tại.

Yêu cầu:
1. Chưa sửa logic nghiệp vụ.
2. Kiểm tra Python version.
3. Kiểm tra có virtual environment hay chưa.
4. Kiểm tra tồn tại:
   - ner_kb/metadata.csv
   - ner_kb/content.csv
5. Kiểm tra các package:
   - pandas
   - beautifulsoup4
   - python-dotenv
   - google-genai
   - neo4j
6. Kiểm tra file .env.
7. Không được in GEMINI_API_KEY hoặc NEO4J_PASSWORD ra terminal.
8. Kiểm tra có thể import các package.
9. Nếu package thiếu, cài vào virtual environment hiện tại.
10. Kiểm tra Neo4j có thể kết nối nếu cấu hình đã có.
11. Báo cáo PASS/FAIL cho từng mục.

Chưa thực hiện xử lý dữ liệu.
Chưa viết NER.
Chưa gọi Gemini extraction.
Chưa import Neo4j.

Sau khi kiểm tra xong hãy dừng lại.
```

## Điều kiện PASS

Chỉ sang Bước 1 khi:

```text
[PASS] Python
[PASS] Virtual environment
[PASS] metadata.csv
[PASS] content.csv
[PASS] Python packages
[PASS] Gemini configuration
[PASS] Neo4j configuration
```

Nếu chưa cấu hình Gemini hoặc Neo4j, có thể hoàn thiện cấu hình trước khi đi tiếp.

---

# BƯỚC 1 — Kiểm tra dữ liệu và làm sạch HTML

## Mục tiêu

Kiểm tra chất lượng dữ liệu đầu vào và tạo một file dữ liệu sạch dùng thống nhất cho các bước sau.

## Input

```text
ner_kb/metadata.csv
ner_kb/content.csv
```

## Việc cần làm

1. Đọc hai file bằng `pandas`.
2. Kiểm tra số dòng.
3. Kiểm tra ID trùng.
4. Kiểm tra ID thiếu giữa hai file.
5. Ghép dữ liệu theo `id`.
6. Thống kê missing values.
7. Phát hiện giá trị chưa chuẩn như:
   - NULL.
   - Rỗng.
   - `Chưa phân loại`.
8. Làm sạch `content_html` bằng BeautifulSoup.
9. Không paraphrase nội dung.
10. Không xóa các cụm pháp lý quan trọng như:
    - `Căn cứ`.
    - `Sửa đổi, bổ sung`.
    - `bãi bỏ`.
    - `thay thế`.
11. Tạo cột:

```text
content_clean
```

## Output bắt buộc

```text
ner_kb/cleaned_documents.csv
```

File này là input chính cho Bước 2 và Bước 3.

Không sửa:

```text
metadata.csv
content.csv
```

## Prompt cho Agent

```text
Thực hiện BƯỚC 1: kiểm tra dữ liệu và làm sạch HTML.

Input:
- ner_kb/metadata.csv
- ner_kb/content.csv

Yêu cầu:
1. Đọc bằng pandas.
2. Kiểm tra số dòng, số cột.
3. Kiểm tra duplicate id.
4. Kiểm tra id chỉ có ở một trong hai file.
5. Merge theo id.
6. Thống kê missing values cho metadata.
7. Phát hiện NULL, chuỗi rỗng và "Chưa phân loại".
8. Làm sạch content_html bằng BeautifulSoup.
9. Chỉ loại HTML và chuẩn hóa whitespace; không viết lại nội dung.
10. Giữ nguyên các số hiệu văn bản và các cụm "Căn cứ",
    "Sửa đổi, bổ sung", "bãi bỏ", "thay thế".
11. Tạo cột content_clean.
12. Lưu:
    ner_kb/cleaned_documents.csv
13. In:
    - số document;
    - số duplicate id;
    - số id mismatch;
    - missing values;
    - 2 mẫu content_html và content_clean.
14. Không sửa metadata.csv hoặc content.csv.

Chưa làm candidate extraction.
Chưa gọi Gemini.
Chưa tạo Knowledge Graph.

Hãy chạy bước này, báo PASS/FAIL và dừng lại.
```

## Điều kiện PASS

Đảm bảo:

```text
cleaned_documents.csv tồn tại
```

và:

- Có đủ số document tương ứng với dữ liệu đầu vào.
- Không mất `id`.
- `content_clean` không rỗng bất thường.
- Số hiệu văn bản vẫn còn.
- Các câu pháp lý quan trọng vẫn còn.

---

# BƯỚC 2 — Rule-based Candidate Extraction

## Mục tiêu

Dùng rule để phát hiện các văn bản có khả năng liên quan trước khi gọi LLM.

Không sử dụng Gemini ở bước này.

## Input

```text
ner_kb/cleaned_documents.csv
```

## Candidate cần phát hiện

Các mẫu như:

```text
Căn cứ ...
Thông tư số ...
Nghị định số ...
Luật số ...
Sửa đổi, bổ sung ...
bãi bỏ ...
thay thế ...
```

Ví dụ:

```text
Thông tư số 22/2023/TT-NHNN
Sửa đổi, bổ sung một số điều của Thông tư số 41/2016/TT-NHNN
```

Candidate:

```text
22/2023/TT-NHNN
        ↓
41/2016/TT-NHNN
```

Lưu ý:

> Candidate chưa phải relationship cuối cùng.

## Output bắt buộc

```text
ner_kb/relation_candidates.csv
```

Schema tối thiểu:

```text
source_id
source_so_ky_hieu
target_so_ky_hieu
trigger
evidence
```

Trong đó:

- `trigger`: từ khóa phát hiện candidate.
- `evidence`: đoạn text chứa candidate.

## Prompt cho Agent

```text
Thực hiện BƯỚC 2: rule-based candidate extraction.

Input:
ner_kb/cleaned_documents.csv

Yêu cầu:
1. Không dùng Gemini.
2. Phát hiện số hiệu văn bản được nhắc trong content_clean.
3. Ưu tiên các context có:
   - Căn cứ
   - Sửa đổi, bổ sung
   - bãi bỏ
   - thay thế
4. Với mỗi candidate lưu:
   - source_id
   - source_so_ky_hieu
   - target_so_ky_hieu
   - trigger
   - evidence
5. Loại candidate tự tham chiếu chính văn bản hiện tại nếu không có ý nghĩa.
6. Loại duplicate candidate.
7. Không kết luận relationship_type cuối cùng ở bước này.
8. Lưu:
   ner_kb/relation_candidates.csv
9. In thống kê:
   - tổng số candidate;
   - số candidate theo trigger;
   - 10 candidate mẫu.

Chưa gọi Gemini.
Chưa tạo relationships.csv.
Chưa import Neo4j.

Chạy, báo PASS/FAIL và dừng lại.
```

## Điều kiện PASS

- `relation_candidates.csv` tồn tại.
- Không có duplicate rõ ràng.
- `evidence` không rỗng.
- Có thể xem thủ công một số candidate và thấy target thực sự xuất hiện trong evidence.

---

# BƯỚC 3 — Entity Extraction và Metadata Enrichment bằng Gemini

## Mục tiêu

Dùng Gemini để bổ sung metadata khó trích xuất bằng rule.

## Input

```text
ner_kb/cleaned_documents.csv
```

Không sử dụng `relationships_ground_truth.csv`.

## Các entity

### `CoQuan`

Cơ quan ban hành.

Ví dụ:

```text
Quốc hội
Chính phủ
Bộ Tài chính
Ngân hàng Nhà nước Việt Nam
```

### `NguoiKy`

Người ký / người có thẩm quyền ban hành.

### `DoiTuongApDung`

Đối tượng chịu sự điều chỉnh.

Ví dụ:

```text
Ngân hàng thương mại
Chi nhánh ngân hàng nước ngoài
Quỹ tín dụng nhân dân
Tổ chức tín dụng
```

### `LinhVuc`

Lĩnh vực pháp lý.

Ví dụ:

```text
Tín dụng
Kiểm toán
Bảo hiểm
Chứng khoán
Quản lý ngoại hối
Phát hành và kho quỹ
```

## Nguyên tắc

Ưu tiên metadata gốc khi giá trị đã rõ.

Gemini chủ yếu dùng để:

```text
Bổ sung missing
Phân loại "Chưa phân loại"
Trích xuất DoiTuongApDung
Làm giàu LinhVuc
Kiểm tra metadata cần xem lại
```

LLM không được tự ý ghi đè raw metadata.

## Structured output

Output của Gemini phải được parse thành cấu trúc ổn định.

Ví dụ logic:

```json
{
  "co_quan": [],
  "nguoi_ky": [],
  "doi_tuong_ap_dung": [],
  "linh_vuc": []
}
```

Mỗi entity được lưu kèm:

```text
entity
entity_type
source
method
confidence
evidence
```

Ví dụ:

```json
{
  "entity": "Ngân hàng thương mại",
  "entity_type": "DoiTuongApDung",
  "source": "content_clean",
  "method": "gemini",
  "confidence": 0.94,
  "evidence": "Thông tư này áp dụng đối với ngân hàng thương mại..."
}
```

## Quy tắc an toàn

- Nếu không có bằng chứng → không tạo entity.
- Nếu Gemini trả response rỗng → ghi nhận lỗi document đó, không crash toàn batch.
- Nếu JSON không hợp lệ → xử lý lỗi, không crash toàn batch.
- Không tự đặt confidence = 1 cho tất cả.
- Không được dùng confidence thay cho evidence.
- Không log API key.

## Output bắt buộc

```text
ner_kb/extracted_entities_raw.csv
ner_kb/enriched_metadata.csv
```

### `extracted_entities_raw.csv`

Chứa entity chưa normalize.

### `enriched_metadata.csv`

Chứa metadata gốc cùng metadata được bổ sung.

Không sửa:

```text
metadata.csv
content.csv
cleaned_documents.csv
```

## Prompt cho Agent

```text
Thực hiện BƯỚC 3: Entity Extraction và Metadata Enrichment bằng Gemini.

Input:
ner_kb/cleaned_documents.csv

Yêu cầu:
1. Trước khi sửa code, đọc output và code của Bước 1 và Bước 2.
2. Không thay đổi raw data.
3. Trích xuất:
   - CoQuan
   - NguoiKy
   - DoiTuongApDung
   - LinhVuc
4. Ưu tiên metadata gốc nếu đã rõ.
5. Dùng Gemini để bổ sung metadata thiếu hoặc chưa chuẩn.
6. Mỗi entity cần:
   - entity
   - entity_type
   - source
   - method
   - confidence
   - evidence
7. Nếu không có evidence, không tạo entity.
8. Dùng structured JSON output hoặc cơ chế parse JSON ổn định.
9. Có xử lý:
   - API error
   - empty response
   - malformed JSON
   - missing field
10. Một document lỗi không được làm dừng toàn batch.
11. Lưu:
    ner_kb/extracted_entities_raw.csv
    ner_kb/enriched_metadata.csv
12. Không ghi đè:
    metadata.csv
    content.csv
13. Sau khi chạy, báo:
    - số document thành công;
    - số document thất bại;
    - số entity theo loại;
    - số giá trị metadata được bổ sung;
    - 5 ví dụ metadata gốc so với metadata làm giàu;
    - danh sách lỗi nếu có.

Chưa normalize entity.
Chưa trích xuất relationship cuối cùng.
Chưa tạo Neo4j graph.

Chạy, báo PASS/FAIL và dừng lại.
```

## Điều kiện PASS

Phải có:

```text
extracted_entities_raw.csv
enriched_metadata.csv
```

Kiểm tra ít nhất:

- `CoQuan` hợp lý.
- `NguoiKy` không bị thay đổi vô lý.
- `DoiTuongApDung` có evidence.
- `LinhVuc` cải thiện các giá trị thiếu/chưa phân loại.
- Không có entity rõ ràng bị hallucination.

Nếu chưa đạt, sửa Bước 3 trước khi sang Bước 4.

---

# BƯỚC 4 — Chuẩn hóa Entity

## Mục tiêu

Loại tình trạng một thực thể xuất hiện dưới nhiều tên.

Ví dụ:

```text
NHNN
Ngân hàng Nhà nước
Ngân hàng Nhà nước Việt Nam
```

không nên tạo thành ba node khác nhau.

## Input

```text
ner_kb/extracted_entities_raw.csv
ner_kb/enriched_metadata.csv
```

## Thực hiện

- Trim khoảng trắng.
- Chuẩn hóa Unicode.
- Chuẩn hóa viết hoa/viết thường khi so sánh.
- Loại duplicate.
- Dùng alias mapping có kiểm soát.
- Không merge entity nếu chưa chắc cùng một đối tượng.

## Output bắt buộc

```text
ner_kb/entities.csv
```

Schema gợi ý:

```text
entity_id
entity_type
canonical_name
original_name
source_doc_id
method
confidence
evidence
```

## Prompt cho Agent

```text
Thực hiện BƯỚC 4: chuẩn hóa entity.

Input:
- ner_kb/extracted_entities_raw.csv
- ner_kb/enriched_metadata.csv

Yêu cầu:
1. Chuẩn hóa whitespace và Unicode.
2. Loại duplicate.
3. Chuẩn hóa alias rõ ràng.
4. Ví dụ có thể chuẩn hóa:
   NHNN -> Ngân hàng Nhà nước Việt Nam
   nếu context xác nhận đúng cùng thực thể.
5. Không fuzzy-merge mạnh các tên người hoặc cơ quan.
6. Không merge nếu chưa chắc chắn.
7. Giữ original_name và canonical_name để truy vết.
8. Lưu:
   ner_kb/entities.csv
9. In:
   - số entity trước normalize;
   - số entity sau normalize;
   - các alias đã merge;
   - 10 entity mẫu.

Chưa tạo relationship cuối cùng.
Chưa import Neo4j.

Chạy, báo PASS/FAIL và dừng lại.
```

## Điều kiện PASS

- `entities.csv` tồn tại.
- Không còn duplicate hiển nhiên.
- Không merge nhầm tên người.
- Có thể truy ngược `canonical_name` về `original_name`.

---

# BƯỚC 5 — Relationship Extraction

## Mục tiêu

Phân loại candidate thành relationship thực sự và tạo các quan hệ giữa Document với Entity.

## Input

```text
ner_kb/cleaned_documents.csv
ner_kb/relation_candidates.csv
ner_kb/entities.csv
ner_kb/enriched_metadata.csv
```

## Quan hệ Document → Document

### `THAM_CHIEU`

```text
(Document A)-[:THAM_CHIEU]->(Document B)
```

Dùng khi A căn cứ hoặc viện dẫn B.

### `SUA_DOI_BO_SUNG`

```text
(Document A)-[:SUA_DOI_BO_SUNG]->(Document B)
```

Dùng khi A sửa đổi hoặc bổ sung B.

### `THAY_THE_BOI`

Chiều quan hệ:

```text
(Document cũ)-[:THAY_THE_BOI]->(Document mới)
```

Ví dụ:

```text
44/2011/TT-NHNN
        |
        | THAY_THE_BOI
        v
62/2025/TT-NHNN
```

Không đồng nhất:

```text
SUA_DOI_BO_SUNG != THAY_THE_BOI
```

## Quan hệ Document → Entity

### `BAN_HANH_BOI`

```text
(Document)-[:BAN_HANH_BOI]->(CoQuan)
```

### `KY_BOI`

```text
(Document)-[:KY_BOI]->(NguoiKy)
```

### `AP_DUNG_CHO`

```text
(Document)-[:AP_DUNG_CHO]->(DoiTuongApDung)
```

### `THUOC_LINH_VUC`

```text
(Document)-[:THUOC_LINH_VUC]->(LinhVuc)
```

## Khi nào dùng Gemini?

Không gửi toàn bộ corpus cho Gemini một lần.

Với Document → Document:

```text
Rule candidate
      ↓
Lấy evidence/context ngắn
      ↓
Gemini phân loại khi rule chưa đủ chắc
```

Mỗi relation cần:

```text
source
target
relationship_type
method
confidence
evidence
```

## Output bắt buộc

```text
ner_kb/relationships_raw.csv
```

## Prompt cho Agent

```text
Thực hiện BƯỚC 5: Relationship Extraction.

Input:
- ner_kb/cleaned_documents.csv
- ner_kb/relation_candidates.csv
- ner_kb/entities.csv
- ner_kb/enriched_metadata.csv

Yêu cầu:
1. Tạo Document -> Document relations:
   - THAM_CHIEU
   - SUA_DOI_BO_SUNG
   - THAY_THE_BOI
2. Tạo Document -> Entity relations:
   - BAN_HANH_BOI
   - KY_BOI
   - AP_DUNG_CHO
   - THUOC_LINH_VUC
3. Với Document -> Document:
   - ưu tiên rule nếu trigger rõ;
   - chỉ gọi Gemini trên context cần phân loại;
   - không gửi toàn bộ corpus nếu không cần.
4. Với THAY_THE_BOI, giữ đúng chiều:
   Document cũ -> Document mới.
5. SUA_DOI_BO_SUNG không được tự đổi thành THAY_THE_BOI.
6. Mỗi relation phải lưu:
   - source
   - target
   - relationship_type
   - method
   - confidence
   - evidence
7. Không tạo relation nếu evidence không đủ.
8. Loại duplicate.
9. Lưu:
   ner_kb/relationships_raw.csv
10. In:
    - số relation theo type;
    - 10 relation mẫu;
    - evidence đi kèm.

Chưa import Neo4j.
Chưa sử dụng ground truth.

Chạy, báo PASS/FAIL và dừng lại.
```

## Điều kiện PASS

- `relationships_raw.csv` tồn tại.
- Mọi edge có source, target và type.
- Relation do extraction tạo có evidence.
- Không có duplicate rõ ràng.
- Chiều `THAY_THE_BOI` đúng.

---

# BƯỚC 6 — Validate Relationship và tạo output chính thức

## Mục tiêu

Không đưa dữ liệu lỗi vào Neo4j.

## Input

```text
ner_kb/relationships_raw.csv
ner_kb/cleaned_documents.csv
ner_kb/entities.csv
```

## Kiểm tra bắt buộc

### Document target

Nếu target là Document:

- Source tồn tại trong corpus.
- Target tồn tại trong corpus nếu bài yêu cầu closed-corpus graph.
- Không self-loop vô nghĩa.

### Entity target

Nếu target là Entity:

- Entity tồn tại trong `entities.csv`.

### Relationship

Chỉ chấp nhận:

```text
THAM_CHIEU
SUA_DOI_BO_SUNG
THAY_THE_BOI
BAN_HANH_BOI
KY_BOI
AP_DUNG_CHO
THUOC_LINH_VUC
```

Kiểm tra:

- Missing field.
- Duplicate edge.
- Evidence rỗng.
- Relationship type sai.
- Source/target không tồn tại.

## Output bắt buộc

```text
ner_kb/relationships.csv
ner_kb/validation_report.csv
```

`relationships.csv` là file **đã validate** và là input cho Neo4j.

## Prompt cho Agent

```text
Thực hiện BƯỚC 6: validate relationship.

Input:
- ner_kb/relationships_raw.csv
- ner_kb/cleaned_documents.csv
- ner_kb/entities.csv

Yêu cầu:
1. Validate source.
2. Validate target.
3. Validate relationship_type.
4. Kiểm tra self-loop.
5. Kiểm tra duplicate.
6. Kiểm tra missing evidence.
7. Không tự sửa semantic relation bằng suy đoán.
8. Relation không đạt phải đưa vào validation_report.csv
   cùng lý do bị loại.
9. Relation đạt lưu vào:
   ner_kb/relationships.csv
10. Lưu toàn bộ báo cáo:
    ner_kb/validation_report.csv
11. In:
    - tổng relation raw;
    - số PASS;
    - số FAIL;
    - số theo relationship_type;
    - nguyên nhân fail phổ biến;
    - 10 relation PASS mẫu.

Chưa dùng ground truth.
Chưa import Neo4j.

Chạy, báo PASS/FAIL và dừng lại.
```

## Điều kiện PASS

Phải có:

```text
relationships.csv
validation_report.csv
```

và:

```text
FAIL nghiêm trọng = 0
```

hoặc các dòng FAIL đã được loại khỏi `relationships.csv`.

---


# BƯỚC 7 — Kiểm tra cấu hình Neo4j trước khi import

## Mục tiêu

Không để lỗi connection xuất hiện giữa quá trình import.

## Input

```text
.env
```

## Kiểm tra

- URI.
- Username.
- Password.
- Database.
- Driver connection.

Dùng package Python:

```text
neo4j
```

## Prompt cho Agent

```text
Thực hiện BƯỚC 7: kiểm tra kết nối Neo4j.

Yêu cầu:
1. Đọc cấu hình từ .env.
2. Không in password.
3. Dùng official neo4j Python driver.
4. Mở driver.
5. Verify connectivity.
6. Chạy query đọc đơn giản để xác nhận database hoạt động.
7. Đóng driver đúng cách.
8. Không import dữ liệu ở bước này.

Báo:
- connection PASS/FAIL;
- database đang sử dụng;
- lỗi cụ thể nếu có.

Sau đó dừng lại.
```

## Điều kiện PASS

```text
Neo4j connection: PASS
```

mới được sang Bước 9.

---

# BƯỚC 8 — Import Knowledge Graph vào Neo4j

## Input

```text
ner_kb/cleaned_documents.csv
ner_kb/entities.csv
ner_kb/relationships.csv
```

## Node

```text
(:Document)
(:CoQuan)
(:NguoiKy)
(:DoiTuongApDung)
(:LinhVuc)
```

## Relationship

```text
[:THAM_CHIEU]
[:SUA_DOI_BO_SUNG]
[:THAY_THE_BOI]
[:BAN_HANH_BOI]
[:KY_BOI]
[:AP_DUNG_CHO]
[:THUOC_LINH_VUC]
```

## Graph schema

```text
(:Document)
   |
   +--[:BAN_HANH_BOI]----> (:CoQuan)
   |
   +--[:KY_BOI]-----------> (:NguoiKy)
   |
   +--[:AP_DUNG_CHO]------> (:DoiTuongApDung)
   |
   +--[:THUOC_LINH_VUC]---> (:LinhVuc)
   |
   +--[:THAM_CHIEU]-------> (:Document)
   |
   +--[:SUA_DOI_BO_SUNG]--> (:Document)
   |
   +--[:THAY_THE_BOI]-----> (:Document)
```

## Nguyên tắc import

Ưu tiên:

```cypher
MERGE
```

để tránh tạo node trùng khi chạy lại.

Không dùng `CREATE` một cách không kiểm soát cho entity dùng chung.

Khuyến nghị tạo uniqueness constraint cho khóa định danh phù hợp trước khi import.

Ví dụ logic:

```text
Document → id hoặc so_ky_hieu
Entity   → entity_type + canonical_name
```

## Prompt cho Agent

```text
Thực hiện BƯỚC 8: import Knowledge Graph vào Neo4j.

Input:
- ner_kb/cleaned_documents.csv
- ner_kb/entities.csv
- ner_kb/relationships.csv

Yêu cầu:
1. Đọc Neo4j config từ .env.
2. Không hard-code password.
3. Tạo uniqueness constraint hợp lý trước khi import.
4. Dùng MERGE cho node dùng chung.
5. Tạo:
   Document
   CoQuan
   NguoiKy
   DoiTuongApDung
   LinhVuc
6. Tạo:
   THAM_CHIEU
   SUA_DOI_BO_SUNG
   THAY_THE_BOI
   BAN_HANH_BOI
   KY_BOI
   AP_DUNG_CHO
   THUOC_LINH_VUC
7. Nếu source hoặc target không tìm thấy, không tạo node rác;
   ghi lỗi import riêng.
8. Import phải idempotent:
   chạy lại không làm tăng duplicate node/edge.
9. Sau import in:
   - số node theo label;
   - số relationship theo type;
   - số lỗi import.
10. Đóng Neo4j driver đúng cách.

Chạy import, báo PASS/FAIL và dừng lại.
```

## Điều kiện PASS

Chạy import **hai lần**.

Nếu thiết kế đúng:

```text
Lần 2 không tạo duplicate đáng kể
```

Đây là kiểm tra quan trọng của `MERGE`.

---

# BƯỚC 9 — Kiểm tra và trực quan hóa trên Neo4j Browser

## 9.1. Kiểm tra số node

```cypher
MATCH (n)
RETURN labels(n) AS labels, count(*) AS total
ORDER BY total DESC;
```

## 9.2. Kiểm tra relationship

```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(*) AS total
ORDER BY total DESC;
```

## 9.3. Xem graph mẫu

```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100;
```

## 9.4. Văn bản và người ký

```cypher
MATCH (d:Document)-[:KY_BOI]->(p:NguoiKy)
RETURN d, p
LIMIT 50;
```

## 9.5. Đối tượng áp dụng

```cypher
MATCH (d:Document)-[:AP_DUNG_CHO]->(o:DoiTuongApDung)
RETURN d, o
LIMIT 50;
```

## 9.6. Quan hệ Document → Document

```cypher
MATCH path=(a:Document)-[:THAM_CHIEU|SUA_DOI_BO_SUNG|THAY_THE_BOI]->(b:Document)
RETURN path
LIMIT 50;
```

## 9.7. Chuỗi tham chiếu

```cypher
MATCH path=(d1:Document)-[:THAM_CHIEU*1..3]->(d2:Document)
RETURN path
LIMIT 20;
```

## Prompt cho Agent

```text
Thực hiện BƯỚC 9: kiểm tra Knowledge Graph sau import.

Không sửa dữ liệu.

Hãy chạy các query kiểm tra:
1. node count theo label;
2. relationship count theo type;
3. một số Document -> NguoiKy;
4. một số Document -> DoiTuongApDung;
5. Document -> Document relations.

Đối chiếu số liệu với các CSV trước khi import.
Nếu chênh lệch bất thường, tìm nguyên nhân nhưng chưa tự xóa database.

Báo PASS/FAIL và dừng lại.
```

---

# 4. Cấu trúc thư mục cuối cùng

Sau khi hoàn thành bài:

```text
project/
│
├── .env
├── .env.example
│
└── ner_kb/
    ├── metadata.csv
    ├── content.csv
    │
    ├── cleaned_documents.csv
    ├── relation_candidates.csv
    ├── extracted_entities_raw.csv
    ├── enriched_metadata.csv
    ├── entities.csv
    ├── relationships_raw.csv
    ├── relationships.csv
    ├── validation_report.csv
```


---

# 5. Ý nghĩa của từng file

| File | Vai trò |
|---|---|
| `metadata.csv` | Raw metadata, không sửa |
| `content.csv` | Raw content, không sửa |
| `cleaned_documents.csv` | Dữ liệu đã merge và clean |
| `relation_candidates.csv` | Candidate từ rule |
| `extracted_entities_raw.csv` | Entity LLM/rule chưa normalize |
| `enriched_metadata.csv` | Metadata sau enrichment |
| `entities.csv` | Entity canonical đã normalize |
| `relationships_raw.csv` | Relation trước validation |
| `relationships.csv` | Relation chính thức sau validation |
| `validation_report.csv` | Báo cáo relation bị loại / lỗi |

---

# 6. Checklist trước khi kết thúc

## Raw data

```text
[ ] metadata.csv không bị sửa
[ ] content.csv không bị sửa
```

## Data processing

```text
[ ] cleaned_documents.csv được tạo
[ ] relation_candidates.csv được tạo
[ ] extracted_entities_raw.csv được tạo
[ ] enriched_metadata.csv được tạo
[ ] entities.csv được tạo
```

## Relationships

```text
[ ] relationships_raw.csv được tạo
[ ] relationships.csv đã validate
[ ] mọi relation có source/target/type
[ ] relation extraction có evidence khi phù hợp
[ ] không có duplicate edge rõ ràng
[ ] chiều THAY_THE_BOI đúng
```

## Neo4j

```text
[ ] Neo4j connection PASS
[ ] Document nodes tồn tại
[ ] Entity nodes tồn tại
[ ] Relationships tồn tại
[ ] chạy import lần 2 không tạo duplicate
[ ] query visualization chạy được
```

---

# 7. Các lỗi cần tránh

## Lỗi 1 — Yêu cầu Agent làm toàn bộ bài một lần

Không nên:

```text
Hãy làm toàn bộ hệ thống NER + Gemini + Neo4j.
```

Nên làm từng bước.

---

## Lỗi 2 — Gọi Gemini trước khi kiểm tra data

Sai pipeline:

```text
CSV
 ↓
Gemini
```

Pipeline đúng:

```text
CSV
 ↓
Validation + Cleaning
 ↓
Rule
 ↓
Gemini khi cần
```

---

## Lỗi 3 — Ghi đè raw metadata

Không sửa:

```text
metadata.csv
content.csv
```

Luôn sinh derived data thành file mới.

---

## Lỗi 4 — Tin confidence mà không xem evidence

```text
confidence = 0.99
```

không đồng nghĩa với dữ liệu chắc chắn đúng.

Kết quả extraction phải có khả năng truy vết về source/evidence.

---

## Lỗi 5 — Nhầm sửa đổi với thay thế

```text
SUA_DOI_BO_SUNG
```

không đồng nghĩa:

```text
THAY_THE_BOI
```

---

## Lỗi 6 — Import thẳng output chưa validate

Sai:

```text
Gemini
  ↓
Neo4j
```

Đúng:

```text
Gemini
  ↓
Normalize
  ↓
Validate
  ↓
Neo4j
```

---

## Lỗi 7 — Tạo duplicate khi chạy Neo4j import lại

Import cần được thiết kế idempotent bằng constraint và `MERGE` phù hợp.

---

# 8. Kết quả học viên cần hiểu

Sau bài thực hành, học viên không chỉ cần tạo được một graph.

Học viên cần hiểu:

```text
Văn bản thô
    ↓
Thông tin có cấu trúc
    ↓
Entity
    ↓
Relationship
    ↓
Validation
    ↓
Knowledge Graph
```

Quan trọng hơn:

> **LLM là công cụ hỗ trợ extraction và enrichment; LLM không phải nguồn sự thật cuối cùng của Knowledge Graph.**

Một Knowledge Graph có giá trị cần:

```text
Extraction
    ↓
Normalization
    ↓
Validation
    ↓
Traceability / Evidence
    ↓
Graph Storage
```

---

# 9. Thứ tự chạy toàn bài

```text
BƯỚC 0
Environment Check
        ↓ PASS
BƯỚC 1
Data Validation + HTML Cleaning
        ↓ cleaned_documents.csv
BƯỚC 2
Rule-based Candidate Extraction
        ↓ relation_candidates.csv
BƯỚC 3
Gemini Entity Extraction + Metadata Enrichment
        ↓ extracted_entities_raw.csv
        ↓ enriched_metadata.csv
BƯỚC 4
Entity Normalization
        ↓ entities.csv
BƯỚC 5
Relationship Extraction
        ↓ relationships_raw.csv
BƯỚC 6
Relationship Validation
        ↓ relationships.csv
        ↓ validation_report.csv
BƯỚC 7
Neo4j Connection Check
        ↓ PASS
BƯỚC 8
Neo4j Import
        ↓ PASS
BƯỚC 9
Cypher Query + Visualization
```

> **Chỉ sang bước tiếp theo khi bước hiện tại PASS.**
