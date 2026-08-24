# Bài thực hành: Xây dựng Wiki Risk Graph bằng Vibe Coding

## Mục tiêu

Trong bài thực hành này, người học sử dụng **AI Coding Agent (Claude Code hoặc công cụ tương đương)** để xây dựng một **Wiki tri thức rủi ro dạng đồ thị** từ dữ liệu rủi ro mô phỏng.

Sản phẩm cuối cùng cho phép:

- Tra cứu hồ sơ rủi ro.
- Xem kiểm soát nào giảm thiểu từng rủi ro.
- Xem các sự kiện đã ghi nhận liên quan đến rủi ro.
- Duyệt các trang Markdown bằng Obsidian.
- Quan sát mạng lưới liên kết bằng Obsidian Graph View.
- Xuất dữ liệu node/edge để có thể nạp vào Neo4j.
- Kiểm tra nguồn và trạng thái xác minh của từng quan hệ.

> Trọng tâm của bài không phải tự viết toàn bộ code từ đầu. Người học học cách **giao việc, kiểm tra kết quả, chạy thử, phát hiện lỗi và yêu cầu Agent sửa**.

---

# 1. Dữ liệu sử dụng

Sử dụng 4 file CSV được cung cấp:

```text
data/
├── risk_profiles_seed.csv
├── controls_seed.csv
├── risk_events_seed.csv
└── relationships_seed.csv
```

## 1.1. `risk_profiles_seed.csv`

Chứa hồ sơ rủi ro.

Các trường chính:

```text
id
name
category
description
cause
event
impact
inherent_level
residual_level
owner_unit_id
data_origin
verification_status
```

Ví dụ một hồ sơ:

```text
RR-001
Giao dịch chuyển tiền bị hạch toán sai
```

Mỗi hồ sơ đã có cấu trúc:

```text
Nguyên nhân → Sự kiện → Hậu quả
```

Đây sẽ là nhóm node **RuiRo** trong Wiki Risk Graph.

---

## 1.2. `controls_seed.csv`

Chứa các kiểm soát giảm thiểu rủi ro.

Các trường chính:

```text
id
name
control_type
frequency
owner_role_id
effectiveness
data_origin
verification_status
```

Ví dụ:

```text
KS-001
Đối soát tự động giao dịch và sổ cái
```

Đây sẽ là nhóm node **KiemSoat**.

---

## 1.3. `risk_events_seed.csv`

Chứa các sự kiện rủi ro đã quan sát.

Các trường chính:

```text
id
risk_id
occurred_at
discovered_at
severity
loss_amount_vnd
description
data_origin
verification_status
```

Ví dụ:

```text
SK-001
Sai lệch trạng thái giao dịch được phát hiện khi đối soát cuối ngày
```

Đây sẽ là nhóm node **SuKienRuiRo**.

---

## 1.4. `relationships_seed.csv`

Chứa các cạnh của đồ thị.

Trong bộ dữ liệu hiện tại có hai loại quan hệ chính:

```text
KiemSoat -MITIGATES-> RuiRo
RuiRo -OBSERVED_AS-> SuKienRuiRo
```

Ví dụ:

```text
KS-001 -MITIGATES-> RR-001
RR-001 -OBSERVED_AS-> SK-001
```

Mỗi quan hệ có:

```text
source_id
relationship_type
target_id
source
evidence_quote
confidence
verification_status
data_origin
```

Đây là file quan trọng nhất để dựng **edge** của Wiki Risk Graph.

---

# 2. Phạm vi của bài thực hành

Với 4 file hiện tại, chúng ta xây phiên bản MVP:

```text
KiemSoat
   |
   | MITIGATES
   v
RuiRo
   |
   | OBSERVED_AS
   v
SuKienRuiRo
```

Ngoài ra:

- `RuiRo.owner_unit_id` cho biết mã đơn vị sở hữu.
- `KiemSoat.owner_role_id` cho biết mã vai trò phụ trách.

Tuy nhiên bộ dữ liệu hiện tại **chưa có bảng master mô tả tên Đơn vị và Vai trò**.

Vì vậy:

- Agent được phép hiển thị `owner_unit_id` và `owner_role_id`.
- Agent **không được tự bịa tên đơn vị hoặc vai trò** từ các mã này.

Bộ dữ liệu cũng chưa chứa:

```text
VanBan
DieuKhoan
QuyTrinh
```

Do đó bài MVP chưa yêu cầu tạo các quan hệ:

```text
VanBan → DieuKhoan
VanBan → QuyTrinh
QuyTrinh → RuiRo
DieuKhoan → KiemSoat
```

Các phần này có thể bổ sung ở bài Graph RAG nâng cao sau.

---

# 3. Kiến trúc sản phẩm cần tạo

Sau bài thực hành, thư mục dự án nên có dạng:

```text
wiki-risk-graph/
│
├── data/
│   ├── risk_profiles_seed.csv
│   ├── controls_seed.csv
│   ├── risk_events_seed.csv
│   └── relationships_seed.csv
│
├── wiki/
│   ├── Home.md
│   ├── risks/
│   ├── controls/
│   └── events/
│
├── outputs/
│   ├── entities.csv
│   ├── relations.csv
│   └── wiki_validation_report.md
│
├── scripts/
│   ├── inspect_data.py
│   ├── build_entities.py
│   ├── build_wiki.py
│   └── validate_wiki.py
│
├── cypher/
│   ├── schema.cypher
│   └── demo_queries.cypher
│
├── requirements.txt
└── README.md
```

Không bắt buộc Agent phải tạo đúng tuyệt đối cấu trúc trên nếu nó đề xuất một cấu trúc đơn giản hơn và hợp lý.

---

# 4. Cách làm theo Vibe Coding

Không giao toàn bộ dự án bằng một prompt duy nhất.

Thực hiện lần lượt 6 prompt.

Sau mỗi prompt:

1. Đọc phần Agent báo cáo.
2. Xem file Agent vừa tạo.
3. Chạy lệnh Agent đề xuất.
4. Kiểm tra output.
5. Nếu có lỗi, đưa nguyên traceback cho Agent và yêu cầu sửa.
6. Chỉ chuyển sang bước tiếp theo khi bước hiện tại chạy được.

---

# Bước 1 — Cho Agent đọc và hiểu dữ liệu

## Mục tiêu

Chưa viết hệ thống.

Yêu cầu Agent kiểm tra dữ liệu trước để tránh code dựa trên giả định sai.

## Prompt 1

```text
Bạn là AI Coding Agent hỗ trợ tôi xây dựng một Wiki Risk Graph phục vụ đào tạo.

Hãy kiểm tra toàn bộ project hiện tại, đặc biệt 4 file:

- data/risk_profiles_seed.csv
- data/controls_seed.csv
- data/risk_events_seed.csv
- data/relationships_seed.csv

NHIỆM VỤ:

1. Đọc thật sự 4 file CSV.
2. Báo cáo:
   - số dòng của từng file;
   - tên các cột;
   - khóa chính;
   - khóa tham chiếu;
   - các loại relationship_type;
   - số giá trị null;
   - duplicate;
   - khóa tham chiếu bị thiếu nếu có.
3. Giải thích dữ liệu này có thể tạo những loại node và edge nào.
4. Chỉ rõ dữ liệu nào chưa có, tuyệt đối không tự bịa.
5. Đề xuất kiến trúc MVP đơn giản cho:

   KiemSoat -> RuiRo -> SuKienRuiRo

6. Tạo scripts/inspect_data.py để tôi có thể chạy lại việc kiểm tra.

Không xây Wiki ở bước này.

Sau khi tạo script, hãy chạy script và báo cáo kết quả thực tế.
```

## Kết quả cần kiểm tra

Người học cần nhìn thấy Agent xác định được:

```text
RuiRo
KiemSoat
SuKienRuiRo
```

và hai quan hệ:

```text
MITIGATES
OBSERVED_AS
```

Agent cũng phải phát hiện:

```text
owner_unit_id
owner_role_id
```

chỉ là mã tham chiếu và chưa có master data tương ứng.

---

# Bước 2 — Chuẩn hóa dữ liệu thành Node và Edge

## Mục tiêu

Biến 4 CSV nghiệp vụ thành hai bảng chuẩn:

```text
entities.csv
relations.csv
```

## Prompt 2

```text
Tiếp tục project Wiki Risk Graph.

Dựa trên 4 file CSV mà bạn vừa kiểm tra, hãy xây bước chuẩn hóa dữ liệu.

Tạo script:

scripts/build_entities.py

YÊU CẦU:

1. Đọc:
   - risk_profiles_seed.csv
   - controls_seed.csv
   - risk_events_seed.csv
   - relationships_seed.csv

2. Chuẩn hóa thành:

outputs/entities.csv

Schema tối thiểu:

id
type
name
description
source_file
data_origin
verification_status

3. Mapping:

risk_profiles_seed.csv -> type = RuiRo
controls_seed.csv -> type = KiemSoat
risk_events_seed.csv -> type = SuKienRuiRo

4. Giữ các thuộc tính nghiệp vụ cần thiết của từng loại entity.
Có thể bổ sung các cột riêng nhưng không được làm mất dữ liệu gốc quan trọng.

5. Tạo:

outputs/relations.csv

từ relationships_seed.csv.

Phải giữ tối thiểu:

source_id
relationship_type
target_id
source
evidence_quote
confidence
verification_status
data_origin

6. Kiểm tra source_id và target_id đều tồn tại trong entities.csv.

7. Không tự sinh thêm quan hệ.

8. Không tự đổi PROPOSED thành VERIFIED.

9. Không suy luận tên đơn vị từ owner_unit_id.
10. Không suy luận tên vai trò từ owner_role_id.

Sau khi code xong:

- chạy script;
- in số entity theo từng type;
- in số relation theo từng relationship_type;
- báo lỗi nếu có orphan reference.

Không chỉ viết code. Hãy chạy và kiểm tra output.
```

## Kết quả mong đợi

Ví dụ:

```text
entities.csv
```

có các loại:

```text
RuiRo
KiemSoat
SuKienRuiRo
```

và:

```text
relations.csv
```

có:

```text
MITIGATES
OBSERVED_AS
```

---

# Bước 3 — Sinh Wiki Markdown

## Mục tiêu

Mỗi entity trở thành một trang Wiki.

Các trang liên kết với nhau bằng:

```text
[[wikilink]]
```

## Prompt 3

```text
Tiếp tục project Wiki Risk Graph.

Bây giờ hãy tạo Wiki Markdown từ:

outputs/entities.csv
outputs/relations.csv

Tạo script:

scripts/build_wiki.py

và sinh cấu trúc:

wiki/
├── Home.md
├── risks/
├── controls/
└── events/

QUY TẮC:

1. Mỗi RuiRo tạo một trang trong wiki/risks/.
2. Mỗi KiemSoat tạo một trang trong wiki/controls/.
3. Mỗi SuKienRuiRo tạo một trang trong wiki/events/.

4. Mỗi trang phải có YAML frontmatter tối thiểu:

---
id:
type:
verification_status:
data_origin:
---

5. Trang RuiRo phải hiển thị nếu dữ liệu có:

- tên;
- mô tả;
- category;
- cause;
- event;
- impact;
- inherent_level;
- residual_level;
- owner_unit_id;
- kiểm soát liên quan;
- sự kiện liên quan.

6. Kiểm soát liên quan phải dùng Obsidian wikilink:

[[Tên kiểm soát]]

7. Sự kiện liên quan cũng dùng wikilink.

8. Trang KiemSoat phải có wikilink tới RuiRo mà nó MITIGATES.

9. Trang SuKienRuiRo phải có wikilink về RuiRo tương ứng.

10. Mỗi liên kết được dựng từ relation phải hiển thị:

- relationship_type;
- evidence_quote;
- verification_status.

11. Tuyệt đối không tự tạo quan hệ không có trong relations.csv.

12. Không tự bịa tên owner_unit_id hoặc owner_role_id.

13. Tên file phải được xử lý an toàn nhưng wikilink phải hoạt động chính xác trong Obsidian.

14. Tạo wiki/Home.md làm trang bắt đầu.

Home.md phải có:

- link tới danh sách rủi ro;
- link tới danh sách kiểm soát;
- link tới danh sách sự kiện;
- thống kê số node và edge.

Sau khi code xong hãy chạy script và báo cáo:

- số trang Wiki đã tạo;
- số wikilink;
- ví dụ đường đi:
  KiemSoat -> RuiRo -> SuKienRuiRo.
```

---

# Bước 4 — Kiểm tra Wiki trước khi mở Obsidian

## Mục tiêu

Không tin rằng Agent tạo file xong là đúng.

Phải kiểm tra:

```text
broken link
duplicate
orphan page
missing target
```

## Prompt 4

```text
Hãy kiểm thử Wiki Risk Graph vừa tạo.

Tạo:

scripts/validate_wiki.py

và output:

outputs/wiki_validation_report.md

Script phải kiểm tra tối thiểu:

1. Tổng số file Markdown.
2. Tổng số wikilink.
3. Wikilink trỏ tới trang không tồn tại.
4. Entity bị trùng ID.
5. Trang có ID nhưng không tồn tại trong entities.csv.
6. Relation có source hoặc target không tồn tại.
7. RuiRo không có bất kỳ KiemSoat nào.
8. RuiRo không có bất kỳ SuKienRuiRo nào.
9. Trang không có liên kết với trang khác (orphan page).

Không sửa dữ liệu bằng cách bịa thêm quan hệ.

Nếu phát hiện lỗi do code build_wiki.py gây ra:
- sửa code;
- build lại Wiki;
- chạy validate lại.

Mục tiêu cuối cùng là báo cáo rõ lỗi còn lại nào là lỗi dữ liệu và lỗi nào là lỗi chương trình.
```

## Sau khi chạy

Mở:

```text
outputs/wiki_validation_report.md
```

và kiểm tra kết quả.

---

# Bước 5 — Mở bằng Obsidian

## Mục tiêu

Quan sát Knowledge Graph bằng giao diện trực quan.

## Cách làm

1. Mở Obsidian.
2. Chọn:

```text
Open folder as vault
```

3. Chọn thư mục:

```text
wiki/
```

4. Mở:

```text
Home.md
```

5. Mở **Graph View**.

Người học cần quan sát các cụm:

```text
KiemSoat
    ↓
RuiRo
    ↓
SuKienRuiRo
```

## Câu hỏi quan sát

- Một rủi ro có bao nhiêu kiểm soát?
- Một kiểm soát đang giảm thiểu rủi ro nào?
- Một rủi ro đã xuất hiện thành những sự kiện nào?
- Có node nào đứng một mình không?
- Có rủi ro nào chưa có kiểm soát không?

---

# Bước 6 — Chuẩn bị dữ liệu cho Neo4j

## Mục tiêu

Sử dụng cùng dữ liệu Wiki để dựng Knowledge Graph trong Neo4j.

## Prompt 6

```text
Tiếp tục Wiki Risk Graph.

Từ:

outputs/entities.csv
outputs/relations.csv

hãy tạo:

cypher/schema.cypher
cypher/demo_queries.cypher

Nếu môi trường đã có Neo4j và Python driver, tạo thêm:

scripts/load_neo4j.py

YÊU CẦU:

1. Node tối thiểu:

:RuiRo
:KiemSoat
:SuKienRuiRo

2. Edge:

(:KiemSoat)-[:MITIGATES]->(:RuiRo)

(:RuiRo)-[:OBSERVED_AS]->(:SuKienRuiRo)

3. Dùng id làm khóa duy nhất.

4. Dùng MERGE để chạy lại không tạo duplicate.

5. Python phải dùng parameterized Cypher.

6. Không hard-code password.

Đọc cấu hình từ .env:

NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
NEO4J_DATABASE

7. Nếu Neo4j chưa chạy thì báo hướng dẫn rõ ràng, không làm hỏng các bước Wiki trước đó.

Tạo demo query cho:

A. Xem toàn bộ graph.

B. Tìm kiểm soát giảm thiểu một rủi ro.

C. Tìm sự kiện của một rủi ro.

D. Tìm đường:

KiemSoat -> RuiRo -> SuKienRuiRo

E. Tìm rủi ro không có kiểm soát.

F. Tìm relation chưa VERIFIED.

Sau khi hoàn thành, cập nhật README.md với đúng thứ tự lệnh chạy project.
```

---

# 5. Kết quả cuối cùng

Người học cần tạo được luồng:

```text
CSV
 ↓
Chuẩn hóa
 ↓
entities.csv + relations.csv
 ↓
Wiki Markdown
 ↓
Obsidian Graph View
 ↓
Neo4j
```

Đồ thị MVP:

```text
KiemSoat
   |
   | MITIGATES
   v
RuiRo
   |
   | OBSERVED_AS
   v
SuKienRuiRo
```

---

# 6. Những điều Agent tuyệt đối không được làm

Không:

- bịa quan hệ;
- bịa nguồn;
- bịa tên đơn vị từ `owner_unit_id`;
- bịa tên vai trò từ `owner_role_id`;
- đổi dữ liệu gốc;
- tự chuyển quan hệ sang VERIFIED;
- tạo node chỉ để làm graph trông đẹp hơn;
- che giấu lỗi validation.

Nếu thiếu dữ liệu, phải ghi rõ:

```text
Chưa có dữ liệu.
```

---

# 7. Mở rộng sau khi hoàn thành MVP

Bộ 4 CSV hiện tại phù hợp để học viên hiểu:

```text
Entity
Relationship
Wiki-link
Knowledge Graph
Graph traversal
Evidence
Verification status
```

Sau khi học viên làm được MVP, có thể bổ sung thêm dữ liệu:

```text
units.csv
roles.csv
processes.csv
documents.csv
clauses.csv
```

Khi đó ontology có thể mở rộng thành:

```text
DonVi -OWNS-> RuiRo

VaiTro -PERFORMS-> KiemSoat

QuyTrinh -EXPOSES_TO-> RuiRo

VanBan -CONTAINS-> DieuKhoan

DieuKhoan -REQUIRES-> KiemSoat

VanBan -GOVERNS-> QuyTrinh
```

Và tạo đường đi đầy đủ hơn:

```text
VanBan
   ↓
DieuKhoan
   ↓
KiemSoat
   ↓
RuiRo
   ↓
SuKienRuiRo
```

Đây là bước chuyển tiếp tự nhiên từ **Wiki Risk Graph** sang **Graph RAG và Multi-hop Reasoning**.

---

# 8. Yêu cầu nộp bài

Người học nộp:

```text
scripts/inspect_data.py
scripts/build_entities.py
scripts/build_wiki.py
scripts/validate_wiki.py

outputs/entities.csv
outputs/relations.csv
outputs/wiki_validation_report.md

wiki/

cypher/schema.cypher
cypher/demo_queries.cypher

README.md
```

Nếu lớp có Neo4j:

```text
scripts/load_neo4j.py
```

---

# 9. Tiêu chí đạt

Bài được xem là đạt khi:

- 4 CSV được đọc đúng.
- Không có quan hệ bị bịa thêm.
- Node và edge được tạo từ dữ liệu thực tế.
- Wiki có liên kết chéo hoạt động.
- Có thể mở `wiki/` bằng Obsidian.
- Graph View thể hiện được quan hệ giữa kiểm soát, rủi ro và sự kiện.
- Không có broken link do lỗi chương trình.
- `evidence_quote` và `verification_status` được giữ lại.
- Có thể truy vết từ một quan hệ về dữ liệu nguồn.
- Nếu dùng Neo4j, graph có thể được truy vấn theo đường nhiều bước.
