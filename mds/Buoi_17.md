# BÀI THỰC HÀNH BUỔI 17
# RBAC, Audit Trail và AI Compliance Gap Checker bằng Vibe Coding

## Mục tiêu

Buổi 17 không xây lại RAG từ đầu. Học viên tái sử dụng dữ liệu và Secure Retrieval đã hình thành ở Buổi 16, đồng thời tận dụng Hybrid Search, Reranking, Metadata và Knowledge Graph từ các buổi trước để bổ sung ba năng lực:

```text
RBAC → chỉ truy xuất tài liệu đúng quyền
Audit Trail → truy vết câu hỏi, nguồn, kết quả
Compliance Gap → so sánh quy định nội bộ với Thông tư NHNN
```

Sản phẩm cuối buổi:

```text
AI tra cứu quy định nội bộ có phân quyền
+ Audit log
+ AI Compliance Gap Checker
+ Streamlit demo
```

Tài liệu thực hành bám mục tiêu Module 3: dùng RAG có trích dẫn, tăng cường bằng Hybrid/Graph và bổ sung governance/security.

---

# 1. Hai use case chính

## Use Case 1 — Tra cứu quy định nội bộ

```text
Câu hỏi
→ User Role
→ RBAC Filter
→ Hybrid + Rerank
→ Top-k được phép xem
→ Answer + Citation
→ Audit Log
```

Điểm quan trọng:

> Tài liệu không có quyền phải bị loại trước khi đưa vào context cho LLM.

## Use Case 2 — Compliance Gap Analysis

```text
Yêu cầu NHNN
→ tìm điều khoản nội bộ liên quan
→ so sánh evidence hai phía
→ DAP_UNG / THIEU / CHENH_LECH / CHUA_DU_BANG_CHUNG
→ Human Review
```

Không dùng kết quả AI như kết luận kiểm toán cuối cùng.

---

# 2. Nguyên tắc bắt buộc

- Không sửa dữ liệu nguồn.
- Không rebuild Hybrid/Rerank nếu đã có.
- RBAC phải lọc trước retrieval/context.
- Không tự bịa chính sách phân quyền Agribank; policy trong bài là mô phỏng.
- Không tự bịa gap.
- Mỗi gap phải có evidence/citation hai phía nếu có.
- Nếu chưa đủ bằng chứng, trả `CHUA_DU_BANG_CHUNG`.
- Mọi finding phải `NEEDS_HUMAN_REVIEW`.

---

# 3. Cấu trúc project đề xuất

```text
thuchanh/
├── buoi_16/
│   └── data/
│       └── processed/
│           ├── chunks_normalized.csv
│           └── chunks_secure.csv
└── buoi_17/
    ├── .env
    ├── README.md
    ├── config/
    │   └── rbac_policy.json
    ├── scripts/
    │   ├── inspect_dependencies.py
    │   ├── rbac.py
    │   ├── secure_retrieval.py
    │   ├── audit_logger.py
    │   ├── internal_lookup.py
    │   ├── compliance_gap.py
    │   └── final_validation.py
    ├── outputs/
    │   ├── dependency_report.md
    │   ├── rbac_test_report.md
    │   ├── audit_log.jsonl
    │   ├── internal_lookup_demo.md
    │   ├── compliance_gap_results.csv
    │   ├── compliance_gap_report.md
    │   └── final_validation_report.md
    └── app.py
```

---

# 4. `.env`

Đặt tại:

```text
buoi_17/.env
```

Ví dụ (sử dụng Gemini API Key miễn phí với model `gemini-2.5-flash`):

```env
# Dữ liệu đầu vào Buổi 16 nằm ở thư mục cùng cấp
SOURCE_SECURE_CSV=../buoi_16/data/processed/chunks_secure.csv
SOURCE_NORMALIZED_CSV=../buoi_16/data/processed/chunks_normalized.csv

NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_PASSWORD
NEO4J_DATABASE=neo4j

# Gemini API Key free tier & model
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_FREE
LLM_API_KEY=YOUR_GEMINI_API_KEY_FREE
LLM_MODEL=gemini-2.5-flash

APP_ENV=training
```

Thêm vào `.gitignore`:

```text
.env
*.key
__pycache__/
```

Không đưa key thật vào tài liệu nộp bài.

---

# 5. Cách làm Vibe Coding

```text
Prompt
→ Agent đọc project thật
→ chỉ làm đúng một bước
→ chạy thật
→ kiểm tra output
→ PASS mới sang bước sau
```

---

# PROMPT SETUP — Kiểm tra môi trường

```text
Kiểm tra giúp tôi môi trường cho Buổi 17.

Folder Buổi 17 nằm cùng cấp với Buổi 16.

Dữ liệu đầu vào chính:
../buoi_16/data/processed/chunks_secure.csv

Dữ liệu đối chiếu:
../buoi_16/data/processed/chunks_normalized.csv

Kiểm tra:
- Python và virtual environment;
- file chunks_secure.csv có đọc được không;
- file chunks_normalized.csv có đọc được không;
- chunks_secure.csv có đúng 14 cột và có allowed_roles không;
- chunks_normalized.csv có đúng 13 cột không;
- số dòng của hai file có khớp không;
- .env của buoi_17;
- code SecureRetriever từ Buổi 16 nằm ở đâu và import được không;
- Neo4j chỉ kiểm tra nếu bước sau thực sự dùng.

Chỉ kiểm tra, chưa sửa dữ liệu Buổi 16 và chưa viết lại retriever.

Cuối cùng báo:
ENVIRONMENT READY: YES / NO
SOURCE DATA READY: YES / NO
SECURE RETRIEVER FOUND: YES / NO
```

# PROMPT 0 — Đọc dữ liệu và code Buổi 16 trước khi làm

```text
Đọc Buổi 16 trước khi làm Buổi 17.

Dùng đúng:
../buoi_16/data/processed/chunks_secure.csv
../buoi_16/data/processed/chunks_normalized.csv

Kiểm tra và báo chính xác:
- số dòng;
- danh sách cột;
- chunk_id;
- document_id;
- citation;
- title;
- loai_van_ban;
- co_quan_ban_hanh;
- ngay_ban_hanh;
- allowed_roles.

So sánh hai file và xác nhận:
chunks_secure.csv = chunks_normalized.csv + allowed_roles
hay có khác biệt dữ liệu nào khác.

Tìm code SecureRetriever mà Buổi 16 đang dùng.
Báo:
- file/module;
- hàm/class chính;
- input role;
- output;
- nó filter allowed_roles trước hay sau retrieval;
- document_id/chunk_id/citation có được giữ không.

Không tạo policy mới.
Không viết retriever mới.
Không sửa file Buổi 16.

Tạo:
buoi_17/outputs/dependency_report.md

Cuối report:
SOURCE DATA: PASS / FAIL
RBAC DATA AVAILABLE: YES / NO
SECURE RETRIEVER REUSABLE: YES / NO
REUSE PLAN: ...
```

## Dữ liệu kỳ vọng

`chunks_secure.csv` hiện có:

```text
787 dòng
14 cột
```

và thêm trường:

```text
allowed_roles
```

so với `chunks_normalized.csv`.

---

# PROMPT 1 — Kiểm tra và tái sử dụng RBAC từ `allowed_roles`

```text
Tiếp tục Buổi 17.

Không tạo RBAC policy mới nếu dữ liệu hiện tại đã đủ.

Dùng:
../buoi_16/data/processed/chunks_secure.csv

Kiểm tra:
- allowed_roles có những role nào;
- số chunk theo từng role;
- chunk nào cho nhiều role;
- chunk nào hạn chế quyền;
- format allowed_roles có parse ổn định không;
- unknown role được xử lý thế nào.

Sau đó kiểm tra SecureRetriever của Buổi 16:
- có đọc allowed_roles không;
- có loại chunk không được phép trước retrieval/context không.

Chạy cùng một query với:
Admin
HR
Risk_Manager
Staff
Guest

Không sửa chunks_secure.csv.

Nếu SecureRetriever đã đúng:
reuse nguyên trạng.

Nếu chỉ khác interface:
tạo adapter trong buoi_17, không copy retriever.

Tạo:
buoi_17/outputs/rbac_reuse_report.md

Cuối:
RBAC REUSED: YES / NO
FILTER BEFORE RETRIEVAL: PASS / FAIL
UNKNOWN ROLE DEFAULT DENY: PASS / FAIL
```

# PROMPT 2 — Secure Retrieval bằng code Buổi 16

```text
Tái sử dụng SecureRetriever của Buổi 16 cho Buổi 17.

Không viết retriever mới.

Nếu cần thống nhất output thì tạo:
buoi_17/scripts/secure_retrieval_adapter.py

Adapter chỉ gọi lại SecureRetriever cũ và chuẩn hóa kết quả thành:
- rank;
- chunk_id;
- document_id;
- title;
- article;
- citation;
- allowed_roles;
- access_decision;
- retrieval method.

Chạy test chứng minh:
1. role được phép nhận được chunk;
2. role không được phép không nhận chunk đó;
3. unauthorized chunk không xuất hiện trong context;
4. citation/document_id/chunk_id không bị mất.

Dùng dữ liệu:
../buoi_16/data/processed/chunks_secure.csv

Xuất:
buoi_17/outputs/secure_retrieval_test.md

Cuối:
SECURE RETRIEVAL REUSE: PASS / FAIL
NO UNAUTHORIZED CONTEXT: PASS / FAIL
CITATION PRESERVED: PASS / FAIL
```

# PROMPT 3 — Audit Trail

```text
Thêm audit trail cho Buổi 17.

Tạo:
buoi_17/scripts/audit_logger.py
buoi_17/outputs/audit_log.jsonl

Mỗi request ghi:
- timestamp UTC;
- request_id;
- user_id_demo;
- user_role;
- action;
- query;
- retrieval method;
- retrieved document IDs;
- retrieved chunk IDs;
- citation IDs;
- số candidate bị RBAC loại;
- status SUCCESS / DENIED / ERROR.

Không ghi:
password, API key, secret.

Request bị DENIED vẫn phải có audit event.

Chạy 3 request demo:
1. allowed;
2. denied;
3. bình thường.

Cuối:
AUDIT TRAIL: PASS / FAIL
```

---

# PROMPT 4 — Demo Encryption cục bộ

```text
Tạo demo encryption nhỏ cho Buổi 17.

Mục tiêu chỉ minh họa bảo vệ dữ liệu at-rest, không tuyên bố production-ready.

Dùng cryptography/Fernet nếu phù hợp.

Tạo:
buoi_17/scripts/encryption_demo.py
buoi_17/outputs/encryption_demo_report.md

Yêu cầu:
- key không hard-code;
- *.key nằm trong .gitignore;
- encrypt một file audit demo;
- decrypt lại và so khớp;
- không sửa dữ liệu nguồn.

Cuối report:
ENCRYPT: PASS / FAIL
DECRYPT MATCH: PASS / FAIL
PRODUCTION READY: NO
```

Giải thích với học viên: hệ thống thật còn cần TLS, key management, rotation, backup và IAM.

---

# PROMPT 5 — Use Case 1: AI tra cứu quy định nội bộ

```text
Xây Use Case 1 cho Buổi 17:
AI tra cứu quy định nội bộ.

Tái sử dụng SecureRetriever Buổi 16 thông qua adapter của Buổi 17 nếu cần.

Tạo:
buoi_17/scripts/internal_lookup.py

Input:
- question;
- user_role;
- top_k.

Output:
- answer;
- citations;
- document_id/chunk_id;
- access scope;
- request_id.

LLM chỉ được trả lời từ chunk sau RBAC.

Nếu context không đủ:
"Không tìm thấy đủ thông tin trong phạm vi tài liệu được phép truy cập."

Không được:
- dùng knowledge ngoài context để bù;
- lộ tài liệu DENY;
- tạo citation giả.

Chạy ít nhất 3 câu hỏi từ corpus.

Xuất:
buoi_17/outputs/internal_lookup_demo.md

Cuối report:
CITATION: PASS / FAIL
RBAC: PASS / FAIL
AUDIT: PASS / FAIL
```

---

# PROMPT 6 — Kiểm tra dữ liệu có đủ cho Gap Analysis không

```text
Chuẩn bị dữ liệu cho Compliance Gap Checker.

Nguồn:
../buoi_16/data/processed/chunks_secure.csv

Đọc các trường:
- document_id;
- title;
- so_ky_hieu;
- loai_van_ban;
- co_quan_ban_hanh;
- ngay_ban_hanh;
- article;
- citation.

Phân loại từng document theo evidence thật:
- EXTERNAL_REQUIREMENT nếu là Thông tư/Nghị định/Luật/cơ quan bên ngoài;
- INTERNAL_POLICY chỉ khi dữ liệu thật chứng minh đó là văn bản nội bộ.

Không được gọi một Thông tư/Nghị định khác là "quy định nội bộ" chỉ để chạy demo.

Tạo:
buoi_17/outputs/gap_input_catalog.md

Report cần có:
- tổng số document;
- document_id;
- title;
- loại văn bản;
- cơ quan ban hành;
- external/internal classification;
- evidence dùng để phân loại.

Nếu không có INTERNAL_POLICY thật:
ghi rõ:
COMPLIANCE GAP DATA: INSUFFICIENT
DATA GAP: INTERNAL POLICY NOT FOUND

và không kết luận compliance trên corpus này.

Nếu có đủ hai phía:
COMPLIANCE GAP DATA: READY
```

# PROMPT 7 — Xây AI Compliance Gap Checker

```text
Chỉ chạy AI Compliance Gap Checker nếu Prompt 6 báo COMPLIANCE GAP DATA: READY.

Nếu Prompt 6 báo dữ liệu chưa đủ, không tự tạo văn bản và không sinh kết luận giả; thay vào đó tạo report DATA GAP cho use case này.

Nếu đủ dữ liệu, xây/tái sử dụng AI Compliance Gap Checker cho Buổi 17.

Tạo:
buoi_17/scripts/compliance_gap.py

Luồng:
1. nhận một requirement/điều khoản NHNN;
2. Hybrid + Rerank tìm điều khoản nội bộ liên quan trong phạm vi được phép;
3. nếu Neo4j có quan hệ hữu ích thì dùng graph để bổ sung candidate, không bịa edge;
4. tạo evidence package:
   - external requirement;
   - external citation;
   - internal evidence;
   - internal citation;
5. phân loại:
   - DAP_UNG
   - THIEU
   - CHENH_LECH
   - CHUA_DU_BANG_CHUNG
6. reason ngắn;
7. confidence;
8. review_status = NEEDS_HUMAN_REVIEW.

Không kết luận chỉ từ similarity score.
Không gán DAP_UNG nếu không có internal evidence.
Không gán THIEU chỉ vì retriever chưa tìm thấy.

Xuất:
buoi_17/outputs/compliance_gap_results.csv
buoi_17/outputs/compliance_gap_report.md

Cuối report:
GAP CHECKER: PASS / FAIL
HUMAN REVIEW REQUIRED: YES
```

Schema tối thiểu:

```text
gap_id
external_document_id
external_chunk_id
external_requirement
external_citation
internal_document_id
internal_chunk_id
internal_evidence
internal_citation
classification
reason
confidence
review_status
request_id
```

---

# PROMPT 8 — Kiểm tra vai trò Knowledge Graph

```text
Kiểm tra Knowledge Graph hiện có có giúp Compliance Gap Checker không.

Không tự tạo edge.

Tìm relationship type thực sự tồn tại trong Neo4j và đánh giá:
- relation nào giúp nối văn bản/điều khoản;
- relation nào chỉ là CONTAINS/NEXT;
- relation nào không liên quan.

Nếu có giá trị:
thêm graph candidate expansion vào compliance_gap.py.

Nếu không:
giữ Hybrid + Rerank và ghi GRAPH NOT USED FOR GAP MATCHING.

Xuất:
buoi_17/outputs/graph_gap_integration_report.md

Báo:
GRAPH USED: YES / NO
và lý do.
```

Ý cần nhớ:

```text
Hybrid = tìm nội dung liên quan
KG = mở rộng theo quan hệ đã biết
Gap Checker = so sánh evidence
```

---

# PROMPT 9 — Streamlit UI

```text
Tạo giao diện Streamlit cho Buổi 17 tại:

buoi_17/app.py

Không viết lại logic retrieval/gap trong app.py.

Sidebar:
- User ID demo;
- User Role;
- trạng thái Neo4j nếu dùng.

Tab 1: TRA CỨU QUY ĐỊNH
- question;
- Top-k;
- answer;
- citation;
- document/chunk;
- request_id;
- access decision.

Tab 2: COMPLIANCE GAP CHECKER
- requirement NHNN;
- bảng external citation;
- internal citation;
- classification;
- reason;
- confidence;
- review_status.

Tab 3: AUDIT
- chỉ hiển thị audit event phù hợp với role demo;
- không hiển thị secret.

Banner:
"Demo đào tạo — kết quả AI cần kiểm toán viên xác minh."

Nếu access denied:
không hiển thị snippet/citation bị cấm.

Chạy:
streamlit run buoi_17/app.py
```

Giao diện mong đợi:

```text
┌───────────────────────────────────────────────┐
│ SECURE RAG & COMPLIANCE — BUỔI 17            │
├───────────────────────────────────────────────┤
│ User: demo01       Role: KiemToanVien         │
├───────────────────────────────────────────────┤
│ [TRA CỨU] [GAP CHECKER] [AUDIT]              │
├───────────────────────────────────────────────┤
│ Question / Requirement                        │
│ [...........................................] │
│                                     [RUN]     │
├───────────────────────────────────────────────┤
│ Answer / Evidence                             │
│ Citation                                      │
│ Access Decision                               │
│ Request ID                                    │
├───────────────────────────────────────────────┤
│ NHNN | INTERNAL | STATUS                      │
│ ...  | ...      | CHENH_LECH                  │
│              NEEDS_HUMAN_REVIEW               │
└───────────────────────────────────────────────┘
```

---

# PROMPT 10 — Security Tests

```text
Đóng vai tester và kiểm thử Buổi 17.

Tạo:
buoi_17/scripts/security_tests.py
buoi_17/outputs/security_test_report.md

Test:
1. role được phép → PASS;
2. role không được phép → không lộ text/citation;
3. tài liệu bị cấm không vào LLM context;
4. unknown role → DENY;
5. audit ghi SUCCESS và DENIED;
6. log không chứa password/API key;
7. citation tồn tại;
8. gap có evidence hoặc CHUA_DU_BANG_CHUNG;
9. mọi gap result NEEDS_HUMAN_REVIEW;
10. Neo4j down thì báo thật, không giả.

Chạy test thật.

Cuối:
SECURITY TESTS: PASS / FAIL
```

---

# PROMPT 11 — Final Validation

```text
Audit toàn bộ project Buổi 17.

Kiểm tra:
- không sửa source data;
- reuse Hybrid/Rerank cũ;
- RBAC filter trước retrieval/context;
- không unauthorized leakage;
- audit trail đầy đủ;
- secret không hard-code;
- encryption demo ghi rõ không production;
- internal lookup có citation;
- compliance gap có citation hai phía;
- classification đúng enum;
- không dùng "không retrieve thấy" để tự kết luận THIEU;
- human review luôn được yêu cầu;
- Streamlit chạy;
- Neo4j đúng trạng thái thật.

Tạo:
buoi_17/outputs/final_validation_report.md

Cuối file:

RBAC: PASS / FAIL
SECURE RETRIEVAL: PASS / FAIL
AUDIT TRAIL: PASS / FAIL
CITATION: PASS / FAIL
COMPLIANCE GAP: PASS / FAIL
HUMAN REVIEW GUARDRAIL: PASS / FAIL
STREAMLIT: PASS / FAIL
WORKSPACE ISOLATION: PASS / FAIL

READY FOR DEMO: YES / NO
```

---

# 5A. Nguồn dữ liệu chính xác trên máy học viên

Theo cấu trúc hiện tại:

```text
D:\agribank\thuchanh\
├── buoi_16\
│   └── data\
│       └── processed\
│           ├── chunks_normalized.csv
│           └── chunks_secure.csv
│
└── buoi_17\
```

Từ code chạy trong `buoi_17/`, đường dẫn tương đối phải ưu tiên:

```text
../buoi_16/data/processed/chunks_secure.csv
../buoi_16/data/processed/chunks_normalized.csv
```

Không copy hai file này sang Buổi 17 nếu không cần.

Buổi 17 phải đọc dữ liệu Buổi 16 ở chế độ read-only.

---

# 6. Trình tự demo cuối buổi

1. **Cùng query, hai role** → chứng minh RBAC.
2. **Audit log** → chứng minh truy vết.
3. **Tra cứu quy định nội bộ** → Answer + Citation.
4. **Compliance Gap Checker** → evidence hai phía + classification.
5. **Human Review** → nhấn mạnh AI không tự kết luận kiểm toán.

---

# 7. Các câu hỏi nên hỏi học viên

**RBAC lọc trước hay sau khi LLM nhìn thấy tài liệu?**  
→ Trước.

**Không retrieve thấy điều khoản nội bộ có được kết luận ngay THIẾU không?**  
→ Không.

**Gap Checker cần citation mấy phía?**  
→ Hai phía.

**Audit trail dùng để làm gì?**  
→ Truy vết ai hỏi gì, hệ thống dùng nguồn nào và kết quả ra sao.

**Gap Checker có phải kết luận kiểm toán cuối cùng không?**  
→ Không, cần human review.

---

# 8. Những điều Agent tuyệt đối không được làm

```text
- sửa corpus nguồn;
- bịa policy thật;
- search tài liệu bị cấm rồi mới che ở UI;
- gửi chunk bị cấm vào LLM;
- log API key/password;
- bịa citation;
- bịa relation Neo4j;
- coi retrieval fail = THIEU;
- tự VERIFIED finding;
- tuyên bố encryption demo là production security;
- xóa graph buổi khác;
- rebuild RAG không cần thiết.
```

---

# 9. Kết quả cần nộp

```text
buoi_17/
├── config/rbac_policy.json
├── scripts/rbac.py
├── scripts/secure_retrieval.py
├── scripts/audit_logger.py
├── scripts/internal_lookup.py
├── scripts/compliance_gap.py
├── scripts/final_validation.py
├── outputs/dependency_report.md
├── outputs/rbac_test_report.md
├── outputs/audit_log.jsonl
├── outputs/internal_lookup_demo.md
├── outputs/compliance_gap_results.csv
├── outputs/compliance_gap_report.md
├── outputs/security_test_report.md
├── outputs/final_validation_report.md
├── app.py
└── README.md
```

---

# 10. Tiêu chí đạt

```text
☐ Không rebuild RAG không cần thiết.
☐ RBAC hoạt động ở tầng retrieval.
☐ Không lộ unauthorized context.
☐ Audit log ghi được request.
☐ Không log secret.
☐ Tra cứu nội bộ có citation.
☐ Gap Checker có evidence hai phía.
☐ Có DAP_UNG / THIEU / CHENH_LECH / CHUA_DU_BANG_CHUNG.
☐ Mọi finding cần human review.
☐ Streamlit demo chạy.
☐ Final validation PASS.
```

---

# 11. Flow toàn Buổi 17

```text
Hybrid + Rerank + Metadata + KG
            │
            ▼
           RBAC
            │
            ▼
     Secure Retrieval
       ┌────┴─────┐
       ▼          ▼
  Tra cứu      Gap Analysis
       │          │
       └────┬─────┘
            ▼
         Citation
            │
            ▼
        Audit Trail
            │
            ▼
        Human Review
```

---

# 12. Câu chốt

> “Buổi 17 chuyển hệ thống từ việc chỉ tìm đúng tài liệu sang việc kiểm soát ai được thấy tài liệu nào, truy vết hệ thống đã làm gì và hỗ trợ kiểm toán viên đối chiếu quy định bằng evidence.”

> “Trong hệ thống ngân hàng, trả lời đúng chưa đủ. Kết quả còn phải đúng quyền, có nguồn, có log và có người chịu trách nhiệm kiểm tra.”
