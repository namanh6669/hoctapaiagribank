# BUỔI 15 — Cài đặt Kiểm soát Truy cập dựa trên Vai trò (Role-Based Access Control - RBAC) ở mức Dữ liệu và Retrieval Pipeline

## Mục tiêu

Trong bài thực hành này, **học viên** sử dụng **AI Coding Agent / Vibe Coding** để thiết kế và cài đặt hệ thống kiểm soát truy cập dựa trên vai trò (RBAC) trực tiếp ở tầng dữ liệu và tích hợp vào pipeline Retrieval (Search) của RAG.

Toàn bộ quá trình thực hành sẽ diễn ra trực tiếp bên trong thư mục `buoi_15/`, sử dụng các tài nguyên dữ liệu và môi trường có sẵn tại đây để phát triển nâng cấp tính năng bảo mật.

Các mục tiêu cốt lõi:
1. **Thiết kế phân quyền dữ liệu (Học viên tự chọn vai trò)**:
   - Học viên tự chọn và thiết kế tối thiểu 3 vai trò (roles) tùy thuộc vào nghiệp vụ giả định (Ví dụ: Hệ thống tài liệu ngân hàng: `Admin`, `Risk_Manager`, `Staff`, `Guest` hoặc Hệ thống nội bộ doanh nghiệp: `Admin`, `HR_Specialist`, `Employee`).
   - Phân loại tài liệu trong tập dữ liệu quy định theo các mức bảo mật tương ứng với vai trò.
2. **Gán thuộc tính phân quyền (Security Tagging)**:
   - Viết script để gắn thẻ quyền truy cập (`allowed_roles`) vào metadata của từng văn bản/đoạn văn (chunk).
   - Nạp các thuộc tính bảo mật này vào các Node (`VanBan`, `DieuKhoan`) trong Neo4j.
3. **Nâng cấp Secure Retrieval Pipeline**:
   - Viết các truy vấn Cypher và logic tìm kiếm (BM25, Dense, Hybrid) có lọc quyền truy cập (Access Filtering) ở mức cơ sở dữ liệu và bộ nhớ.
   - Đảm bảo Cross-Encoder Reranker chỉ xếp hạng các ứng viên mà người dùng có quyền xem.
4. **Xây dựng giao diện mô phỏng phân quyền**:
   - Nâng cấp Streamlit App demo của Buổi 14 thành một phiên bản bảo mật để cho phép học viên lựa chọn đóng vai (impersonate) các vai trò khác nhau khi thực hiện truy vấn.
5. **Kiểm tra và đánh giá rò rỉ dữ liệu (Security Audit)**:
   - Viết script tự động chạy thử nghiệm bảo mật (Security Integration Tests) nhằm khẳng định người dùng ở vai trò thấp không thể tiếp cận dữ liệu nhạy cảm của vai trò cao hơn.

---

# 1. Kiến thức cần hiểu trước khi thực hành

Để cài đặt hệ thống tìm kiếm an toàn (Secure Retrieval), học viên chỉ cần nắm vững 2 khái niệm cốt lõi sau:

## 1.1. Phân quyền mức dữ liệu (Property-based Security)

Thay vì thiết lập phân quyền phức tạp trên hệ quản trị cơ sở dữ liệu (Native RBAC), chúng ta sử dụng phương pháp **gán thuộc tính bảo mật trực tiếp lên Node**:
* **Cách làm**: Thêm thuộc tính `allowed_roles` (danh sách các vai trò được phép đọc) vào các Node trong đồ thị.
  * Ví dụ: Một node `VanBan` chứa thuộc tính `allowed_roles = ["Admin", "HR"]`.
* **Ưu điểm**: Hoạt động tốt trên mọi phiên bản Neo4j (bao gồm cả bản Community Edition miễn phí) và rất dễ dàng lọc bằng câu lệnh truy vấn Cypher hoặc code Python.

---

## 1.2. Cơ chế lọc quyền truy cập (Access Filtering)

Khi người dùng thực hiện truy vấn, RAG pipeline sẽ lọc kết quả dựa trên danh sách vai trò hiện tại của người dùng (`user_roles`):

1. **Lọc trong truy vấn Cypher (Neo4j)**:
   ```cypher
   MATCH (v:VanBan)-[:CONTAINS]->(d:DieuKhoan)
   WHERE any(role IN v.allowed_roles WHERE role IN $user_roles)
   RETURN v, d
   ```
   *(Chỉ trả về các Điều khoản nằm trong Văn bản mà người dùng có quyền xem)*

2. **Lọc trong BM25 & Dense Search (Pandas/Vector Metadata)**:
   - Trước khi tính điểm BM25 hoặc tìm kiếm Vector, lọc bỏ hoàn toàn các chunk dữ liệu không thuộc quyền truy cập của người dùng.

```text
Ý tưởng lọc quyền truy cập:

    Người dùng (Vai trò: Guest)
               │
               ▼ (Gửi câu hỏi)
    [Bộ lọc quyền truy cập] ───► Loại bỏ tài liệu có allowed_roles = ["HR", "Admin"]
               │
               ▼
    [Chỉ tìm kiếm trên tài liệu chung] (allowed_roles chứa "Guest")
```

---

# 2. Quy trình thực hành và Dữ liệu sử dụng

Học viên làm việc hoàn toàn trong thư mục `buoi_15/`.
- Dữ liệu nguồn và các tệp dữ liệu đã xử lý nằm trong `buoi_15/data/`.
- Toàn bộ thông tin cấu hình kết nối cơ sở dữ liệu Neo4j sẽ được đọc từ file cấu hình của riêng học viên tại **`buoi_15/.env`** (bao gồm `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, v.v.).

## Cấu trúc project sau khi hoàn thành buổi 15:
```text
buoi_15/
│
├── data/
│   └── processed/
│       ├── chunks_normalized.csv
│       └── chunks_secure.csv       # [NEW] Chứa dữ liệu chunks đã gán quyền
│
├── src/
│   ├── ...
│   └── secure_retriever.py         # [NEW] Bộ lọc truy vấn an toàn
│
├── scripts/
│   ├── ...
│   ├── assign_security_tags.py     # [NEW] Script phân loại & gắn tag bảo mật
│   ├── secure_search_demo.py       # [NEW] CLI demo tìm kiếm an toàn
│   └── security_audit.py           # [NEW] Script tự động kiểm thử rò rỉ dữ liệu
│
├── app_secure.py                   # [NEW] Streamlit Web App nâng cấp RBAC
└── outputs/
    └── security_audit_report.md    # [NEW] Báo cáo kết quả kiểm thử an toàn
```

---

# 3. Hướng dẫn các bước thực hành & Prompts cho AI Agent

Dưới đây là chuỗi các Prompt mẫu mà **học viên** sẽ sao chép, tùy biến các vai trò tự chọn của mình và gửi cho AI Coding Agent để xây dựng hệ thống.

---

## 3.1. PROMPT SETUP — Thiết kế vai trò của học viên

> [!IMPORTANT]
> **Nhiệm vụ của học viên**: Trước khi chạy prompt bên dưới, hãy chọn danh sách vai trò mình muốn cài đặt.
> Ví dụ lựa chọn:
> - *Lựa chọn A*: `Admin`, `Staff`, `Guest`
> - *Lựa chọn B*: `Admin`, `HR_Manager`, `Risk_Officer`, `Employee`, `Guest`
> - *Lựa chọn C (Tự chọn theo ý thích)*: `Manager`, `Developer`, `Customer`

```text
Bạn là AI Coding Agent hỗ trợ tôi thực hiện bài thực hành Buổi 15: Cài đặt Kiểm soát Truy cập dựa trên Vai trò (RBAC) ở mức dữ liệu.

Toàn bộ code và cấu hình của bài này phải được triển khai trực tiếp bên trong thư mục `buoi_15/`.
Thông tin kết nối Neo4j sẽ được đọc từ file cấu hình của tôi tại `buoi_15/.env` (hoặc `.env` trong thư mục hiện hành).

Tôi tự chọn các VAI TRÒ (Roles) cho hệ thống của mình bao gồm:
[HỌC VIÊN ĐIỀN CÁC VAI TRÒ ĐÃ CHỌN VÀO ĐÂY, Ví dụ: "Admin", "HR", "Staff", "Guest"]

Hãy giúp tôi:
1. Xác nhận thư mục làm việc hiện tại là `buoi_15/`.
2. Kiểm tra lại môi trường Python, các dependencies đã cài đặt từ Buổi 14 (pandas, neo4j, sentence-transformers, streamlit).
3. Thiết lập danh sách vai trò hợp lệ này trong một cấu hình code riêng của `buoi_15/` (ví dụ tạo file `buoi_15/src/config.py` hoặc `buoi_15/roles.json`) để tránh lỗi gõ sai (typo) trong suốt quá trình triển khai. Đọc cấu hình database từ file `.env` cục bộ trong thư mục hiện tại.

Trả về kết quả kiểm tra môi trường và cấu hình vai trò dưới dạng:
RBAC SETUP CONFIRMATION
- Working Directory: buoi_15/
- Selected Roles: ...
- Database Env Path: .env
- Status: Ready to proceed
```

---

## 3.2. PROMPT 1 — Phân loại tài liệu và Gán tag bảo mật (Security Tagging)

Học viên sẽ yêu cầu Agent viết script tự động phân bổ quyền truy cập cho tập dữ liệu. Quy tắc phân bổ mẫu:
- Quyết định nhân sự, lương thưởng, kỷ luật chỉ cho `HR`, `Admin`.
- Quy định quản trị rủi ro tín dụng chỉ cho `Risk_Manager`, `Admin`.
- Nội quy lao động, quy trình chung cho `Staff`, `Guest`.

```text
Tiếp tục bài thực hành Buổi 15. Chúng ta làm việc hoàn toàn trong `buoi_15/`. Không sửa dữ liệu gốc trong `../kb+hops/`.

Nhiệm vụ: Hãy tạo script `buoi_15/scripts/assign_security_tags.py` để phân loại bảo mật cho tập dữ liệu và lưu vào file mới.

YÊU CẦU:
1. Đọc file dữ liệu đã chuẩn hóa ở buổi trước: `buoi_15/data/processed/chunks_normalized.csv`.
2. Thiết kế logic gán tag quyền dựa trên tiêu chí sau:
   - Tạo cột mới tên là `allowed_roles` chứa danh sách các vai trò (dạng chuỗi JSON hoặc list được phân tách bằng dấu phẩy) được phép truy cập chunk đó.
   - Quy tắc phân bổ:
     * Dựa trên `document_id` hoặc từ khóa trong `text` để phân loại:
       + Nếu tài liệu liên quan đến "nhân sự", "lương thưởng", "tuyển dụng", "bổ nhiệm" (hoặc mã tài liệu nhân sự): allowed_roles = ["Admin", "HR"]
       + Nếu tài liệu liên quan đến "tín dụng", "rủi ro", "hạn mức", "phê duyệt duyệt vay": allowed_roles = ["Admin", "Risk_Manager", "Staff"]
       + Các tài liệu quy định chung khác: allowed_roles = ["Admin", "HR", "Risk_Manager", "Staff", "Guest"]
3. Ghi file kết quả ra `buoi_15/data/processed/chunks_secure.csv`.
4. Viết hàm kiểm tra để đảm bảo:
   - Mọi dòng đều có ít nhất 1 role được phân quyền (không bị trống/null).
   - In ra thống kê số lượng chunk thuộc về mỗi nhóm phân quyền.
   - Hiển thị 3 mẫu dòng dữ liệu đại diện cho 3 cấp độ bảo mật khác nhau.

Hãy viết code, giải thích logic phân loại và thực thi script này.
```

---

## 3.3. PROMPT 2 — Nạp dữ liệu bảo mật vào Neo4j (Secure Graph Loading)

```text
Tiếp tục bài thực hành Buổi 15. Hãy cập nhật cơ sở dữ liệu đồ thị Neo4j của chúng ta để chứa thông tin phân quyền mới.

Nhiệm vụ: Cập nhật dữ liệu từ `buoi_15/data/processed/chunks_secure.csv` vào đồ thị Neo4j.

YÊU CẦU:
1. Đọc cấu hình kết nối Neo4j từ file `.env` cục bộ (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE).
2. Viết câu lệnh Cypher sử dụng `MERGE` để cập nhật thuộc tính `allowed_roles` vào các node `VanBan` và `DieuKhoan` (hoặc các node chunk tương ứng).
   * Lưu ý: Thuộc tính `allowed_roles` lưu trên Neo4j phải là kiểu mảng chuỗi (List of Strings) để dễ dàng kiểm tra bằng hàm `any()` hoặc `IN` của Cypher.
3. Không DETACH DELETE đồ thị hiện tại. Chỉ ghi đè/cập nhật thuộc tính trên các node có sẵn hoặc nạp mới nếu chưa tồn tại, phân biệt bằng tag `lab_session = "buoi_15"`.
4. Viết script kiểm tra nhanh sau khi nạp:
   - Viết truy vấn Cypher đếm số node có chứa thuộc tính `allowed_roles`.
   - Viết truy vấn lấy thử 1 node `VanBan` và các node `DieuKhoan` liên kết để kiểm chứng xem thông tin phân quyền đã được cập nhật chính xác chưa.

Hãy viết script `buoi_15/scripts/load_secure_kg.py`, thực thi nó và báo cáo kết quả.
```

---

## 3.4. PROMPT 3 — Xây dựng Bộ lọc Truy vấn An toàn (Secure Retrieval Pipeline)

Đây là bước quan trọng nhất của bài thực hành. AI Agent cần thiết kế một lớp bao bọc (Wrapper) hoặc nâng cấp `unified_retriever.py` để nhận tham số `user_roles`.

```text
Tiếp tục bài thực hành Buổi 15. Hãy nâng cấp hệ thống tìm kiếm (Retrieval Pipeline) thành một hệ thống tìm kiếm an toàn (Secure Retrieval) trực tiếp trong thư mục `buoi_15/`.

Nhiệm vụ: Tạo bộ tìm kiếm an toàn trong file `buoi_15/src/secure_retriever.py`.

YÊU CẦU:
1. Hàm tìm kiếm phải nhận vào hai tham số bắt buộc: `query` (câu hỏi) và `user_roles` (danh sách vai trò của người dùng hiện tại, ví dụ: `["Guest"]` hoặc `["HR", "Staff"]`).
2. Cấu hình kết nối cơ sở dữ liệu Neo4j cho phần tìm kiếm đồ thị được đọc từ file `.env` cục bộ.
3. Tích hợp kiểm tra quyền truy cập vào cả 3 phương thức tìm kiếm:
   - **BM25 Search**: Lọc trực tiếp trên Pandas DataFrame (dữ liệu `buoi_15/data/processed/chunks_secure.csv`) để chỉ giữ lại các dòng mà có ít nhất một vai trò trong `user_roles` trùng với danh sách `allowed_roles` của dòng đó trước khi tính toán điểm BM25 (hoặc lọc kết quả sau khi tính điểm).
   - **Dense Search**: Thực hiện hậu lọc (Post-Filtering) hoặc tiền lọc metadata (nếu vector DB hỗ trợ) dựa trên thuộc tính `allowed_roles` để loại bỏ các kết quả không hợp lệ.
   - **Graph Retrieval (Neo4j)**: Truy vấn Cypher phải tích hợp mệnh đề kiểm tra quyền:
     `WHERE any(role IN node.allowed_roles WHERE role IN $user_roles)`
4. Đảm bảo luồng Hybrid Fusion (RRF) và Reranker chỉ làm việc trên các ứng viên đã vượt qua bước lọc quyền bảo mật. Tuyệt đối không để xảy ra trường hợp tài liệu bị cấm lọt vào danh sách đưa sang Reranker.
5. Trả về cấu trúc kết quả chuẩn hóa tương tự buổi 14 nhưng bổ sung trường `allowed_roles` của tài liệu để đối sánh.
```

---

## 3.5. PROMPT 4 — Tích hợp Phân quyền vào Streamlit Web App

Học viên sẽ yêu cầu Agent nâng cấp giao diện Web để có thể trải nghiệm trực quan cơ chế RBAC.

```text
Tiếp tục bài thực hành Buổi 15. Hãy nâng cấp giao diện Streamlit Web App của chúng ta để minh họa trực quan tính năng kiểm soát truy cập dữ liệu.

Nhiệm vụ: Tạo hoặc cập nhật file `buoi_15/app_secure.py`.

YÊU CẦU GIAO DIỆN & TÍNH NĂNG:
1. **Thanh điều hướng bên trái (Sidebar)**:
   - Giữ nguyên các cấu hình của Buổi 14 (lựa chọn Method, thông số k, candidate_k).
   - Bổ sung một mục cấu hình mới: **"Vai trò của bạn (Your Roles)"** dạng Multiselect cho phép học viên tự chọn một hoặc nhiều vai trò (trong danh sách các vai trò đã thiết lập ở Prompt Setup).
2. **Khu vực tìm kiếm chính**:
   - Khi người dùng nhập câu hỏi và bấm tìm kiếm, ứng dụng sẽ gọi `secure_retriever.py` (với các cấu hình DB đọc từ `.env` cục bộ) và truyền danh sách vai trò đang được chọn ở Sidebar.
3. **Hiển thị kết quả**:
   - Hiển thị danh sách kết quả tìm kiếm kèm theo nhãn bảo mật rõ ràng (Ví dụ: `Quyền xem: [Admin, HR]`).
   - Nếu có tài liệu bị ẩn đi do không đủ quyền, hiển thị một thông báo thống kê nhỏ ở góc màn hình: `"Đã lọc bỏ X kết quả do không đủ quyền truy cập"`.
4. **Bảo mật Citation**:
   - Đảm bảo phần hiển thị gợi ý cấu trúc đồ thị (Graph Hints) hoặc liên kết trích dẫn cũng được lọc theo quyền của vai trò hiện tại.

Hãy viết code cho `app_secure.py`, cung cấp hướng dẫn khởi chạy ứng dụng Streamlit và hướng dẫn cách học viên kiểm thử tính năng này trên giao diện web.
```

---

## 3.6. PROMPT 5 — Tự động hóa kiểm thử rò rỉ bảo mật (Security Integration Test)

Không chỉ kiểm tra bằng mắt, học viên cần chạy các bộ test tự động để xác nhận tính an toàn của RAG.

```text
Tiếp tục bài thực hành Buổi 15. Hãy viết một script kiểm định bảo mật tự động để đảm bảo hệ thống không bị rò rỉ dữ liệu (data leakage).

Nhiệm vụ: Tạo file `buoi_15/scripts/security_audit.py`.

YÊU CẦU:
1. Thiết kế ít nhất 5 test case tự động. Mỗi test case gồm:
   - `query`: Câu hỏi chứa các từ khóa nhạy cảm (Ví dụ: "Bảng lương cấp quản lý", "Quy trình thu hồi nợ xấu").
   - `target_sensitive_document_id`: Mã văn bản nhạy cảm tương ứng (ví dụ văn bản HR hoặc Risk Management).
   - `unauthorized_roles`: Các vai trò không có quyền xem tài liệu này (Ví dụ: `["Guest"]`, `["Staff"]`).
   - `authorized_roles`: Các vai trò có quyền xem (Ví dụ: `["HR"]`, `["Admin"]`).
2. Script sẽ tự động chạy truy vấn tìm kiếm với hai cấu hình:
   - Chạy với `unauthorized_roles`: Assert (khẳng định) kết quả trả về Top-K **không được phép** chứa bất kỳ chunk nào thuộc về `target_sensitive_document_id`.
   - Chạy với `authorized_roles`: Xác nhận tài liệu đích có thể xuất hiện trong danh sách kết quả (nếu điểm tương đồng đủ cao).
   - Đọc kết nối database từ `.env` cục bộ trong thư mục hiện hành.
3. Xuất báo cáo tự động ra file `buoi_15/outputs/security_audit_report.md` theo cấu trúc:
   - Tổng quan số lượng bài test chạy.
   - Kết quả từng test case (PASS/FAIL).
   - Bằng chứng kiểm thử (nếu PASS, ghi nhận không tìm thấy tài liệu cấm; nếu FAIL, báo động rò rỉ dữ liệu).
   - Kết luận: Hệ thống đạt chứng nhận an toàn dữ liệu mức cơ bản hay chưa.

Hãy viết code, chạy thử script kiểm định này và báo cáo nội dung file `security_audit_report.md` sau khi chạy thành công.
```

---

## 4. Câu hỏi thảo luận và Đánh giá năng lực của học viên

Sau khi hoàn thành bài thực hành, học viên cần tự trả lời các câu hỏi sau để củng cố kiến thức:

1. **Câu hỏi về Kế thừa bảo mật**:
   * *“Nếu một node `DieuKhoan` không chứa trường `allowed_roles` nhưng node `VanBan` chứa nó lại quy định `allowed_roles = ['Admin', 'HR']`, chúng ta nên xử lý logic lọc quyền trong câu lệnh Cypher như thế nào để đảm bảo an toàn tối đa?”*
2. **Câu hỏi về rò rỉ thông tin qua Embeddings (Embedding Leakage)**:
   * *“Nếu ta không thực hiện lọc quyền truy cập trước khi tính toán độ tương đồng Vector (Dense Search) mà lại gửi toàn bộ câu hỏi lên một Vector Database chung không phân quyền, liệu hacker có thể lợi dụng phản hồi lỗi hoặc khoảng cách tương đồng để đoán biết sự tồn tại của tài liệu nhạy cảm không? Giải pháp khắc phục là gì?”*
3. **Câu hỏi về Reranker**:
   * *“Tại sao bắt buộc phải lọc quyền truy cập trước khi đưa các đoạn văn bản vào Cross-Encoder Reranker, thay vì lọc sau khi Reranker đã xếp hạng xong?”*
   *(Gợi ý: Tránh việc Reranker tính điểm và đưa tài liệu cấm lên vị trí số 1, sau đó bộ lọc xóa đi khiến Top-k bị thiếu hụt hoặc Reranker lãng phí tài nguyên xử lý dữ liệu mà người dùng không được xem).*
Displaying buoi15.md.