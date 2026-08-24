# Bài thực hành 1: Phân tách dữ liệu (Chunking), Tạo Vector nhúng (Embeddings) và Nạp dữ liệu vào Cơ sở dữ liệu đồ thị Neo4j

## Mục tiêu
Học sinh nắm được cách làm sạch các văn bản pháp luật dưới dạng HTML, phân tách chúng thành cấu trúc phân cấp Cha-Con (hierarchical parent-child), tạo các vector nhúng dày đặc (dense embeddings) bằng mô hình tiếng Việt chuyên dụng và nạp toàn bộ dữ liệu đồ thị văn bản và phân đoạn vào Neo4j.
Tải neo4j ở: neo4j desktop 2.0
---
## Ràng buộc
Chỉ làm việc ở trong folder graph_rag_labs/ nếu chưa có thì tạo ra nó.

## Các bước thực hiện
 
### **Bước 1: Phân tích HTML, Làm sạch và Phân tách cấu trúc phân cấp (Chunking)**
- Làm sạch nội dung HTML từ các tệp dữ liệu nhưng vẫn giữ nguyên cấu trúc văn bản (các tiêu đề, đoạn văn, bảng biểu).
- Phân tách văn bản thành các phân đoạn (chunks) có cấu trúc cha-con rõ ràng (Ví dụ: Chương ➔ Mục ➔ Điều ➔ Các đoạn văn/Bảng biểu chi tiết).
- Loại bỏ các trường HTML cồng kềnh khỏi các nút, liên kết trực tiếp các phân đoạn con tới nút gốc `Document` (sử dụng tiêu đề tương ứng).
- Nối các phân đoạn anh em liền kề bằng quan hệ `NEXT` để giữ nguyên luồng đọc của văn bản.
- **Yêu cầu**: Phải in ra màn hình console kết quả phân tách mẫu để minh họa trực quan cách thuật toán làm sạch và chia nhỏ HTML hoạt động.

### **Bước 2: Tạo Vector Nhúng (Embedding)**
- Thực hiện nhúng (embed) các đoạn văn bản bằng mô hình hỗ trợ tiếng Việt sau từ HuggingFace:
  `thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5`
- **Lưu ý**: Để phù hợp với điều kiện máy của học sinh không có GPU, chỉ cài đặt và sử dụng phiên bản PyTorch chạy trên CPU (`pytorch-cpu`).

### **Bước 3: Cấu hình kết nối Cơ sở dữ liệu**
- Tìm cổng kết nối Neo4j cục bộ (mặc định là `7687` đối với giao thức Bolt, và `7474` đối với HTTP).
- Thiết lập tài khoản và mật khẩu kết nối tới thực thể cơ sở dữ liệu Neo4j cục bộ.
(Cần người thực hiện tạo trước instance qua neo4j desktop)
### **Bước 4: Nạp dữ liệu vào Neo4j**
- Nạp siêu dữ liệu (metadata), các phân đoạn văn bản và vector nhúng tương ứng vào một cơ sở dữ liệu Neo4j cục bộ có tên là `kb-hops`.
- Thiết lập cấu trúc các nhãn nút (Nodes) và quan hệ (Relationships):
  - `(:Document)`: Lưu trữ siêu dữ liệu của văn bản luật.
  - `(:Chunk)`: Lưu trữ nội dung văn bản sạch và vector nhúng.
  - `[:PART_OF]`: Kết nối phân đoạn văn bản trở lại tài liệu gốc.
  - `[:PARENT_OF]`: Thể hiện cấu trúc phân cấp từ tiêu đề lớn xuống đoạn văn nhỏ.
  - `[:NEXT]`: Liên kết trình tự đọc.
  - Các quan hệ cấp tài liệu (ví dụ: `[:CAN_CU]`, `[:THAY_THE]`, `[:HOP_NHAT]`).

### **Bước 5: Kiểm tra và Xác minh**
- Kết nối vào công cụ Neo4j Browser trên máy cục bộ và kiểm tra số lượng các thực thể đã được nạp:
  - Số lượng nút Document: 15
  - Số lượng quan hệ giữa các tài liệu Document: 8
  - Đảm bảo các phân đoạn, phân cấp và liên kết tuần tự được tạo chính xác.

Video: https://youtu.be/IEwkpJIZP58