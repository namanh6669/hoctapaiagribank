Đầu vào là PDF tiếng Việt trong `datademo/`; đầu ra gồm text OCR chuẩn Unicode NFC, metadata `source`, `page`, `ocr_used`, `language`, và báo cáo của ba chiến lược chunking. Xác định rõ ba cách cần so sánh:

- **Fixed-size:** cắt theo số ký tự/token với overlap.
- **Semantic:** ưu tiên ranh giới đoạn văn thường ngắt như ngắt đoạn, kết đoạn, cách dòng.
- **Hierarchical:** chia theo cấu trúc mà mỗi Chương → Mục → Điều/Khoản → Điểm sẽ thành mốc bắt đầu của 1 chunk
Nêu việc cần sử dụng key trong .env thuộc folder src nhưng không được phép đọc giá trị của các key.
Không tạo embedding, không lưu vector database và không gọi LLM trong Buổi 5, code ở mức demo đơn giản không phức tạp hóa, không bỏ sót yêu cầu.