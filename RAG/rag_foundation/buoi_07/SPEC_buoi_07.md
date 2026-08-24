# Agent Specification - Buổi 07

## Workspace

Agent được phép đọc:

- `rag_foundation/buoi_05/output/chunks/`
- `rag_foundation/buoi_05/.venv/`
- `rag_foundation/buoi_06/`
- `rag_foundation/buoi_07/`

Agent chỉ được ghi trong:

- `rag_foundation/buoi_07/`

Không sửa code, output, cấu hình hoặc dữ liệu của Buổi 05 và Buổi 06.

## Python

- Dùng lại `.venv` của Buổi 05: `rag_foundation/buoi_05/.venv/`.
- Không tạo virtual environment mới cho Buổi 07.
- Khi viết code dùng đường dẫn dựa trên `Path(__file__).resolve()`, không hard-code đường dẫn theo máy.

## Input

- Input chính là JSON trong `rag_foundation/buoi_05/output/chunks/`.
- Buổi 05 là nguồn dữ liệu đã chuẩn bị sẵn.
- Không OCR lại.
- Không parse PDF lại.
- Không chunk lại.

## Packages

Chỉ dùng các package trực tiếp được quy định trong `requirements.txt`:

- `streamlit>=1.61,<2`
- `google-genai>=2.16,<3`
- `chromadb>=1.5,<2`
- `python-dotenv>=1.2,<2`

Không thêm package trực tiếp khác nếu chưa có yêu cầu riêng.

## Pipeline

Các bước triển khai Buổi 07 theo thứ tự:

1. Validate input chunk theo data contract.
2. Tạo embedding thật bằng Gemini embedding model.
3. Lưu vector vào Chroma persistent.
4. Retrieval top-k theo câu hỏi.
5. Confidence gate bằng distance threshold.
6. Generation bằng Gemini generation model khi evidence đủ mạnh.
7. Citation từ metadata thật.
8. Streamlit UI để người mới thao tác.
9. Unittest offline với mock API và temporary storage.

## Data Contract

Mỗi chunk bắt buộc có các field:

- `chunk_id`
- `strategy`
- `source`
- `page_start`
- `page_end`
- `text`

Field `text` phải là chuỗi không rỗng sau khi strip. `page_start` và `page_end` phải biểu diễn trang hợp lệ, trong đó `page_start <= page_end`.

## Index Contract

- Một `strategy` dùng một Chroma collection riêng.
- Model và dimension của index/query phải khớp.
- Dùng embedding thật từ Gemini.
- Không dùng vector giả, vector random hoặc hash vector.
- Chặn embedding có `NaN`.
- Chặn embedding có `Infinity` hoặc `-Infinity`.
- Chặn boolean trong vector embedding.
- Chặn zero vector.
- Chroma dùng cosine distance.
- Chroma collection dùng `embedding_function=None`; code tự truyền embedding vào Chroma.
- Index phải idempotent: chạy lại không tạo duplicate chunk.
- Status là read-only, không tự reset hoặc upsert.
- Phải validate embedding xong trước khi reset/upsert collection.

## Retrieval Contract

- Retrieval phải trả evidence thật lấy từ Chroma và metadata đã lưu.
- Mỗi evidence phải có distance.
- Chỉ evidence đạt threshold mới được đưa vào generation.
- Nếu evidence yếu hoặc không có evidence đạt threshold thì không gọi generation.
- Khi không gọi generation, trả lời theo hướng chưa đủ dữ liệu và kèm warning phù hợp.

## Citation Contract

- Citation lấy từ metadata thật đã lưu khi index.
- Không tin `source`, `page`, `chunk_id` do LLM tự tạo.
- Kết quả trả về có `citations` và `warnings`.
- Code có trách nhiệm thay label hợp lệ trong câu trả lời bằng citation thật từ evidence.

## Security

- Không in secret ra terminal, UI, log hoặc exception message.
- Không tạo `.env` có key thật.
- `.env.example` chỉ chứa key rỗng và cấu hình mẫu.
- Nếu thiếu API key, code phải báo thiếu key theo cách an toàn.

## Testing

- Dùng `unittest` trong thư viện chuẩn.
- Mock Gemini API; không gọi Internet trong test.
- Dùng temporary storage cho Chroma/test data.
- Test không cần API key thật.
- Fixture nhỏ nằm trong `tests/fixtures/chunks_sample.json`.

## Coding Style

- Ít file.
- Ít class.
- Ít function.
- Không kiến trúc phức tạp.
- Code ưu tiên rõ ràng, dễ đọc cho người mới học RAG.
- Mỗi bước chỉ triển khai đúng phạm vi của bước đó.
