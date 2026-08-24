## Workspace
Chỉ được phép đọc:
- RAG/rag_foundation/buoi_05/output/chunks/
- RAG/rag_foundation/buoi_05/.venv/
- RAG/rag_foundation/buoi_06/
Không đọc:
- source code của Buổi 5
- README các buổi trước
- notebook
- git history
- các thư mục khác
Buổi 5 là black box. Không reverse engineering. Không phân tích cách Buổi 5 hoạt
động.
## Python
Sử dụng đúng interpreter trong: `RAG/rag_foundation/buoi_05/.venv/`
Không tạo virtual environment mới.
## Package
Chỉ cài:
- streamlit
- google-genai
- chromadb
- psycopg
- python-dotenv
Không cài framework khác.
## Coding Style
Ưu tiên: ít file, ít class, ít function, code dễ đọc.
Không tạo: repository pattern, service layer, dependency injection, factory,
plugin.
## Scope
Chỉ cần: index, retrieval, answer, streamlit.
Không phát triển ngoài yêu cầu.
## Error Handling
Chỉ cần try/except tối thiểu.
Không cần: retry, logging, monitoring.
## Security
Không in: API Key, password, secret.
## Code Size
Mục tiêu khoảng 300–500 dòng Python.
Nếu vượt khoảng 700 dòng, hãy đơn giản hóa thiết kế.