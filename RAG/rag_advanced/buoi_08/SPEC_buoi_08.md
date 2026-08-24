# SPEC Buổi 08 - Advanced RAG

> Trạng thái: specification khởi tạo cho workshop. Buổi 08 chưa triển khai BM25, semantic candidate orchestration, RRF, reranker, UI hoàn chỉnh hoặc evaluation chính thức.

## 1. Workspace và security

- Chỉ ghi trong `rag_advanced/buoi_08/`.
- Không sửa source, test, output, `.env`, storage hoặc cấu hình của Buổi 05, Buổi 06, Buổi 07.
- Không đọc hoặc in giá trị secret từ `.env`; chỉ được kiểm tra tên biến có/thiếu.
- `.env.example` chỉ chứa giá trị mẫu/rỗng, không chứa API key thật.
- Không tự động tải model, không gọi API ngoài, không tạo index trong bước khởi tạo project.
- Mọi dữ liệu fixture/eval starter là dữ liệu mô phỏng phục vụ học tập, không phải tư vấn pháp lý và chưa được chuyên gia pháp lý duyệt.

## 2. Quan hệ với Buổi 05 và Buổi 07

- Buổi 05 cung cấp chunk JSON đã chuẩn hóa tại `rag_foundation/buoi_05/output/chunks/`.
- Buổi 07 cung cấp semantic baseline gồm loader/validator, Gemini embedding, Chroma persistent storage, semantic retrieval, confidence gate, generation và citation mapping.
- Buổi 08 sao chép `rag_foundation/buoi_07/rag.py` thành `rag_advanced/buoi_08/rag.py` để làm baseline độc lập.
- Buổi 08 không import runtime trực tiếp từ Buổi 07.
- Bản sao `rag.py` phải dùng `Path(__file__).resolve()` để đọc `.env`, `storage/`, fixture và output cục bộ của Buổi 08; riêng input mặc định có thể trỏ về chunk output của Buổi 05 qua workspace root.
- Buổi 08 mở rộng baseline bằng pipeline hybrid: BM25 candidate + semantic candidate + RRF fusion + optional cross-encoder reranker + final evidence selection.

## 3. Data contract

Mỗi chunk đầu vào bắt buộc có các field chuẩn của Buổi 07:

- `chunk_id`: string không rỗng, unique trong tập được load.
- `strategy`: string thuộc `fixed-size`, `semantic`, `hierarchical`.
- `source`: string không rỗng, tên nguồn/tài liệu.
- `page_start`: integer >= 1.
- `page_end`: integer >= `page_start`.
- `text`: string không rỗng sau khi strip.

Quy định thêm cho Buổi 08:

- Không dựa vào metadata do LLM sinh ra.
- Pipeline có thể bổ sung field runtime như score/rank/trace nhưng không ghi đè metadata gốc.
- Candidate/evidence object phải giữ được `chunk_id` để map citation và evaluation.

## 4. BM25 tokenizer/retrieval contract

- Tokenizer BM25 phải deterministic và offline.
- Tokenizer không gọi LLM, API hoặc model ngoài.
- Tokenizer tiếng Việt tối thiểu cần:
  - lowercase;
  - chuẩn hóa khoảng trắng;
  - giữ số Điều/Khoản và số trang quan trọng;
  - tách token theo chữ/số Unicode;
  - có stopword list nhỏ, có test rõ ràng nếu dùng.
- BM25 index trong Buổi 08 phải tạo từ chunk text đã validate.
- BM25 retrieval trả danh sách candidate gồm:
  - `chunk_id`;
  - `rank_bm25`;
  - `score_bm25`;
  - `source`, `page_start`, `page_end`, `text`;
  - trace ngắn giải thích số token match/top terms nếu có.
- BM25 score chỉ dùng để xếp hạng trong nhánh BM25, không so sánh trực tiếp với distance semantic.

## 5. Semantic candidate contract

- Semantic candidate dùng baseline Buổi 07 trong `rag.py` hoặc hàm wrapper cục bộ của Buổi 08.
- Không gọi query/generation thật trong unit test; test phải mock embedder/collection/generator.
- Semantic retrieval phải trả candidate gồm:
  - `chunk_id`;
  - `rank_semantic`;
  - `distance`;
  - `source`, `page_start`, `page_end`, `text`;
  - `accepted_by_distance` theo threshold semantic.
- Distance thấp hơn thường liên quan hơn khi dùng cosine distance trong Chroma.
- Query model, dimension và collection metadata phải khớp với index metadata.

## 6. RRF fusion contract

- Reciprocal Rank Fusion dùng rank, không dùng raw BM25 score và semantic distance trực tiếp.
- Công thức mặc định:

```text
rrf_score = sum(weight_source / (rrf_k + rank_source))
```

- `rrf_k` mặc định đề xuất: `60`.
- Weight mặc định đề xuất:
  - BM25: `1.0`;
  - Semantic: `1.0`.
- Nếu một chunk xuất hiện ở nhiều nguồn candidate, merge theo `chunk_id` và cộng đóng góp RRF.
- Fusion result phải lưu trace:
  - rank/score BM25 nếu có;
  - rank/distance semantic nếu có;
  - từng contribution RRF;
  - `rrf_score` cuối cùng;
  - `fused_rank`.
- Tie-breaker phải deterministic, ví dụ `rrf_score desc`, best available source rank asc, `chunk_id asc`.

## 7. Cross-encoder reranker contract

- Reranker là bước tùy chọn sau RRF, không bắt buộc trong baseline khởi tạo.
- Reranker nhận query và top-N fused candidates.
- Reranker trả:
  - `rerank_score`;
  - `rerank_rank`;
  - model/provider/version nếu có;
  - trace cho biết candidate nào được rerank.
- Unit test không được tải model thật; phải dùng fake scorer deterministic.
- Nếu model local/chậm/không có, pipeline phải có chế độ fallback dùng RRF-only.
- Reranker không được tạo citation; citation chỉ map từ metadata thật của candidate.

## 8. Final evidence và citation contract

- Final evidence được chọn từ fused hoặc reranked candidates.
- Mỗi evidence có:
  - `evidence_id` dạng `E1`, `E2`, ...;
  - `chunk_id`;
  - `source`;
  - `page_start`, `page_end`;
  - `text`;
  - `distance` nếu đến từ semantic;
  - `score_bm25` nếu đến từ BM25;
  - `rrf_score`;
  - `rerank_score` nếu có;
  - `trace`.
- Confidence gate final phải rõ ràng: có thể dùng semantic distance, fused rank, rerank threshold hoặc rule kết hợp nhưng phải được log trong trace.
- Citation chỉ được map từ metadata thật trong final evidence.
- Nếu câu trả lời sinh label không có trong final evidence, label đó bị loại và sinh warning.
- Không tự tạo nguồn, trang, Điều/Khoản hoặc chunk id từ output của LLM.

## 9. Pipeline trace contract

Mỗi lần query Advanced RAG phải trả `trace` đủ để debug:

- query id hoặc query text đã normalize;
- config snapshot không chứa secret;
- số candidate BM25 request/return;
- số candidate semantic request/return;
- danh sách fused candidates và contribution;
- reranker enabled/disabled/fallback reason;
- final evidence ids;
- warnings/errors an toàn, không lộ API key.

Trace dùng cho học tập và kiểm thử nên ưu tiên dễ đọc hơn tối ưu dung lượng.

## 10. Evaluation metrics contract

- `eval/questions.json` là starter set có gold labels ban đầu với `needs_human_review=true`.
- Không tuyên bố gold labels đã được chuyên gia pháp lý duyệt.
- Metrics tối thiểu cần hỗ trợ khi triển khai:
  - Recall@K theo `relevant_chunk_ids`;
  - MRR@K;
  - Hit@K;
  - số câu out-of-scope bị retrieve/generate sai;
  - coverage theo BM25-only, semantic-only, fused, reranked nếu có.
- Evaluation phải chạy offline với fake hoặc collection có sẵn; không tự động gọi Gemini/generation thật trong unit test.
- Report lưu trong `reports/`, không ghi đè không hỏi nếu report có nội dung thủ công quan trọng.

## 11. Offline testing contract

- Test dùng `unittest` hoặc thư viện chuẩn trừ khi có yêu cầu riêng.
- Test không gọi Internet, Gemini thật hoặc tải model.
- Test dùng fixture nhỏ trong `tests/fixtures/chunks_advanced_sample.json`.
- Test dùng temporary storage khi cần Chroma.
- Các phần cần test tối thiểu khi triển khai:
  - tokenizer deterministic;
  - BM25 ranking cơ bản;
  - semantic candidate adapter với mock;
  - RRF merge/tie-break;
  - reranker fake scorer/fallback;
  - final evidence/citation mapping;
  - metrics Recall@K/MRR@K;
  - out-of-scope behavior.

## 12. UI comparison contract

- UI Buổi 08 là comparison UI để người học thấy khác biệt giữa retrieval modes.
- UI tối thiểu khi hoàn chỉnh cần hiển thị:
  - BM25 candidates;
  - Semantic candidates;
  - RRF fused candidates;
  - reranked candidates nếu bật;
  - final evidence và citations;
  - pipeline trace.
- UI không hiển thị secret.
- UI phải có cảnh báo rõ rằng fixture/eval starter chưa được chuyên gia pháp lý duyệt.
- UI không tự index/query thật khi chỉ mở trang; mọi hành động gọi model/index phải do người dùng bấm rõ ràng.
