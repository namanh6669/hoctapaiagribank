# BÀI THỰC HÀNH — BUỔI 09
## Multi-query Retrieval và Parent–Child Retrieval cho văn bản pháp luật ngân hàng

## 1. Mục tiêu

Buổi 08 đã hoàn thiện Advanced RAG theo pipeline:

```text
Một câu hỏi → BM25 + Semantic → RRF → rerank chunk → Gemini
```

Buổi 09 giải quyết hai hạn chế còn lại:

1. Một cách diễn đạt duy nhất có thể không bao phủ hết các ý trong câu hỏi phức tạp.
2. Chunk nhỏ phù hợp để tìm kiếm nhưng đôi khi thiếu Điều, Khoản hoặc nội dung xung
   quanh cần thiết để trả lời đầy đủ.

Pipeline mới:

```text
                               ┌→ Q0: câu hỏi gốc ───────────────┐
Question → Multi-query ────────┼→ Q1: cách diễn đạt khác ───────┤
                               ├→ Q2: trọng tâm pháp lý khác ───┤
                               └→ Q3: thuật ngữ tra cứu khác ───┘
                                              ↓
                              Hybrid retrieval cho từng query
                                              ↓
                              Cross-query RRF trên child hits
                                              ↓
                     Child → Parent mapping và parent aggregation
                                              ↓
                         Rerank parent bằng câu hỏi gốc
                                              ↓
                         Parent evidence → Gemini answer
```

Mục tiêu bắt buộc:

- Sinh nhiều truy vấn tìm kiếm có kiểm soát từ một câu hỏi gốc.
- Luôn giữ câu hỏi gốc là `Q0`; truy vấn sinh thêm không được thay thế nó.
- Dùng Hybrid Search của Buổi 08 cho từng truy vấn nhưng chưa rerank từng nhánh.
- Hợp nhất kết quả của nhiều truy vấn bằng RRF dựa trên rank, không cộng raw score.
- Dựng hierarchy registry từ `chapter/article/clause/point` và nội dung văn bản.
- Tìm ở child nhỏ nhưng trả về parent context lớn hơn.
- Rerank parent bằng cross-encoder và câu hỏi gốc.
- Hiển thị query fan-out, child hit, parent expansion và rank movement.
- So sánh bốn chế độ trên cùng câu hỏi và cùng corpus.
- Test offline không gọi Gemini, không tải model và không sửa storage Buổi 05–08.

## Điểm khác biệt nhìn thấy rõ so với Buổi 08

| Buổi 08 | Buổi 09 |
|---|---|
| Một câu hỏi tạo một luồng retrieval | Một câu hỏi tạo Q0 và nhiều query variant |
| Retrieval một lần | Retrieval độc lập cho từng query |
| RRF hợp nhất BM25 với semantic | Thêm một tầng RRF hợp nhất kết quả giữa các query |
| Candidate là chunk phẳng | Candidate có quan hệ child → parent |
| Rerank chunk nhỏ | Rerank parent context bằng câu hỏi gốc |
| Evidence chỉ là chunk được tìm thấy | Evidence gồm parent và các anchor child dẫn tới parent |
| Không thấy mức đóng góp của từng query | Có query fan-out và ma trận query–child |
| Context có thể thiếu nội dung Điều/Khoản xung quanh | Parent expansion bổ sung ngữ cảnh phân cấp |
| So sánh BM25/Semantic/Hybrid/Rerank | So sánh Single/Multi-query và Flat/Parent |

Buổi 09 không xây lại BM25, semantic, RRF lexical–semantic hoặc cross-encoder.
Những thành phần đó phải được kế thừa có kiểm soát từ Buổi 08.

---

# 2. Cách dùng tài liệu

Thực hiện đúng thứ tự:

```text
Prompt 01 → Audit baseline Buổi 08 và dữ liệu phân cấp
Prompt 02 → Tạo project và specification Buổi 09
Prompt 03 → Dựng hierarchy registry và parent store
Prompt 04 → Xây Multi-query Generator
Prompt 05 → Retrieval từng query và Cross-query Fusion
Prompt 06 → Parent–Child Retrieval và Parent Aggregation
Prompt 07 → Parent Reranking và Answer Pipeline
Prompt 08 → Streamlit Multi-query & Hierarchy Explorer
Prompt 09 → Test, Evaluation, README và nghiệm thu
```

Quy tắc sử dụng:

1. Mở thư mục gốc `RAG` làm workspace.
2. Dán đúng một prompt mỗi lần.
3. Chờ Agent chạy kiểm tra và báo kết quả thực tế.
4. Không chuyển bước nếu còn FAIL hoặc BLOCKED.
5. Không cho Agent làm trước prompt tiếp theo.
6. Không đánh dấu PASS khi command chưa thực sự được chạy.

---

# 3. Quy tắc chung

## Workspace

Được đọc:

```text
rag_foundation/buoi_05/output/chunks/
rag_foundation/buoi_05/.venv/
rag_advanced/buoi_08/
rag_advanced/buoi_09/
```

Chỉ được ghi:

```text
rag_advanced/buoi_09/
```

Không sửa:

- Code, chunks và PDF của Buổi 05.
- Code, `.env`, tests, reports và storage của Buổi 06–08.
- Virtual environment, ngoại trừ cài dependency có trong requirements Buổi 09.

## Python

Windows:

```text
rag_foundation/buoi_05/.venv/Scripts/python.exe
```

Linux/macOS:

```text
rag_foundation/buoi_05/.venv/bin/python
```

Trong prompt, `<PYTHON>` là interpreter tương ứng ở trên. Không gõ nguyên chuỗi
`<PYTHON>` vào terminal.

## Nguyên tắc triển khai

- Python standard library và các dependency Buổi 08 được ưu tiên tái sử dụng.
- Không thêm LangChain, LlamaIndex hoặc framework RAG khác.
- Không dùng LLM để tự viết lại hoặc tóm tắt nội dung parent trong lúc index.
- Không tạo hierarchy giả nếu không đủ bằng chứng từ metadata hoặc text.
- Mọi fallback hierarchy phải có `resolution_method` và warning.
- Query variant chỉ dùng để retrieval, không được xem là bằng chứng hoặc câu trả lời.
- Câu hỏi gốc là nguồn duy nhất để rerank parent và sinh answer.
- Không gọi generation nếu evidence cuối không đạt gate.
- Không dùng fake embedding, fake query generator hoặc fake reranker trong runtime.
- Fake deterministic chỉ được dùng trong unit test.

## Bốn chế độ bắt buộc

| Mode | Query | Evidence unit | Rerank |
|---|---|---|---|
| `single_flat` | Chỉ Q0 | Child chunk | Rerank child như baseline Buổi 08 |
| `multi_flat` | Q0 + query variants | Child chunk sau cross-query fusion | Rerank child bằng Q0 |
| `single_parent` | Chỉ Q0 | Parent mở rộng từ child hit | Rerank parent bằng Q0 |
| `multi_parent` | Q0 + query variants | Parent mở rộng từ fused child hits | Rerank parent bằng Q0 |

`multi_parent` là mode mặc định của Buổi 09. Bốn mode phải dùng cùng strategy,
candidate limits, model identity và corpus version khi so sánh.

---

# PROMPT 01 — AUDIT BASELINE BUỔI 08 VÀ DỮ LIỆU PHÂN CẤP

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là senior RAG engineer thực hiện preflight read-only cho Buổi 09.

[CURRENT STEP]

Đây là Bước 01. Chỉ đọc, audit và chạy kiểm tra an toàn. Không tạo thư mục Buổi
09, không sửa file, không cài package, không gọi Gemini, không tải reranker và
không tạo/reset collection.

[WORKSPACE]

Mở thư mục gốc RAG. Được đọc:

- rag_foundation/buoi_05/output/chunks/
- rag_foundation/buoi_05/.venv/
- rag_advanced/buoi_08/

[AUDIT BUỔI 08]

1. Liệt kê source, tests, README, SPEC và requirements Buổi 08.
2. Đọc `advanced_rag.py`, `rag.py`, `evaluate.py`, `app.py`.
3. Xác nhận có các primitive có thể tái sử dụng:
   - load/validate hierarchical chunks
   - Vietnamese legal tokenizer và BM25
   - semantic retrieval
   - BM25 + semantic RRF
   - cross-encoder reranker có dependency injection
   - answer generation và citations
4. Ghi chính xác public function/class và CLI đang có; không đoán tên.
5. Kiểm tra `.env.example` và chỉ báo tên biến, không in API key thật.
6. Chạy compile và toàn bộ unittest Buổi 08, không gọi service thật.
7. Chạy status read-only nếu command thực sự không tạo resource.

[AUDIT HIERARCHICAL DATA]

Chỉ đọc các file `*__hierarchical.json` của Buổi 05.

Báo:

- số file, số record và số source
- required fields: chunk_id, strategy, source, page_start, page_end, text
- số record theo tổ hợp structure keys
- số record có chapter/article/clause/point
- số record không có structure
- số heading Chương/Điều có thể nhận diện trong text
- độ dài text min/median/p95/max
- chunk_id có thứ tự số ổn định theo source hay không
- metadata article có được lặp ở mọi child hay không
- ví dụ metadata thiếu nhưng text vẫn chứa heading
- ví dụ văn bản sửa đổi có Điều được trích dẫn bên trong nội dung

Không kết luận rằng mọi child đã có sẵn parent_id. Không hard-code số liệu nếu kết
quả đọc thực tế khác.

[RISK REPORT]

Nêu rõ:

1. Rủi ro dùng riêng `structure.article` để group parent.
2. Rủi ro regex nhầm Điều được trích dẫn thành heading cấp cao.
3. Rủi ro parent quá dài.
4. Rủi ro một query variant làm mất số Điều/Khoản của câu hỏi gốc.
5. Rủi ro số lần retrieval, latency và API call tăng theo query count.

[COMMANDS]

Dùng đúng interpreter Buổi 05. Chỉ chạy lệnh read-only/compile/test. Không dùng
command có `--reset`, không index và không download.

[OUTPUT]

Trả báo cáo:

## Baseline Buổi 08
## Hierarchy statistics
## Reusable interfaces
## Risks
## Commands và kết quả thật
## Kết luận PASS/FAIL/BLOCKED

PASS khi baseline compile/test được và hierarchical chunks đọc được. Dừng sau báo
cáo; không làm Bước 02.
```

## Kiểm tra sau Prompt 01

- Có số liệu thật về 3 file hierarchical và toàn bộ record tương ứng.
- Agent nhận ra metadata hierarchy không đồng đều.
- Không có file hoặc storage nào bị thay đổi.

---

# PROMPT 02 — TẠO PROJECT VÀ SPECIFICATION BUỔI 09

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là software architect tạo skeleton độc lập cho Buổi 09.

[CURRENT STEP]

Đây là Bước 02. Chỉ tạo cấu trúc project, sao chép baseline cần thiết và viết
specification. Chưa triển khai hierarchy builder, multi-query, retrieval hoặc UI.

[WRITE SCOPE]

Chỉ ghi trong `rag_advanced/buoi_09/`. Không sửa Buổi 05–08.

[PROJECT STRUCTURE]

Tạo:

rag_advanced/buoi_09/
├── .env.example
├── .gitignore
├── requirements.txt
├── rag.py
├── advanced_rag.py
├── hierarchical_rag.py
├── evaluate.py
├── app.py
├── README.md
├── SPEC_buoi_09.md
├── eval/
│   └── questions.json
├── reports/
│   └── .gitkeep
├── storage/
│   ├── chroma/
│   │   └── .gitkeep
│   ├── hierarchy/
│   │   └── .gitkeep
│   └── huggingface/
│       └── .gitkeep
└── tests/
    ├── __init__.py
    └── fixtures/
        └── hierarchical_sample.json

[BASELINE COPY]

1. Sao chép `rag_advanced/buoi_08/rag.py` và `rag_advanced/buoi_08/advanced_rag.py` vào Buổi 09.
2. Thêm docstring nói rõ đây là baseline được snapshot từ Buổi 08.
3. Không import runtime từ directory Buổi 08.
4. Không sao chép `.env`, storage, cache, report hoặc `__pycache__`.
5. Không thay đổi logic baseline ở bước này.
6. Ghi SHA-256 của hai source gốc và hai bản copy vào báo cáo để chứng minh snapshot.

[REQUIREMENTS]

Giữ dependency trực tiếp của Buổi 08. Không thêm framework RAG. Nếu code Buổi 08
đang import dependency trực tiếp nào nhưng requirements thiếu thì báo rõ và chỉ
bổ sung dependency thực sự cần thiết.

[ENV EXAMPLE]

Tạo `.env.example` gồm cấu hình Buổi 08 và thêm:

MULTI_QUERY_COUNT=3
MULTI_QUERY_MAX_CHARS=300
MULTI_QUERY_TEMPERATURE=0.2
MULTI_QUERY_ORIGINAL_WEIGHT=1.5
MULTI_QUERY_VARIANT_WEIGHT=1.0
MULTI_QUERY_RRF_K=60
PER_QUERY_CANDIDATES=12
PARENT_MAX_CHARS=6000
PARENT_SCORE_CHILD_LIMIT=3
PARENT_RRF_K=60
PARENT_CANDIDATES=10
FINAL_PARENT_TOP_K=3
TOTAL_CONTEXT_MAX_CHARS=16000

Giữ:

GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=768
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANK_MIN_SCORE=0.50
RERANK_DEVICE=auto

Không tạo key giả và không sao chép key thật.

[SPECIFICATION]

Viết `SPEC_buoi_09.md` bằng tiếng Việt, gồm:

1. Mục tiêu và khác biệt Buổi 08/09.
2. Sơ đồ Q0 + variants → per-query hybrid → cross-query RRF → child-to-parent →
   parent aggregation → parent rerank → generation.
3. Bốn mode: single_flat, multi_flat, single_parent, multi_parent.
4. QueryVariant schema và validation.
5. Hierarchy registry schema.
6. ParentDocument schema.
7. MultiQueryChildHit và ParentCandidate schema.
8. Quy tắc hierarchy resolution và ambiguous warning.
9. Công thức cross-query RRF và parent aggregation.
10. Context budget và citation contract.
11. Status/failure contract.
12. Testability/dependency injection.
13. Evaluation metrics và acceptance criteria.
14. Xác nhận chỉ ghi Buổi 09.

[PLACEHOLDER FILES]

`hierarchical_rag.py`, `evaluate.py`, `app.py` chỉ có docstring/TODO an toàn, import
được và chưa có side effect. Không tạo code giả tuyên bố tính năng đã chạy.

[VALIDATION]

- In cây file.
- Compile toàn bộ `.py` Buổi 09.
- Import không gọi API, không tải model, không tạo collection và không build store.
- Chứng minh storage Buổi 08 không đổi bằng read-only listing/timestamp phù hợp.

[OUTPUT]

Báo files tạo, hash baseline, compile result và giới hạn chưa triển khai. Dừng,
không làm Bước 03.
```

## Kiểm tra sau Prompt 02

- Buổi 09 độc lập với runtime Buổi 08.
- Có specification mô tả đúng hai tầng fusion và parent expansion.
- Import chưa tạo file trong storage.

---

# PROMPT 03 — DỰNG HIERARCHY REGISTRY VÀ PARENT STORE

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là document structure engineer xây parent–child registry deterministic cho
văn bản pháp luật ngân hàng.

[CURRENT STEP]

Đây là Bước 03. Chỉ triển khai config, hierarchy builder, parent store, CLI và
tests liên quan. Chưa gọi Gemini, chưa retrieval và chưa rerank.

[IMPLEMENTATION FILE]

Triển khai trong `hierarchical_rag.py`. Có thể gọi loader/validator từ snapshot
Buổi 09 nhưng không sửa dữ liệu Buổi 05.

[CONFIG VALIDATION]

Validate:

- MULTI_QUERY_COUNT: integer từ 1 đến 5
- MULTI_QUERY_MAX_CHARS: 50 đến 1000
- temperature: 0 đến 1
- weights: float không âm, không đồng thời bằng 0
- RRF K: integer dương
- candidate counts: integer dương, tối đa 100
- PARENT_MAX_CHARS: 1000 đến 20000
- PARENT_SCORE_CHILD_LIMIT: 1 đến 20
- FINAL_PARENT_TOP_K <= PARENT_CANDIDATES
- TOTAL_CONTEXT_MAX_CHARS >= PARENT_MAX_CHARS
- model names không rỗng

Load `.env` theo `Path(__file__).resolve()`, không phụ thuộc cwd.

[INPUT ORDER]

1. Chỉ load strategy `hierarchical`.
2. Group theo `source`.
3. Sắp xếp child theo phần sequence số cuối của `chunk_id`; không sort lexical nếu
   lexical làm `...:10` đứng trước `...:2`.
4. Duplicate chunk_id phải fail.
5. Page range, text và structure invalid phải báo file/record cụ thể.

[HIERARCHY RESOLUTION]

Mỗi child phải có:

- `child_id` bằng chunk_id gốc
- source/page/text gốc không sửa
- `chapter_label`, `article_label`, `clause_label`, `point_label` nếu xác định được
- `structural_path`
- `resolution_method`: `metadata`, `heading_inferred`, `carried_forward`, hoặc
  `document_fallback`
- `ambiguous`: boolean
- `warnings`: list

Độ ưu tiên:

1. Metadata structure hợp lệ của chính record.
2. Heading cấp cao rõ ràng ở đầu chunk.
3. Carry forward chapter/article gần nhất trong cùng source.
4. Document fallback khi không xác định được article.

Không carry qua source khác. Không coi mọi cụm `Điều N` xuất hiện giữa một câu là
heading. Nếu metadata xung đột với heading hoặc một chunk chứa nhiều ứng viên
article không phân giải chắc chắn, giữ quy tắc deterministic, đặt `ambiguous=true`
và ghi warning; không tự chọn im lặng.

[PARENT BUILDING]

Parent là article block; nếu chưa xác định article thì dùng document fallback
block. Trong một article quá dài, chia thành window liên tiếp theo ranh giới child
để không vượt `PARENT_MAX_CHARS` khi có thể.

Quy tắc:

- Không cắt giữa child chỉ để đạt giới hạn.
- Một child thuộc đúng một parent window.
- Parent text được ghép từ text gốc theo thứ tự; không dùng LLM tóm tắt.
- Không lặp child text.
- Parent page_start là min và page_end là max của children.
- Parent structural header lấy từ registry, không bịa.
- Parent ID stable, ví dụ hash từ source + resolved article key + window index.
- Cùng input/config phải tạo byte-equivalent logical registry và ID giống nhau.
- Nếu một child đơn lẻ dài hơn PARENT_MAX_CHARS, giữ nguyên, đánh warning
  `oversized_single_child`, không truncate pháp lý âm thầm.

[SCHEMAS]

Hierarchy child record:

{
  "child_id": "...",
  "parent_id": "...",
  "source": "...pdf",
  "page_start": 1,
  "page_end": 2,
  "text": "...",
  "structural_path": {
    "chapter": "... hoặc null",
    "article": "... hoặc null",
    "clause": "... hoặc null",
    "point": "... hoặc null"
  },
  "resolution_method": "...",
  "ambiguous": false,
  "warnings": []
}

Parent document:

{
  "parent_id": "...",
  "source": "...pdf",
  "page_start": 1,
  "page_end": 3,
  "article_key": "...",
  "window_index": 1,
  "child_ids": ["..."],
  "text": "...",
  "char_count": 1234,
  "ambiguous_child_count": 0,
  "warnings": []
}

[STORE]

Build chủ động vào:

- `storage/hierarchy/children.json`
- `storage/hierarchy/parents.json`
- `storage/hierarchy/manifest.json`

Manifest gồm schema_version, input file fingerprints, strategy, config identity,
counts, warning counts và build timestamp. Ghi atomically qua temporary file cùng
directory rồi replace. Không xóa store hợp lệ trước khi build mới thành công.

Status phải read-only: không mkdir, không build và không sửa timestamp.

[CLI]

Thêm:

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py hierarchy-audit`

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py build-hierarchy`

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py hierarchy-status`

[TEST]

Dùng fixture nhỏ, temporary directory:

1. Metadata precedence.
2. Heading inferred ở đầu chunk.
3. Carry forward trong cùng source.
4. Không carry qua source.
5. Inline `Điều N` không bị nhận nhầm.
6. Conflict đặt ambiguous/warning.
7. Numeric chunk ordering.
8. Stable parent ID.
9. Parent split tại child boundary.
10. Oversized child warning.
11. Mỗi child đúng một parent.
12. Parent pages/count/text đúng.
13. Atomic build và manifest fingerprint.
14. Status không tạo/sửa file.

[OUTPUT]

Báo hierarchy statistics, warning examples, parent size distribution, commands và
test result. Không che ambiguous data. Dừng, không làm Bước 04.
```

## Kiểm tra sau Prompt 03

- Mọi child được ánh xạ đúng một parent.
- Parent được ghép từ text gốc, không tóm tắt bằng LLM.
- Các trường hợp hierarchy không chắc chắn được hiển thị thành warning.

---

# PROMPT 04 — MULTI-QUERY GENERATOR

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là retrieval engineer xây query expansion có kiểm soát cho tiếng Việt pháp lý.

[CURRENT STEP]

Đây là Bước 04. Chỉ triển khai Multi-query Generator, schema, cache trong process,
CLI và unit tests. Chưa retrieval, parent expansion, rerank hoặc answer generation.

[QUERY SET CONTRACT]

Output luôn gồm:

- Q0: nguyên văn câu hỏi người dùng sau trim/NFC, `origin=original`
- Q1..Qn: tối đa MULTI_QUERY_COUNT query sinh thêm, `origin=generated`

Schema:

{
  "original_question": "...",
  "queries": [
    {
      "query_id": "Q0",
      "text": "...",
      "origin": "original",
      "focus": "original_intent"
    },
    {
      "query_id": "Q1",
      "text": "...",
      "origin": "generated",
      "focus": "exact_legal_terms | paraphrase | missing_aspect"
    }
  ],
  "model": "...",
  "generation_latency_ms": 0.0,
  "status": "ready"
}

[GENERATION]

1. Q0 được code tạo trực tiếp từ câu hỏi gốc; không yêu cầu Gemini viết lại Q0.
2. Dùng Gemini generation model trong config Buổi 09 để sinh riêng Q1..Qn.
3. Một Generation API call duy nhất sinh toàn bộ variants.
4. JSON do model trả chỉ có các generated variants theo schema tối thiểu:

   `{"queries": [{"text": "...", "focus": "..."}]}`

   Sau validation, code mới ghép Q0 vào đầu Query Set hoàn chỉnh.
5. Yêu cầu structured JSON output bằng schema mà phiên bản `google-genai` đang cài
   hỗ trợ. Đọc API/library thật trước khi viết, không đoán tham số.
6. Temperature lấy từ config, mặc định 0.2.
7. Prompt tiếng Việt yêu cầu tạo các cách tra cứu đa dạng, không trả lời câu hỏi.
8. Query variants nên bao phủ:
   - thuật ngữ pháp lý chính xác
   - cách diễn đạt tương đương
   - một khía cạnh còn thiếu nếu câu hỏi có nhiều ý
9. Không thêm thông tin sự kiện, kết luận pháp lý hoặc nguồn ngoài câu hỏi.
10. Nếu câu hỏi chứa `Điều`, `Khoản`, `Điểm`, số hiệu văn bản hoặc năm, ít nhất một
   variant phải giữ nguyên reference đó.
11. Không phát minh số Điều/Khoản không có trong câu hỏi.

[VALIDATION]

- question không rỗng và không vượt giới hạn hợp lý
- JSON đúng schema
- số generated query từ 1 đến MULTI_QUERY_COUNT
- mỗi query sau trim/NFC không rỗng, không quá MULTI_QUERY_MAX_CHARS
- deduplicate bằng Unicode NFC + casefold + chuẩn hóa whitespace/punctuation
- Q0 không bị thay đổi nội dung có nghĩa
- query_id được gán lại deterministic sau validation
- model trả duplicate thì loại duplicate và báo `dropped_duplicate_count`
- model trả ít query hợp lệ thì dùng số còn lại, không tạo query giả
- không bao giờ đưa answer/citation do model sinh vào retrieval metadata

Nếu API/JSON/schema lỗi, trả status `query_generation_unavailable` với lỗi rõ. Không
âm thầm gọi mode multi như thể đã có variants. Các mode single vẫn được phép chạy
khi người dùng chủ động chọn.

[INJECTION AND CACHE]

- Hàm nhận optional `query_generator_fn` để unit test.
- Fake generator chỉ trong test.
- Cache trong process theo hash của original question + generation config + model.
- Không ghi prompt/query chứa dữ liệu người dùng xuống disk mặc định.
- Cache hit phải ghi `cache_hit=true` và không gọi API lần hai.

[CLI]

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py expand-query --question "Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?"`

Command này có thể gọi Gemini khi người dùng chủ động chạy.

[TEST]

1. Q0 luôn đứng đầu và giữ nội dung.
2. Strict schema validation.
3. NFC/trim/max length.
4. Duplicate removal.
5. Legal reference preservation check.
6. Không chấp nhận số Điều bịa thêm trong fixture rule-based validation nếu có thể
   xác định chắc chắn.
7. Deterministic IDs.
8. Một generator call.
9. Cache hit không gọi lại.
10. API lỗi trả explicit status.
11. Unit test không gọi mạng.

[OUTPUT]

Báo schema, prompt contract, cache behavior, command và tests. Không in API key.
Dừng, không làm Bước 05.
```

## Kiểm tra sau Prompt 04

- Q0 luôn là câu hỏi gốc.
- Query variants chỉ phục vụ tìm kiếm, không chứa câu trả lời.
- Một lần expansion chỉ gọi Gemini tối đa một lần.

---

# PROMPT 05 — RETRIEVAL TỪNG QUERY VÀ CROSS-QUERY FUSION

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là search engineer triển khai fan-out retrieval và hợp nhất kết quả nhiều query.

[CURRENT STEP]

Đây là Bước 05. Chỉ triển khai per-query hybrid retrieval, cross-query RRF, trace,
CLI và tests. Chưa parent expansion, rerank hoặc generation.

[PER-QUERY RETRIEVAL]

Với mỗi Q0..Qn:

1. Gọi Hybrid Search của snapshot Buổi 09: BM25 + semantic → inner RRF.
2. Không gọi cross-encoder trong từng query.
3. Lấy tối đa PER_QUERY_CANDIDATES child hits.
4. Cùng strategy `hierarchical`, corpus, collection identity và config.
5. Mỗi query chỉ được gọi mỗi retriever đúng một lần.
6. Giữ trace BM25 rank, semantic rank, inner RRF rank nhưng không dùng raw score
   của chúng để fusion giữa các query.

[CROSS-QUERY RRF]

Với child d:

multi_query_rrf_score(d) =
  tổng trên các query q tìm thấy d của:
  query_weight(q) / (MULTI_QUERY_RRF_K + rank_q(d))

Trong đó:

- Q0 dùng MULTI_QUERY_ORIGINAL_WEIGHT
- generated query dùng MULTI_QUERY_VARIANT_WEIGHT
- rank_q là inner fused rank của child trong query q

Không cộng BM25 score, cosine distance, inner RRF score hoặc rerank score vào công
thức này.

[MERGE CONTRACT]

1. Union theo child_id/chunk_id, không duplicate.
2. Metadata cùng child phải khớp; mismatch fail.
3. Giữ candidate chỉ xuất hiện ở một query.
4. `support_query_count` đếm số query unique tìm thấy child.
5. `support_query_ids` theo thứ tự Q0, Q1...
6. `per_query_ranks` lưu rank của từng query.
7. `best_query_rank` là rank nhỏ nhất.
8. Sort:
   - multi_query_rrf_score giảm
   - support_query_count giảm
   - best_query_rank tăng
   - child_id
9. Gán `multi_query_rank` từ 1.

Schema child hit hợp nhất:

{
  "child_id": "...",
  "text": "...",
  "source": "...",
  "page_start": 1,
  "page_end": 2,
  "multi_query_rrf_score": 0.05,
  "multi_query_rank": 1,
  "support_query_count": 3,
  "support_query_ids": ["Q0", "Q1", "Q3"],
  "per_query_ranks": {"Q0": 2, "Q1": 1, "Q3": 4},
  "per_query_trace": {...}
}

[FAILURE CONTRACT]

- Nếu Q0 retrieval lỗi: toàn pipeline fail.
- Nếu generated query retrieval lỗi: ghi lỗi theo query và status `partial`, không
  giả vờ query đó trả zero result.
- Nếu mọi generated query lỗi nhưng Q0 thành công: trả trace Q0 nhưng mode multi có
  status `multi_query_partial`; UI phải hiển thị warning.
- Không âm thầm đổi nhãn mode thành thành công đầy đủ.

[TRACE]

Trả:

- query count requested/valid/executed/failed
- generated query latency và retrieval latency từng query
- result count từng query
- union child count
- overlap distribution: hit bởi 1, 2, 3... query
- fusion latency
- Gemini expansion call count
- semantic embedding call count nếu đo được

[CLI]

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py multi-child --question "Điều kiện vay vốn và các trường hợp không được cho vay là gì?"`

In query list và bảng child_id, per-query ranks, support count, MQ-RRF score.

[TEST]

1. Công thức MQ-RRF tính tay.
2. Original/variant weights.
3. Deduplicate union.
4. Missing query contribution.
5. Support query count/IDs.
6. Metadata mismatch fail.
7. Deterministic tie-break.
8. Mỗi query gọi hybrid đúng một lần.
9. Không gọi reranker/generation.
10. Q0 failure và generated-query partial status.
11. Trace counts/latency schema.
12. Tests dùng fake hybrid retriever và fake query generator, không mạng/storage thật.

[OUTPUT]

Báo hai tầng RRF khác nhau, schema, command và test result. Dừng, không làm Bước 06.
```

## Kiểm tra sau Prompt 05

- Mỗi query có danh sách retrieval độc lập.
- Có tầng RRF thứ hai giữa các query.
- Bảng kết quả cho biết child được query nào hỗ trợ.

---

# PROMPT 06 — PARENT–CHILD RETRIEVAL VÀ PARENT AGGREGATION

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là retrieval architect triển khai “retrieve child, return parent”.

[CURRENT STEP]

Đây là Bước 06. Chỉ map fused child hits sang parent, aggregate parent rank, áp
context budget, trace và tests. Chưa cross-encoder rerank parent và chưa generation.

[PRECONDITION]

Hierarchy store phải tồn tại và manifest phải khớp fingerprint/config hiện tại.
Store thiếu hoặc stale trả `hierarchy_not_ready`; không tự build trong query.

[CHILD TO PARENT]

1. Với mỗi child hit, lookup đúng một parent_id từ children registry.
2. Load parent document từ parent store.
3. Không tìm kiếm vector trực tiếp trên parent ở Buổi 09.
4. Không ghép parent từ dữ liệu retrieval bị thiếu; parent store là source of truth.
5. Child hoặc parent lookup thiếu phải fail rõ với ID.

[PARENT AGGREGATION]

Group fused child hits theo parent_id. Mỗi parent chỉ lấy tối đa
PARENT_SCORE_CHILD_LIMIT child tốt nhất theo multi_query_rank để tính điểm:

parent_rrf_score(p) =
  tổng 1 / (PARENT_RRF_K + multi_query_rank(child))

Không cộng raw MQ-RRF score hoặc rerank score vào công thức.

Giữ toàn bộ supporting child IDs để explainability, nhưng tách rõ:

- `scoring_child_ids`: được dùng tính parent score
- `supporting_child_ids`: toàn bộ child hits map vào parent
- `anchor_child_id`: child có multi_query_rank tốt nhất

Sort parent:

1. parent_rrf_score giảm
2. số supporting queries unique giảm
3. best_child_rank tăng
4. parent_id

Chỉ giữ `PARENT_CANDIDATES` trước tầng rerank.

[PARENT CANDIDATE SCHEMA]

{
  "parent_id": "...",
  "source": "...pdf",
  "page_start": 1,
  "page_end": 3,
  "structural_path": {...},
  "text": "parent context...",
  "parent_rrf_score": 0.03,
  "parent_rank": 1,
  "anchor_child_id": "...",
  "scoring_child_ids": ["..."],
  "supporting_child_ids": ["..."],
  "support_query_ids": ["Q0", "Q1"],
  "best_child_rank": 1,
  "ambiguous": false,
  "warnings": []
}

[CONTEXT BUDGET]

- Parent builder đã chia theo PARENT_MAX_CHARS.
- Chọn parent theo rank nhưng tổng context không vượt TOTAL_CONTEXT_MAX_CHARS.
- Chỉ thêm nguyên parent; không cắt giữa parent hoặc child pháp lý.
- Nếu parent đầu tiên vượt budget do oversized child, giữ parent đầu tiên và trả
  warning rõ thay vì trả context rỗng.
- Duplicate parent không được tính hai lần.
- Duplicate child text giữa các parent là lỗi hierarchy invariant.

[TRACE]

Trả:

- input child hit count
- unique parent count
- số child mỗi parent
- child-to-parent mapping table
- parent score components
- parents dropped bởi candidate limit/context budget
- child chars so với expanded parent chars
- context expansion factor
- ambiguous/warning counts
- mapping/aggregation latency

[MODES]

- `single_parent`: query set chỉ Q0 rồi child → parent.
- `multi_parent`: Q0 + variants rồi child → parent.
- Không gọi reranker trong bước này.

[CLI]

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py parent-retrieve --mode multi_parent --question "Điều kiện vay vốn và các trường hợp không được cho vay là gì?"`

In mapping tree:

Parent
└── supporting child
    └── query IDs và ranks

[TEST]

1. Child map đúng parent.
2. Missing/stale hierarchy status.
3. Parent aggregation formula tính tay.
4. Child score cap.
5. Supporting và scoring child tách đúng.
6. Parent deduplicate.
7. Sort/tie-break deterministic.
8. Candidate limit.
9. Context budget chỉ cắt ở parent boundary.
10. Oversized first parent warning.
11. Expansion factor/count trace.
12. Không gọi reranker/generation.

[OUTPUT]

Báo mapping, formula, budget, command và tests. Dừng, không làm Bước 07.
```

## Kiểm tra sau Prompt 06

- Retrieval vẫn tìm child nhỏ.
- Evidence được mở rộng thành parent có ngữ cảnh lớn hơn.
- Có cây giải thích parent nào đến từ child và query nào.

---

# PROMPT 07 — PARENT RERANKING VÀ ANSWER PIPELINE

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là senior RAG engineer hoàn thiện Multi-query Parent–Child answer pipeline.

[CURRENT STEP]

Đây là Bước 07. Tái sử dụng cross-encoder và generation contract của snapshot
Buổi 09 để rerank parent, gate evidence và sinh câu trả lời. Chưa làm Streamlit.

[PARENT RERANK]

1. Cross-encoder input pair là `(original_question, parent_text)`.
2. Không rerank bằng generated query.
3. Lazy-load/cache model đúng contract Buổi 08.
4. Không load model khi import/status/hierarchy build/single retrieval tests.
5. Chỉ rerank tối đa PARENT_CANDIDATES.
6. Giữ `parent_rrf_score`, `parent_rank` và thêm:
   - `parent_rerank_raw_score`
   - `parent_rerank_score = sigmoid(logit)`
   - `parent_rerank_rank`
   - `parent_rank_change = parent_rank - parent_rerank_rank`
7. Sort score giảm, parent_rank tăng, parent_id.
8. Chỉ lấy FINAL_PARENT_TOP_K sau rerank và context budget.
9. Rerank score là điểm chuẩn hóa, không gọi là xác suất đúng.

[MODE CONTRACT]

`single_flat`:

- Q0 → hybrid retrieval → rerank child.
- Tương đương baseline Buổi 08 trên snapshot/config Buổi 09.

`multi_flat`:

- Q0 + variants → per-query hybrid → MQ-RRF → rerank child bằng Q0.

`single_parent`:

- Q0 → hybrid → child-to-parent → parent aggregation → rerank parent bằng Q0.

`multi_parent`:

- Q0 + variants → per-query hybrid → MQ-RRF → child-to-parent → parent
  aggregation → rerank parent bằng Q0.

Không chạy reranker nhiều lần ngoài nhu cầu của mode. Compare retrieval không gọi
answer generation.

[EVIDENCE GATE]

- Flat modes dùng gate baseline Buổi 08.
- Parent modes chỉ nhận parent có `parent_rerank_score >= RERANK_MIN_SCORE`.
- Hierarchy ambiguous không tự động bị loại, nhưng evidence/citation phải mang
  warning để người dùng thấy.
- Không đủ accepted evidence trả `insufficient_evidence` và không gọi Gemini.
- Reranker lỗi trả `reranker_unavailable`; không silent fallback.
- Multi-query generation lỗi trả `query_generation_unavailable` cho multi mode.

[GENERATION]

Tối đa hai Gemini **Generation API calls** trong một query `multi_parent` hoàn chỉnh:

1. Một call sinh query variants.
2. Một call sinh answer nếu evidence đạt gate.

Single mode chỉ cần answer generation call. Các lần gọi Gemini Embedding để embed
Q0/Q1..Qn phải được đếm riêng và không nằm trong giới hạn hai Generation calls.
Query variants không được đưa vào answer prompt như sự thật. Answer prompt chỉ gồm
câu hỏi gốc và accepted evidence.

Quy tắc answer:

- Chỉ trả lời từ evidence.
- Không suy diễn tư vấn pháp lý.
- Mỗi nhận định có citation `[P1]`, `[P2]`.
- Không tự tạo nguồn, trang, Điều/Khoản, parent_id hoặc child_id.
- Nếu evidence mâu thuẫn/ambiguous phải nói rõ giới hạn.

[CITATION]

Mỗi citation object gồm:

- evidence_id: P1, P2...
- parent_id
- anchor_child_id
- supporting_child_ids
- source
- page_start/page_end
- structural_path
- parent_rerank_score
- ambiguous/warnings

Label `[P1]` chỉ được tạo từ evidence thật đã accepted. Citation validation fail
thì không trình bày answer như thành công.

[RESULT AND TRACE]

Result gồm:

- status/mode/original_question
- query_set
- child_hits
- parent_candidates
- accepted_evidence
- answer/citations
- stage latencies
- API call counts
- model/config/corpus/hierarchy identities
- partial/warning/error list

[CLI]

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py query --mode multi_parent --question "Điều kiện vay vốn và các nhu cầu vốn không được cho vay được quy định thế nào?"`

`<PYTHON> rag_advanced/buoi_09/hierarchical_rag.py compare --question "Điều kiện vay vốn và các nhu cầu vốn không được cho vay được quy định thế nào?"`

Compare chạy bốn mode retrieval/rerank nhưng không gọi answer generation.

[TEST]

1. Reranker pair dùng Q0 + parent text.
2. Generated query không dùng để rerank/generation.
3. Sort/rank change/final K.
4. Gate accepted/rejected.
5. Không evidence thì không generation.
6. Flat/parent mode routing.
7. Multi-query failure status.
8. Reranker failure không fallback.
9. Citation dùng parent và anchor child thật.
10. Citation label validation.
11. Multi mode tối đa hai generation API calls.
12. Compare không answer generation.
13. Trace identity/counts.
14. Offline tests dùng injected fakes, không mạng/model/storage thật.

[OUTPUT]

Báo mode matrix, gate, citation schema, API call budget, commands và tests. Dừng,
không làm Bước 08.
```

## Kiểm tra sau Prompt 07

- Reranker chấm parent bằng câu hỏi gốc.
- Answer nhận parent context nhưng citation vẫn truy ngược được anchor child.
- Multi mode không gọi Gemini quá hai lần cho một lượt hỏi đáp.

---

# PROMPT 08 — STREAMLIT MULTI-QUERY & HIERARCHY EXPLORER

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là Streamlit developer tạo giao diện Buổi 09 khác biệt rõ với Buổi 08.

[CURRENT STEP]

Đây là Bước 08. Chỉ hoàn thiện `app.py` và UI helper tests. Không thay đổi retrieval
formula hoặc hierarchy contract đã nghiệm thu.

[PAGE]

Title:

`RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval`

Subtitle hiển thị pipeline:

`Query fan-out → Hybrid per query → Cross-query RRF → Parent expansion → Parent rerank`

[SIDEBAR]

Hiển thị/chọn:

- mode: single_flat, multi_flat, single_parent, multi_parent
- MULTI_QUERY_COUNT
- PER_QUERY_CANDIDATES
- PARENT_CANDIDATES
- FINAL_PARENT_TOP_K
- RERANK_MIN_SCORE
- strategy cố định hierarchical
- Gemini key có/không, không lộ key
- embedding/generation/reranker model
- hierarchy store ready/stale/missing
- child count, parent count, ambiguous count
- collection status

Các widget chỉ thay config runtime hợp lệ. Không tự build/index/download khi render.

[TABS]

## Tab 1 — Ask Advanced RAG

- Text area câu hỏi.
- Mode mặc định `multi_parent`.
- Nút chạy rõ ràng.
- Answer, citations và warning.
- Tổng latency, Generation call count và Embedding call count tách riêng.
- Không tự query khi rerun widget.

## Tab 2 — Query Fan-out

Hiển thị card Q0..Qn:

- query text
- original/generated
- focus
- validation status
- result count
- retrieval latency

Q0 có màu/nhãn riêng. Có ma trận hàng là child, cột là Q0..Qn, ô là rank hoặc `—`.
Hiển thị support query count và MQ-RRF score.

## Tab 3 — Parent–Child Explorer

Hiển thị tree/expander:

Parent P
├── article/chapter path
├── source/pages
├── parent rank → rerank rank
├── parent score/rerank score
└── supporting children
    ├── child ID
    ├── query IDs/ranks
    └── anchor snippet

Cho xem parent text nhưng mặc định thu gọn. Ambiguous/warning phải nổi bật.

## Tab 4 — Mode Comparison

Chạy cùng một câu hỏi qua bốn mode, retrieval-only:

- single_flat
- multi_flat
- single_parent
- multi_parent

Bảng gồm:

- final evidence IDs
- unit type: child/parent
- relevant rank fields
- source/pages
- unique sources/articles
- retrieved child count
- expanded parent count
- context chars
- expansion factor
- latency
- Generation call count và Embedding call count tách riêng
- status/warnings

Không tuyên bố mode thắng nếu không có gold labels.

## Tab 5 — Evaluation

- Đọc latest report nếu tồn tại.
- Không tự chạy evaluator khi render.
- Hiển thị Child Recall@K, Parent Recall@K, MRR@K, nDCG@K, latency và context chars.
- Warning khi gold labels `needs_human_review=true`.

[DISTINCT VISUAL REQUIREMENT]

Giao diện Buổi 09 bắt buộc nhìn thấy:

1. Nhiều query từ một câu hỏi.
2. Ma trận query–child.
3. Cây parent–child.
4. Parent rank trước/sau rerank.
5. Context expansion factor.

Nếu UI chỉ giống Buổi 08 và đổi tiêu đề thì chưa đạt.

[STATE AND CACHE]

- Dùng `st.session_state` giữ result của lần bấm gần nhất.
- Cache resource phù hợp cho pipeline/model nhưng không cache API key trong output.
- Query expansion cache chỉ trong process/session theo contract.
- Nút build hierarchy và prepare semantic là action riêng có xác nhận; không chạy
  trong page load.
- Model download chỉ xảy ra sau action người dùng thực sự cần rerank.

[ERROR UX]

Phân biệt:

- hierarchy_not_ready
- collection_not_ready
- query_generation_unavailable
- multi_query_partial
- reranker_unavailable
- insufficient_evidence
- generation_error

Hiển thị hướng xử lý, không dump stack trace/key.

[RUN]

`<PYTHON> -m streamlit run rag_advanced/buoi_09/app.py`

[TEST]

Unit test helper thuần Python:

- mode comparison row
- query-child matrix
- parent tree data
- citation formatting
- warning/status mapping
- không cần browser và không gọi API/model

[OUTPUT]

Báo UI sections, run command, tests và những action có thể tải model/gọi API. Dừng,
không làm Bước 09.
```

## Kiểm tra sau Prompt 08

- Giao diện cho thấy query fan-out và ma trận query–child.
- Có cây child → parent và rank movement của parent.
- UI khác Buổi 08 ngay cả trước khi đọc answer.

---

# PROMPT 09 — TEST, EVALUATION, README VÀ NGHIỆM THU

Dán nguyên prompt sau vào AI Agent:

```text
[ROLE]

Bạn là senior reviewer nghiệm thu kỹ thuật Buổi 09.

[CURRENT STEP]

Đây là Bước 09 và là bước cuối. Hoàn thiện test, evaluator, README và chạy
acceptance. Không thêm tính năng ngoài Multi-query và Parent–Child Retrieval.

[OFFLINE TEST CONTRACT]

Toàn bộ unittest:

- không Internet
- không Gemini thật
- không tải Hugging Face model
- fake deterministic chỉ qua dependency injection
- temporary directory/storage
- không đọc `.env` thật
- không sửa storage Buổi 05–08

[REQUIRED TEST GROUPS]

## Hierarchy

1. Metadata/heading/carry-forward/fallback precedence.
2. Inline legal reference không bị coi là heading.
3. Conflict ambiguous warning.
4. Numeric child order và stable parent ID.
5. Parent window, pages, text và one-parent-per-child invariant.
6. Manifest fingerprint/stale/status/atomic build.

## Multi-query

7. Q0 preservation.
8. Schema, limit, NFC, deduplicate, legal reference.
9. Một generator call và cache.
10. Explicit generation failure.

## Cross-query fusion

11. Formula/weights tính tay.
12. Union/support/per-query ranks/tie-break.
13. Metadata mismatch.
14. Partial query failure và trace.

## Parent retrieval

15. Child lookup và parent aggregation.
16. Score child cap.
17. Context budget/oversized warning.
18. Expansion factor và explainability mapping.

## Rerank/answer

19. Rerank Q0 + parent.
20. Bốn mode routing.
21. Gate/status/no silent fallback.
22. Citation parent + anchor child.
23. API call budget và compare không generation.

## Isolation/UI

24. Import/status không side effect.
25. Không load model/build store khi page load.
26. UI helper schemas.

[EVALUATION DATA]

Mở rộng `eval/questions.json`. Mỗi item:

{
  "question_id": "Q01",
  "question": "...",
  "question_type": "exact | paraphrase | multi_aspect | hierarchy_context | out_of_scope",
  "relevant_child_ids": ["..."],
  "relevant_parent_ids": ["..."],
  "needs_human_review": true,
  "notes": "..."
}

Không tự coi nhãn do AI suy ra là ground truth đã duyệt. Parent IDs phải được resolve
từ hierarchy store hiện tại; stale IDs phải fail.

[EVALUATOR]

So sánh bốn mode trên cùng question set, corpus identity và K:

- Child Recall@K khi mode trả child hoặc có supporting children
- Parent Recall@K
- MRR@K
- nDCG@K binary relevance
- unique relevant parents/sources retrieved
- query count và child union count
- context chars và expansion factor
- mean/p50 latency
- query-generation call count và embedding call count tách riêng

Evaluation retrieval-only, không gọi answer generation. Query generator và semantic
retrieval có thể gọi service trong real evaluation do người dùng chủ động chạy;
offline metric tests phải dùng fixture/fakes.

Không khẳng định `multi_parent` thắng nếu nhãn cần human review hoặc metrics không
hỗ trợ kết luận đó.

[REPORT]

Lưu JSON atomically trong `reports/`, gồm:

- timestamp
- config/model/corpus/hierarchy identity
- per-question results
- aggregate metrics per mode
- failures/partial statuses
- human-review warning

Tạo `latest_report.json` chỉ sau khi report hoàn chỉnh hợp lệ.

[README]

README tiếng Việt gồm:

1. Mục tiêu và khác biệt Buổi 08/09.
2. Sơ đồ pipeline hai tầng fusion và parent expansion.
3. Bốn mode comparison.
4. Cấu trúc project và setup `.env`.
5. Build hierarchy và giải thích warning/ambiguous.
6. Query expansion contract và API call budget.
7. Công thức inner RRF, cross-query RRF và parent aggregation.
8. Child retrieval/parent return/rerank parent.
9. Lệnh status, build, expand, retrieve, query, compare, evaluate, Streamlit.
10. Giải thích candidate K, parent K và context budget.
11. Evaluation metrics và giới hạn gold labels.
12. Troubleshooting hierarchy stale, model/API, latency, context lớn.
13. Tuyên bố không phải tư vấn pháp lý.

[MANUAL COMPARISON QUESTIONS]

A. Exact reference:

`Điều 8 quy định những nhu cầu vốn nào không được cho vay?`

B. Paraphrase:

`Khách hàng cần đáp ứng những yêu cầu gì để được tổ chức tín dụng xem xét cho vay?`

C. Multi-aspect:

`Điều kiện vay vốn và những nhu cầu vốn không được cho vay được quy định như thế nào?`

D. Hierarchy context:

`Quy định về cơ cấu lại thời hạn trả nợ gồm điều kiện, thời gian và trách nhiệm nào?`

E. Out-of-scope:

`Lãi suất tiết kiệm cao nhất trên thị trường hôm nay là bao nhiêu?`

[ACCEPTANCE COMMANDS]

1. Compile toàn bộ Python Buổi 09.
2. Chạy toàn bộ unittest offline.
3. Chạy hierarchy audit/build/status trên data thật.
4. Kiểm tra invariant: child count input = child count registry; mỗi child một parent.
5. Chạy fixture multi-query/fusion/parent retrieval không mạng.
6. Nếu API key/semantic index sẵn sàng và được phép:
   - chạy expand-query thật một lần
   - chạy multi-child một câu
   - chạy compare retrieval-only
7. Nếu reranker thật sẵn sàng:
   - chạy một multi_parent query
   - ghi device, latency và parent rank movement
8. Bước service/model không sẵn sàng phải ghi NOT RUN, không fake runtime.
9. Xác nhận hash/timestamp hợp lý cho storage Buổi 05–08 không đổi.

[OUTPUT]

Trả báo cáo:

## Files tạo/sửa
## Test commands và PASS/FAIL
## Hierarchy build statistics/warnings
## Bảng bốn mode
## Evaluation metrics hoặc NOT RUN
## API/model calls PASS/FAIL/NOT RUN
## Giới hạn và tài nguyên
## Xác nhận không sửa Buổi 05–08

Không nói PASS nếu chưa chạy. Sau báo cáo thì dừng; không tạo Prompt 10.
```

## Kiểm tra sau Prompt 09

- Offline tests PASS và không gọi service thật.
- Hierarchy store khớp dữ liệu, mỗi child đúng một parent.
- Có bảng so sánh bốn mode và không che giấu NOT RUN.
- README giải thích rõ Buổi 09 khác Buổi 08.

---

# 4. Lệnh chạy tham khảo

Chạy từ thư mục gốc `RAG`.

## Windows PowerShell

### Audit và build hierarchy

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py hierarchy-audit
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py build-hierarchy
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py hierarchy-status
```

### Chuẩn bị semantic index riêng Buổi 09

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\advanced_rag.py prepare-semantic --strategy hierarchical
```

### Xem query variants

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py expand-query --question "Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?"
```

### Multi-query child retrieval

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py multi-child --question "Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?"
```

### Parent retrieval

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py parent-retrieve --mode multi_parent --question "Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?"
```

### Hỏi đáp hoàn chỉnh

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py query --mode multi_parent --question "Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?"
```

### So sánh bốn mode

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py compare --question "Điều kiện vay vốn và nhu cầu vốn không được cho vay là gì?"
```

### Unit test

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m unittest discover -s .\rag_advanced\buoi_09\tests -v
```

### Evaluation

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\evaluate.py --k 5
```

### Streamlit

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m streamlit run .\rag_advanced\buoi_09\app.py
```

---

# 5. Kịch bản so sánh trực tiếp với Buổi 08

## Câu nói mở đầu

> Buổi 08 cải thiện cách xếp hạng một câu hỏi trên các chunk phẳng. Buổi 09 xử lý
> hai vấn đề khác: một câu hỏi có thể cần nhiều cách tra cứu, và chunk tìm kiếm tốt
> có thể quá ngắn để làm ngữ cảnh trả lời. Vì vậy hệ thống tìm bằng nhiều query ở
> child nhỏ, sau đó mở rộng về parent và rerank parent bằng câu hỏi gốc.

## Demo 1 — Multi-aspect question

```text
Điều kiện vay vốn và những nhu cầu vốn không được cho vay được quy định thế nào?
```

Quan sát:

- Q0 được giữ nguyên không?
- Query variants có tách các khía cạnh “điều kiện” và “không được cho vay” không?
- Child nào chỉ được một query tìm thấy?
- Multi-query có tăng phạm vi source/article không?

## Demo 2 — Parent context

```text
Quy định về cơ cấu lại thời hạn trả nợ gồm điều kiện, thời gian và trách nhiệm nào?
```

Quan sát:

- Child anchor chứa phần nào của quy định?
- Parent expansion bổ sung những Khoản/Điểm nào xung quanh?
- Parent có bị ambiguous hoặc quá dài không?
- Parent reranker đẩy evidence nào lên hoặc xuống?

## Demo 3 — So sánh bốn mode

Chạy cùng một câu hỏi ở:

```text
single_flat → multi_flat → single_parent → multi_parent
```

Không chỉ so answer. So sánh child recall, parent recall, context size, latency và
số API call.

## Demo 4 — Out-of-scope

```text
Lãi suất tiết kiệm cao nhất trên thị trường hôm nay là bao nhiêu?
```

Multi-query không được biến câu hỏi ngoài corpus thành một câu có vẻ trả lời được.
Gate cuối vẫn phải chặn generation khi không đủ evidence.

## Câu kết

> Multi-query giúp tăng khả năng tìm thấy tài liệu khi câu hỏi có nhiều cách diễn
> đạt hoặc nhiều khía cạnh. Parent–Child Retrieval giải quyết sự đánh đổi giữa
> chunk nhỏ để tìm chính xác và context lớn để trả lời đầy đủ. Đổi lại, hệ thống
> tốn thêm retrieval, API call, latency và cần quản lý hierarchy chặt chẽ. Vì vậy
> lợi ích phải được chứng minh bằng trace và metric thực tế.

---

# 6. Checklist cuối

- [ ] Chạy đúng Prompt 01–09; không có Prompt 10.
- [ ] Không sửa Buổi 05–08.
- [ ] Q0 luôn là câu hỏi gốc.
- [ ] Query variants không chứa answer hoặc citation giả.
- [ ] Legal references quan trọng được bảo toàn.
- [ ] Mỗi query chạy hybrid retrieval độc lập.
- [ ] Cross-query RRF không cộng raw score khác thang đo.
- [ ] Mỗi child thuộc đúng một parent.
- [ ] Hierarchy ambiguous có warning.
- [ ] Parent được ghép từ text gốc, không LLM summary.
- [ ] Parent window không cắt giữa child.
- [ ] Parent aggregation có supporting/scoring child rõ ràng.
- [ ] Parent reranker dùng Q0, không dùng generated query.
- [ ] Citation truy được parent → anchor child → source/page.
- [ ] `multi_parent` tối đa hai Gemini Generation calls cho một lượt hỏi đáp;
      embedding calls được đếm riêng.
- [ ] Compare không gọi answer generation.
- [ ] UI có query fan-out, query–child matrix và parent tree.
- [ ] Có context expansion factor và latency từng stage.
- [ ] Test offline không gọi API/model hub.
- [ ] Evaluation so sánh cùng corpus/query/K.
- [ ] Không tuyên bố mode thắng khi gold labels chưa duyệt.
- [ ] Không coi rerank score là xác suất đúng.
- [ ] Không coi kết quả là tư vấn pháp lý.

## Tài liệu kỹ thuật tham khảo

- Gemini structured output:
  https://ai.google.dev/gemini-api/docs/structured-output
- RAG-Fusion — multi-query generation và Reciprocal Rank Fusion:
  https://github.com/Raudaschl/rag-fusion
- Query expansion survey:
  https://arxiv.org/abs/1708.00247
- BGE multilingual reranker:
  https://huggingface.co/BAAI/bge-reranker-v2-m3
