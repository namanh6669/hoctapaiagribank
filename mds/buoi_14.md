# BUỔI 14 — Nâng cấp RAG với Hybrid Search + Reranking và xây Knowledge Graph mini

## Mục tiêu

Trong bài thực hành này, người học sử dụng **AI Coding Agent / Vibe Coding** để nâng cấp hệ thống RAG cơ bản theo hai hướng:

1. **Nâng cấp Retrieval**
   - BM25 / lexical search.
   - Dense retrieval bằng embedding.
   - Hybrid Search.
   - Reranking.
   - So sánh chất lượng trước và sau khi nâng cấp.

2. **Xây Knowledge Graph mini cho bộ quy định nội bộ**
   - Tạo node văn bản và điều khoản/chunk.
   - Tạo các quan hệ có thật trong dữ liệu.
   - Nạp vào Neo4j.
   - Trực quan hóa và chạy một số truy vấn nhiều bước đơn giản.

Sản phẩm cuối cùng giúp người học quan sát được pipeline:

```text
Câu hỏi
   │
   ├──────────────► BM25 Search
   │
   └──────────────► Dense Search
                        │
                        ▼
                 Hybrid Fusion
                        │
                        ▼
                Candidate Top-N
                        │
                        ▼
                    Reranker
                        │
                        ▼
                     Top-k
                        │
                        ▼
              Context + Citation
                        │
                        ▼
                    RAG Answer
```

Song song với đó:

```text
metadata.csv
content.csv
relationships.csv
        │
        ▼
Knowledge Graph mini
        │
        ▼
Neo4j
        │
        ▼
VanBan ──CONTAINS──> DieuKhoan
                         │
                         ├── NEXT ──> DieuKhoan
                         │
                         └── các quan hệ thật có trong dữ liệu
```

> Trọng tâm của buổi học không phải tự viết toàn bộ code từ đầu. Người học cần biết cách **giao việc cho Agent, kiểm tra code, chạy thử, đọc kết quả, so sánh retrieval và phát hiện khi Agent làm sai nghiệp vụ**.

---


# 1. Kiến thức cần hiểu trước khi thực hành

## 2.1. Dense Retrieval là gì?

Dense Retrieval biến:

```text
Câu hỏi
Văn bản
```

thành vector embedding.

Sau đó so sánh độ gần nhau trong không gian vector.

Ví dụ:

```text
Câu hỏi:
"Ai được quyền phê duyệt khoản vay?"

Văn bản:
"Thẩm quyền quyết định cấp tín dụng thuộc..."
```

Hai câu không dùng hoàn toàn cùng từ nhưng có thể gần nhau về ngữ nghĩa.

### Điểm mạnh

- hiểu tương đồng ngữ nghĩa;
- xử lý cách diễn đạt khác nhau;
- tốt khi câu hỏi không chứa đúng từ khóa trong tài liệu.

### Điểm yếu

Có thể không ưu tiên tốt:

```text
Điều 12
Mẫu 03
QĐ-125
KTNB-01
```

hoặc các cụm từ rất đặc thù.

---

## 2.2. Lexical Search / BM25 là gì?

BM25 là phương pháp xếp hạng dựa trên từ khóa xuất hiện trong câu hỏi và tài liệu.

Ví dụ:

```text
Câu hỏi:
"Quy định QĐ-125 nói gì về phê duyệt?"

```

BM25 thường rất mạnh vì:

```text
QĐ-125
```

là tín hiệu từ khóa chính xác.

### Điểm mạnh

- mã văn bản;
- số điều;
- biểu mẫu;
- thuật ngữ đặc thù;
- từ khóa chính xác.

### Điểm yếu

Nếu người dùng diễn đạt khác với văn bản, BM25 có thể bỏ sót.

---

## 2.3. Hybrid Search

Hybrid Search kết hợp:

```text
Dense Retrieval
+
BM25 Retrieval
```

Mục tiêu:

```text
không bỏ mất semantic relevance
+
không bỏ mất exact keyword relevance
```

Ví dụ:

```text
Dense Top-k        BM25 Top-k
    │                  │
    └────────┬─────────┘
             ▼
          Fusion
             ▼
      Hybrid Candidates
```

Trong bài này ưu tiên dùng **Reciprocal Rank Fusion — RRF** vì:

- dễ hiểu;
- không cần ép score BM25 và cosine về cùng thang;
- phù hợp để minh họa Hybrid Search.

Ý tưởng:

```text
Tài liệu xuất hiện càng cao trong nhiều bảng xếp hạng
→ điểm fusion càng cao.
```

---

## 2.4. Reranking

Retrieval trả về ứng viên.

Reranker xem xét lại:

```text
Question + Candidate
```

để đánh giá độ liên quan chính xác hơn.

Luồng đúng:

```text
Corpus
   ↓
Hybrid Search
   ↓
Top 20–30 candidates
   ↓
Reranker
   ↓
Top 5
```

Không nên chạy reranker trên toàn corpus vì:

- chậm;
- tốn tài nguyên;
- không đúng vai trò của reranking.

---

## 2.5. Knowledge Graph mini

Trong buổi này không xây ontology lớn.

Mục tiêu chỉ cần thấy:

```text
VanBan
  │
  └── CONTAINS
          ↓
      DieuKhoan
          │
          ├── NEXT
          ↓
      DieuKhoan
```

Nếu dữ liệu thật có quan hệ như:

```text
THAM_CHIEU
THAY_THE
SUA_DOI_BO_SUNG
AP_DUNG_CHO
```

thì mới nạp thêm.

> Không được tự tạo quan hệ chỉ để đồ thị trông đẹp.

---

# 2. Dữ liệu sử dụng

Buổi 14 tái sử dụng trực tiếp bộ dữ liệu quy định đã có trong:

```text
kb+hops/
├── metadata.csv
├── content.csv
└── relationships.csv
```

Nếu thư mục `buoi_14/` nằm cùng cấp với `kb+hops/`, đường dẫn sử dụng trong code là:

```text
../kb+hops/metadata.csv
../kb+hops/content.csv
../kb+hops/relationships.csv
```

### Vai trò của từng file

**`metadata.csv`**

Chứa metadata của văn bản, dùng cho:

- `document_id`;
- tên/loại văn bản;
- trạng thái;
- ngày hiệu lực nếu có;
- citation;
- node `VanBan` trong Knowledge Graph.

**`content.csv`**

Là nguồn dữ liệu chính cho Retrieval:

```text
BM25
Dense Retrieval
Hybrid Search
Reranking
```

Agent phải đọc schema thật của file trước khi xác định:

```text
chunk_id
document_id
text
```

Không được đoán tên cột.

**`relationships.csv`**

Dùng cho phần Mini Knowledge Graph.

Agent phải đọc các `relationship_type` thực sự có trong file và chỉ nạp các quan hệ đã tồn tại trong dữ liệu.

Không được tự tạo quan hệ mới chỉ để graph có nhiều cạnh.

### Nguyên tắc sử dụng dữ liệu

Ba file trong `kb+hops/` là **dữ liệu nguồn chỉ đọc**.

Không sửa, không ghi đè và không xóa các file này.

Mọi dữ liệu trung gian và output của Buổi 14 phải được tạo trong:

```text
buoi_14/
```

Ví dụ:

```text
buoi_14/data/processed/
buoi_14/cache/
buoi_14/index/
buoi_14/outputs/
```

---

# 3. Cấu trúc project mong muốn

Sau khi hoàn thành, có thể có dạng:

```text
buoi_14/
│
├── data/
│   └── processed/
│       └── chunks_normalized.csv
│
├── scripts/
│   ├── inspect_project.py
│   ├── prepare_corpus.py
│   ├── baseline_retrieval.py
│   ├── hybrid_search.py
│   ├── rerank.py
│   ├── compare_retrieval.py
│   └── load_mini_kg.py
│
├── src/
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── citation.py
│
├── cypher/
│   ├── schema.cypher
│   └── demo_queries.cypher
│
├── outputs/
│   ├── inspection_report.md
│   ├── retrieval_examples.md
│   ├── retrieval_comparison.csv
│   ├── evaluation_report.md
│   └── kg_build_report.md
│
├── tests/
│   └── test_retrieval.py
│
├── .env.example
├── requirements.txt
└── README.md
```

Không cần tạo cấu trúc quá phức tạp nếu không cần.

---

# 4. BƯỚC SETUP — Kiểm tra môi trường

## Mục tiêu

Đảm bảo Python và `.venv` hoạt động trước khi Agent viết code.

### Kiểm tra Python

```bash
python --version
```

Kiểm tra:

```bash
python -c "print('Python OK')"
```

Nếu hiện:

```text
Python OK
```

thì tiếp tục.

---

## Tạo `.venv` nếu chưa có

Đứng tại:

```text
buoi_14/
```

chạy:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Kiểm tra:

```bash
python --version
```

---

## Kiểm tra đúng interpreter

### Windows

```bash
where python
where pip
```

### Linux/macOS

```bash
which python
which pip
```

Đường dẫn nên nằm trong:

```text
buoi_14/.venv/
```

---

# 5. PROMPT SETUP — Có thể nhờ Vibe Coding Agent chuẩn bị môi trường

```text
Bạn là AI Coding Agent hỗ trợ tôi chuẩn bị môi trường cho Buổi 14:
Hybrid Search + Reranking + Mini Knowledge Graph.

Toàn bộ code, config, cache và output của bước này phải được tạo trong `buoi_14/`.
Không sửa hoặc ghi đè dữ liệu nguồn trong `../kb+hops/`.

Hãy:

1. Xác nhận terminal/project hiện tại và đường dẫn buoi_14/.
2. Kiểm tra Python.
3. Kiểm tra .venv.
4. Nếu .venv chưa có, tạo buoi_14/.venv.
5. Nếu đã có, kiểm tra interpreter trong .venv thực sự chạy được.
6. Kiểm tra requirements.txt nếu có.
7. Chưa cài hàng loạt package ngay.
8. Chỉ đề xuất các dependency thực sự cần cho:
   - pandas;
   - BM25;
   - dense embedding;
   - reranking;
   - Neo4j.

9. Không cài LangChain hoặc LlamaIndex nếu không có lý do bắt buộc.
10. Không viết pipeline chính ở bước này.

Cuối cùng trả về:

ENVIRONMENT CHECK

Working root:
Python:
Virtual environment:
Existing packages:
Missing packages:
Ready:
YES / NO
```

Chỉ tiếp tục khi:

```text
Ready: YES
```

---

# 6. PROMPT 0 — Kiểm tra project, code cũ và dữ liệu trước khi làm

## Mục tiêu

Không code ngay.

Agent phải hiểu project trước.

```text
Bạn là AI Coding Agent đang chuẩn bị thực hiện Buổi 14:
Hybrid Search + Reranking + Mini Knowledge Graph.

Mọi code và output phải nằm trong `buoi_14/`.
Ba file trong `../kb+hops/` chỉ được đọc, không được sửa hoặc ghi đè.

NHIỆM VỤ:

1. Kiểm tra cấu trúc buoi_14/.
2. Liệt kê các file .py, .md, .csv, .json, requirements.txt và .env hiện có.
3. Kiểm tra và đọc trực tiếp đúng 3 file nguồn:

- ../kb+hops/metadata.csv
- ../kb+hops/content.csv
- ../kb+hops/relationships.csv

4. Không copy, move, sửa hoặc ghi đè ba file nguồn.

5. Đọc thật sự cả 3 CSV trước khi đưa ra bất kỳ giả định schema nào.
6. Với mỗi CSV báo:
   - số dòng;
   - tên cột;
   - encoding;
   - duplicate;
   - null;
   - khóa có thể sử dụng;
   - trường text nào phù hợp retrieval;
   - metadata nào phù hợp citation.

7. Kiểm tra code hiện có trong buoi_14/ nếu có:
   - code đọc gì;
   - ghi gì;
   - có hard-code path/API key/password không;
   - có lệnh xóa dữ liệu không.

8. Đặc biệt tìm:
   - os.remove
   - shutil.rmtree
   - open(..., "w")
   - DELETE
   - DROP
   - DETACH DELETE

9. Chưa chạy thao tác phá dữ liệu.

10. Kiểm tra:
   - Python;
   - .venv;
   - import pandas.

11. Không xây retrieval ở bước này.
12. Không tạo Knowledge Graph ở bước này.

Tạo báo cáo:

buoi_14/outputs/inspection_report.md

Cuối cùng in:

PROJECT PRE-CHECK

Working root:
Data:
Existing code:
Environment:
Potential risks:
Safe to continue:
YES / NO
```

## Người học cần kiểm tra

Phải nhìn thấy:

```text
Safe to continue: YES
```

Nếu là `NO`, xử lý lỗi trước.

---

# 7. PROMPT 1 — Chuẩn hóa corpus cho Retrieval và Citation

## Mục tiêu

Tạo một corpus chuẩn để cả:

```text
BM25
Dense
Hybrid
Reranker
```

đều sử dụng **cùng một tập dữ liệu**.

```text
Tiếp tục Buổi 14.

Tạo toàn bộ code và output trong `buoi_14/`.
Chỉ đọc dữ liệu nguồn từ `../kb+hops/`, không sửa dữ liệu gốc.

Dựa trên dữ liệu đã kiểm tra ở Prompt 0, hãy chuẩn hóa corpus từ:

- ../kb+hops/metadata.csv
- ../kb+hops/content.csv
- ../kb+hops/relationships.csv

Trong đó `content.csv` là nguồn text retrieval chính; `metadata.csv` dùng để bổ sung metadata/citation. Không sửa ba file nguồn.

Tạo:

buoi_14/scripts/prepare_corpus.py

Output:

buoi_14/data/processed/chunks_normalized.csv

YÊU CẦU:

1. Không giả định schema. Dựa trên cột thật đã đọc ở Prompt 0.

2. Mỗi record retrieval phải có một định danh duy nhất.

Schema chuẩn đầu ra tối thiểu:

chunk_id
document_id
text
source_file

Nếu dữ liệu có, giữ thêm:

title
document_type
chapter
section
article
clause
effective_date
status

3. Không tự bịa metadata.

4. Không bỏ mất thông tin citation quan trọng.

5. Chuẩn hóa text:
   - UTF-8;
   - khoảng trắng;
   - dòng trống;
   - không xóa số hiệu điều khoản;
   - không xóa mã văn bản;
   - không stemming quá mức làm mất nghĩa nghiệp vụ.

6. Không ghi đè CSV gốc.

7. Kiểm tra chunk_id unique.

8. In:
   - tổng số chunk;
   - số document;
   - số chunk thiếu text;
   - duplicate;
   - 3 sample record.

9. Sau khi viết code, thực sự chạy script.

10. Nếu lỗi:
   - đọc traceback;
   - sửa;
   - chạy lại.

Cập nhật README với lệnh chạy.

Không xây BM25/Dense ở bước này.
```

## Kết quả cần có

```text
buoi_14/data/processed/chunks_normalized.csv
```

---

# 8. PROMPT 2 — Xây Baseline BM25-only và Dense-only

## Mục tiêu

Phải có baseline trước khi nói Hybrid tốt hơn.

Hai retriever sử dụng **cùng corpus**.

```text
Tiếp tục Buổi 14.

Tạo/sửa code của bước này trong `buoi_14/`.
Không sửa dữ liệu trong `../kb+hops/`.

Hãy xây hai baseline độc lập:

1. BM25-only retrieval.
2. Dense-only retrieval.

Tạo code trong:

buoi_14/src/
buoi_14/scripts/

YÊU CẦU CHUNG:

1. Đọc:
   data/processed/chunks_normalized.csv

2. Cùng một input:
   question
   top_k

3. Cùng một output schema:

rank
chunk_id
document_id
text
retrieval_score
retrieval_method
citation

4. Citation phải dựa trên metadata thật.

Ví dụ nếu có:

[Tên văn bản | Điều 5 | chunk_id]

Không bịa citation.

----------------
BM25
----------------

5. Dùng BM25 hoặc implementation lexical search đơn giản, rõ ràng.

6. Tokenization phải giữ được:
   - mã văn bản;
   - số điều;
   - từ tiếng Việt.

----------------
DENSE
----------------

7. Dùng embedding model phù hợp với tiếng Việt hoặc multilingual.

8. Nếu project/bài trước đã có embedding model thống nhất thì ưu tiên dùng cùng model.

9. Không gọi embedding API cho mỗi lần query nếu có thể cache document embeddings.

10. Lưu cache/index trong:

buoi_14/cache/
hoặc
buoi_14/index/

11. Không lưu cache ra folder buổi trước.

----------------
DEMO
----------------

12. Tạo:

buoi_14/scripts/baseline_retrieval.py

Có thể chạy:

python scripts/baseline_retrieval.py --query "..." --top-k 5

13. In riêng:

BM25 RESULTS
DENSE RESULTS

14. Tạo:

outputs/retrieval_examples.md

với ít nhất 3 loại câu hỏi:
- câu có mã/số hiệu cụ thể;
- câu diễn đạt semantic;
- câu kết hợp cả hai.

15. Chạy thật và báo kết quả.

Chưa Hybrid.
Chưa Rerank.
```

## Điều cần giảng cho học viên

Hỏi học viên:

> “Vì sao BM25 đưa kết quả này lên cao?”

> “Vì sao Dense đưa một đoạn khác lên cao?”

Mục tiêu là để thấy hai retriever **không giống nhau**.

---

# 9. PROMPT 3 — Xây Hybrid Search bằng Rank Fusion

## Mục tiêu

Kết hợp:

```text
BM25 Top-N
+
Dense Top-N
```

thành Hybrid Ranking.

Ưu tiên dùng **Reciprocal Rank Fusion — RRF**.

```text
Tiếp tục Buổi 14.

Tạo/sửa code của bước này trong `buoi_14/`.
Không sửa dữ liệu nguồn trong `../kb+hops/`.

Hãy nâng cấp baseline thành Hybrid Search.

Tạo:

buoi_14/src/hybrid_retriever.py
buoi_14/scripts/hybrid_search.py

YÊU CẦU:

1. Tái sử dụng BM25 và Dense retriever của Buổi 14.

2. Không search corpus hai lần bằng hai bộ dữ liệu khác nhau.
Hai retriever phải dùng cùng:
data/processed/chunks_normalized.csv

3. Với mỗi query:
   - lấy BM25 top_n;
   - lấy Dense top_n;
   - hợp nhất theo chunk_id.

4. Dùng Reciprocal Rank Fusion (RRF) làm phương pháp fusion mặc định.

5. Không cộng trực tiếp raw BM25 score với cosine score nếu chưa chuẩn hóa.

6. Output tối thiểu:

final_rank
chunk_id
document_id
bm25_rank
dense_rank
rrf_score
text
citation

7. Candidate xuất hiện ở một retriever vẫn được giữ nếu hợp lệ.

8. Không duplicate chunk.

9. Cho phép tham số:

--query
--top-k
--candidate-k

10. Ví dụ:

python scripts/hybrid_search.py \
  --query "..." \
  --candidate-k 20 \
  --top-k 5

11. In bảng:

HYBRID RESULTS

Rank | Chunk | BM25 rank | Dense rank | RRF | Citation

12. Với các query ở Prompt 2, so sánh:
   - BM25;
   - Dense;
   - Hybrid.

13. Ghi vào:

outputs/retrieval_examples.md

14. Chạy thật và báo cáo ví dụ nào Hybrid cải thiện, ví dụ nào chưa.

Không reranking ở bước này.
```

---

# 10. PROMPT 4 — Thêm Reranking

## Mục tiêu

Luồng:

```text
BM25 + Dense
      ↓
    Hybrid
      ↓
 Top-N candidates
      ↓
   Reranker
      ↓
    Top-k
```

## Lưu ý

Reranker **không thay Hybrid Search**.

Reranker chỉ xếp lại candidate.

```text
Tiếp tục Buổi 14.

Tạo/sửa code của bước này trong `buoi_14/`.
Không sửa dữ liệu nguồn trong `../kb+hops/`.

Hãy thêm tầng Reranking sau Hybrid Retrieval.

Tạo:

buoi_14/src/reranker.py
buoi_14/scripts/rerank.py

YÊU CẦU:

1. Input của reranker phải là candidates từ Hybrid Search.

2. Không rerank toàn corpus.

3. Luồng:

question
→ Hybrid candidate_k
→ reranker
→ top_k

4. Ưu tiên Cross-Encoder reranker phù hợp multilingual nếu môi trường có thể sử dụng.

5. Trước khi tải model:
   - kiểm tra model/config hiện có;
   - báo rõ model sẽ dùng;
   - không âm thầm tải model rất lớn.

6. Nếu không thể chạy Cross-Encoder do môi trường:
   - báo rõ lý do;
   - không giả vờ rằng reranking đã chạy;
   - có thể tạo fallback đơn giản để demo pipeline nhưng phải ghi rõ FALLBACK, không gọi đó là neural reranker.

7. Output:

final_rank
chunk_id
document_id
hybrid_rank
hybrid_score
rerank_score
text
citation

8. Cho phép:

python scripts/rerank.py \
  --query "..." \
  --candidate-k 20 \
  --top-k 5

9. In:

BEFORE RERANK
AFTER RERANK

để học viên nhìn thấy thứ tự thay đổi.

10. Không làm mất citation.

11. Không thay đổi corpus.

12. Chạy thật ít nhất 3 query và ghi kết quả vào:

outputs/retrieval_examples.md

13. Nếu model reranker không chạy được:
   - ghi vào báo cáo;
   - tiếp tục các phần khác của bài.
```

---

# 11. Kết quả trực quan sau Prompt 4

Đây là phần giảng viên nên dừng lại để demo.

Ví dụ terminal:

```text
QUERY:
"Ai có thẩm quyền phê duyệt giao dịch vượt hạn mức?"

BM25
1. DK-014
2. DK-022
3. DK-008

DENSE
1. DK-031
2. DK-014
3. DK-017

HYBRID
1. DK-014
2. DK-031
3. DK-022

AFTER RERANK
1. DK-031
2. DK-014
3. DK-017
```

Giảng viên giải thích:

```text
Retrieval
→ tìm ứng viên.

Hybrid
→ hợp nhất nhiều loại bằng chứng retrieval.

Reranking
→ xem lại từng ứng viên trong ngữ cảnh câu hỏi
  và sắp xếp thứ tự cuối cùng.
```

---

# 12. PROMPT 5 — Đánh giá BM25 vs Dense vs Hybrid vs Hybrid + Reranking

## Mục tiêu

Không kết luận “Hybrid tốt hơn” chỉ bằng cảm giác.

Cần một tập câu hỏi nhỏ để so sánh.

```text
Tiếp tục Buổi 14.

Mọi code và output của bước này phải nằm trong `buoi_14/`.

Hãy xây evaluation nhỏ cho retrieval.

Tạo:

buoi_14/data/eval/questions.csv

Nếu chưa có bộ câu hỏi vàng:
- đề xuất một bộ câu hỏi dựa trên corpus thật;
- không bịa gold chunk;
- chỉ gán expected_chunk_id nếu có thể xác minh từ dữ liệu.

Schema:

question_id
question
expected_chunk_id
query_type
note

query_type gồm tối thiểu:

EXACT_KEYWORD
SEMANTIC
MIXED

Tạo:

buoi_14/scripts/compare_retrieval.py

So sánh:

1. BM25-only
2. Dense-only
3. Hybrid
4. Hybrid + Rerank

Đo tối thiểu:

Hit@1
Hit@3
Hit@5

Nếu có đủ gold relevance có thể thêm:

MRR

YÊU CẦU:

1. Cùng corpus.
2. Cùng bộ câu hỏi.
3. Cùng evaluation protocol.
4. Không thay gold để làm kết quả đẹp hơn.
5. Không bỏ query thất bại.
6. Báo cả lỗi.

Output:

outputs/retrieval_comparison.csv
outputs/evaluation_report.md

evaluation_report.md phải có:

- số câu hỏi;
- metric từng cấu hình;
- nhóm query nào BM25 mạnh;
- nhóm query nào Dense mạnh;
- Hybrid có giúp không;
- Reranking có đổi ranking không;
- failure cases;
- kết luận có giới hạn.

Chạy thật evaluation.
```

---

# 13. PROMPT 6 — Xây Knowledge Graph mini cho bộ quy định nội bộ

## Mục tiêu

Từ cùng corpus, dựng graph tối giản:

```text
VanBan
   ↓ CONTAINS
DieuKhoan
```

và các quan hệ thật nếu có.

```text
Tiếp tục Buổi 14.

Tạo toàn bộ code/Cypher/output của bước này trong `buoi_14/`.
Đọc dữ liệu nguồn từ `../kb+hops/`.
Không sửa các file nguồn và không xóa toàn bộ graph Neo4j hiện có.

Hãy xây Knowledge Graph mini từ:

- ../kb+hops/metadata.csv
- ../kb+hops/content.csv
- ../kb+hops/relationships.csv
- buoi_14/data/processed/chunks_normalized.csv

Trước tiên:

1. Đọc và kiểm tra trực tiếp ba file CSV trong `../kb+hops/`.
2. Không giả định relationship type.
3. Chỉ nạp những relation thực sự có trong dữ liệu.

ONTOLOGY MVP:

(:VanBan)
(:DieuKhoan)

Quan hệ bắt buộc khi dữ liệu hỗ trợ:

(:VanBan)-[:CONTAINS]->(:DieuKhoan)

Quan hệ cấu trúc nếu xác định được thứ tự:

(:DieuKhoan)-[:NEXT]->(:DieuKhoan)

Nếu relationships.csv có các loại quan hệ khác, hãy mapping có kiểm soát.
Không tạo relation type mới nếu chưa giải thích.

Tạo:

buoi_14/cypher/schema.cypher
buoi_14/cypher/demo_queries.cypher
buoi_14/scripts/load_mini_kg.py

NODE:

VanBan:
- id
- title nếu có
- document_type nếu có
- status nếu có

DieuKhoan:
- id
- document_id
- text
- article/clause nếu có

RELATION:

Mỗi relation lấy từ dữ liệu phải giữ source/provenance nếu dữ liệu có.

AN TOÀN NEO4J:

1. Dùng MERGE theo ID.
2. Parameterized Cypher.
3. Không hard-code password.
4. Đọc:

NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
NEO4J_DATABASE

từ .env.

5. Mọi node/relationship của bài này nên có:

lab_session = "buoi_14"

để phân biệt dữ liệu buổi học.

6. TUYỆT ĐỐI KHÔNG chạy:

MATCH (n) DETACH DELETE n

7. Nếu cần làm sạch dữ liệu Buổi 14, chỉ xóa node/cạnh có:

lab_session = "buoi_14"

và phải báo trước.

8. Nếu Neo4j chưa chạy:
   - báo hướng dẫn;
   - không làm hỏng các phần retrieval.

Sau khi nạp:
- đếm node theo label;
- đếm relation theo type;
- kiểm tra orphan;
- ghi:

outputs/kg_build_report.md

Thực sự chạy nếu Neo4j đang sẵn sàng.
```

---

# 14. Các query trực quan cần có trong Neo4j

Trong:

```text
cypher/demo_queries.cypher
```

cần có ít nhất:

## Query A — Xem graph Buổi 14

```cypher
MATCH (n {lab_session: "buoi_14"})-[r]->(m {lab_session: "buoi_14"})
RETURN n, r, m
LIMIT 100
```

## Query B — Từ văn bản tới điều khoản

```cypher
MATCH (v:VanBan {lab_session: "buoi_14"})-[:CONTAINS]->(d:DieuKhoan)
RETURN v, d
LIMIT 50
```

## Query C — Xem chuỗi điều khoản

Logic:

```text
DieuKhoan
   ↓ NEXT
DieuKhoan
   ↓ NEXT
DieuKhoan
```

## Query D — Quan hệ văn bản có trong dữ liệu

Chỉ tạo query theo relation type thực tế.

## Query E — Tìm node không có liên kết

Dùng để kiểm tra chất lượng graph.

---

# 15. Kết quả trực quan của Knowledge Graph mini

Khi mở Neo4j Browser và chạy demo query, học viên cần thấy:

```text
          [Văn bản A]
          /    |    \
         /     |     \
 CONTAINS  CONTAINS  CONTAINS
      ↓        ↓        ↓
 [Điều 1] → [Điều 2] → [Điều 3]
              NEXT       NEXT
```

Nếu corpus có quan hệ văn bản:

```text
[Văn bản A] ──THAM_CHIEU──> [Văn bản B]
```

hoặc:

```text
[Văn bản mới] ──THAY_THE──> [Văn bản cũ]
```

thì mới hiển thị.

---

# 16. PROMPT 7 — Tích hợp Retrieval với citation và chuẩn bị đường sang Graph RAG

## Mục tiêu

Buổi này chưa cần xây Graph RAG phức tạp.

Chỉ cần output retrieval đủ sạch để bài sau có thể sử dụng graph.

```text
Tiếp tục Buổi 14.

Mọi code và output của bước này phải nằm trong `buoi_14/`.

Hãy hoàn thiện một hàm retrieval thống nhất:

retrieve(question, method, top_k)

method:

bm25
dense
hybrid
hybrid_rerank

Output mỗi result:

rank
chunk_id
document_id
text
score
citation
retrieval_method

Với hybrid_rerank:
- có hybrid score;
- có rerank score nếu neural reranker thực sự chạy.

Tạo:

buoi_14/scripts/query_demo.py

Có CLI:

python scripts/query_demo.py \
  --query "..." \
  --method hybrid_rerank \
  --top-k 5

Output terminal phải dễ đọc cho học viên.

Phần cuối in:

GRAPH HINTS

- document_id của các chunk retrieved;
- chunk_id;
- các relation trực tiếp liên quan nếu Mini KG đã được nạp và Neo4j sẵn sàng.

Không thực hiện traversal nhiều hops phức tạp.
Không biến bài này thành Graph RAG hoàn chỉnh.

Mục tiêu là chuẩn bị dữ liệu sạch cho buổi Graph RAG sau.
```

---


# 17. PROMPT 8 — Xây demo Streamlit cho Hybrid Search + Reranking

## Mục tiêu

Tạo một giao diện trực quan để học viên có thể:

- nhập một câu hỏi;
- chọn phương pháp retrieval;
- so sánh BM25, Dense, Hybrid và Hybrid + Rerank;
- xem Top-k;
- xem citation;
- xem thứ hạng trước và sau reranking;
- nhìn thấy `Graph hints` liên quan tới `document_id` và `chunk_id`.

Knowledge Graph đầy đủ vẫn được xem trong Neo4j Browser.

## Prompt 8

```text
Tiếp tục Buổi 14.

Tạo toàn bộ code và output của bước này trong `buoi_14/`.
Không sửa dữ liệu nguồn trong `../kb+hops/`.

Hãy xây một demo Streamlit đơn giản cho hệ thống retrieval đã hoàn thành.

Tạo:

buoi_14/app.py

Nếu cần helper riêng, tạo trong:

buoi_14/src/

YÊU CẦU GIAO DIỆN:

1. Tiêu đề:

RAG Hybrid Search — Buổi 14

2. Có ô nhập:

Câu hỏi

3. Có lựa chọn method:

BM25
Dense
Hybrid
Hybrid + Rerank

4. Có lựa chọn Top-k.

5. Có nút:

Tìm kiếm

6. Khi chạy, gọi đúng retrieval pipeline đã có của Buổi 14.
Không viết lại một pipeline khác chỉ dành cho Streamlit.

7. Mỗi kết quả phải hiển thị:

- rank;
- chunk_id;
- document_id;
- score;
- retrieval_method;
- citation;
- text.

8. Với Hybrid, nếu dữ liệu có:
- bm25_rank;
- dense_rank;
- rrf_score.

9. Với Hybrid + Rerank, hiển thị thêm:

BEFORE RERANK
AFTER RERANK

hoặc một bảng so sánh thứ hạng trước/sau.

10. Có phần:

Graph hints

hiển thị tối thiểu:

- document_id;
- chunk_id;
- relation trực tiếp nếu Neo4j đang sẵn sàng.

11. Không cố render toàn bộ Knowledge Graph trong Streamlit.
Knowledge Graph đầy đủ vẫn xem bằng Neo4j Browser.

12. Nếu Neo4j không chạy:
- Streamlit vẫn phải hoạt động cho retrieval;
- Graph hints ghi rõ Neo4j chưa sẵn sàng.

13. Không hard-code password, API key hoặc Neo4j credentials.

14. Nếu cần thêm Streamlit vào requirements.txt, hãy cập nhật.

15. Chạy kiểm tra import:

python -c "import streamlit; print('Streamlit OK')"

16. Sau đó hướng dẫn chạy:

streamlit run app.py

17. Không giả định localhost nếu Streamlit báo một URL khác.
Báo lại URL thực tế mà Streamlit cung cấp.

18. Sau khi app chạy, kiểm tra ít nhất:
- một query exact keyword;
- một query semantic;
- một query với Hybrid + Rerank.

19. Nếu app lỗi:
- đọc traceback;
- sửa code;
- chạy lại.

20. Cập nhật README.md với:
- cách chạy Streamlit;
- cách dừng Streamlit;
- cách chọn method;
- cách hiểu các trường kết quả.

Cuối cùng báo:

STREAMLIT DEMO

App:
Method options:
Top-k:
Citation:
Before/After Rerank:
Graph hints:
Run command:
Status:
READY / NOT READY
```

## Kết quả trực quan mong đợi

Giao diện có thể gần dạng:

```text
┌──────────────────────────────────────────────┐
│         RAG HYBRID SEARCH — BUỔI 14         │
├──────────────────────────────────────────────┤
│ Câu hỏi                                     │
│ [........................................]  │
│                                              │
│ Method                                      │
│ [BM25 | Dense | Hybrid | Hybrid+Rerank]     │
│                                              │
│ Top-k [5]                    [Tìm kiếm]      │
├──────────────────────────────────────────────┤
│ RESULT #1                                   │
│ Chunk: ...                                  │
│ Document: ...                               │
│ Score: ...                                  │
│ Citation: ...                               │
│ Text: ...                                   │
├──────────────────────────────────────────────┤
│ BEFORE RERANK          AFTER RERANK          │
│ 1. Chunk A             1. Chunk C            │
│ 2. Chunk B      →      2. Chunk A            │
│ 3. Chunk C             3. Chunk B            │
├──────────────────────────────────────────────┤
│ GRAPH HINTS                                  │
│ Document → Chunk → direct relations          │
└──────────────────────────────────────────────┘
```

## Chạy app

Tại:

```text
buoi_14/
```

chạy:

```bash
streamlit run app.py
```

Streamlit thường mở giao diện local trong trình duyệt. Dùng URL thực tế mà terminal Streamlit hiển thị.

---



---

# 21. Những lỗi cần đặc biệt chú ý

## Lỗi 1 — Agent cộng thẳng BM25 score với cosine score

Sai nếu chưa có normalization hợp lý.

Trong bài này ưu tiên:

```text
RRF
```

để dễ kiểm soát.

---

## Lỗi 2 — Gọi Hybrid nhưng thực tế chỉ chạy Dense

Kiểm tra output phải có:

```text
bm25_rank
dense_rank
```

---

## Lỗi 3 — Gọi Reranker nhưng chỉ sort lại hybrid_score

Đó không phải reranking thực sự.

Nếu dùng neural reranker, report phải ghi model.

Nếu fallback, phải ghi:

```text
FALLBACK
```

---

## Lỗi 4 — Rerank toàn corpus

Không đúng pipeline.

Phải là:

```text
retrieval candidate_k
→ reranking
→ top_k
```

---

## Lỗi 5 — Citation bị mất sau Hybrid/Rerank

Mọi result cuối phải giữ:

```text
chunk_id
document_id
citation
```

---

## Lỗi 6 — Agent tạo quan hệ Knowledge Graph từ suy đoán

Không chấp nhận.

Quan hệ phải đến từ:

```text
cấu trúc document/chunk
hoặc
relationships.csv
```

---

## Lỗi 7 — Xóa Neo4j của buổi trước

Đặc biệt không chạy:

```cypher
MATCH (n)
DETACH DELETE n
```

Dữ liệu Buổi 14 phải có:

```text
lab_session = "buoi_14"
```

để quản lý riêng.

---

## Lỗi 8 — Code bị tạo ra ngoài folder Buổi 14

Cuối buổi phải kiểm tra:

```text
mọi code/output mới
→ nằm trong buoi_14/
```

Đây là tiêu chí bắt buộc.


---

# 23. Sản phẩm cuối cùng

anh/chị phải nhìn thấy hai sản phẩm.

## Sản phẩm A — Retrieval Pipeline

```text
Question
   ↓
BM25 + Dense
   ↓
Hybrid RRF
   ↓
Reranking
   ↓
Top-k + Citation
```

## Sản phẩm B — Streamlit Demo

```text
Question
   ↓
[BM25 | Dense | Hybrid | Hybrid + Rerank]
   ↓
Top-k + Citation
   ↓
Before / After Rerank
```

## Sản phẩm C — Mini Knowledge Graph

```text
VanBan
   ↓
DieuKhoan
   ↓
DieuKhoan
```

và các quan hệ có thật trong corpus.

---

# 24. Yêu cầu nộp bài

Tối thiểu:

```text
buoi_14/
├── scripts/
├── src/
├── data/processed/chunks_normalized.csv
├── outputs/
│   ├── inspection_report.md
│   ├── retrieval_examples.md
│   ├── retrieval_comparison.csv
│   ├── evaluation_report.md
│   ├── kg_build_report.md
│   └── final_validation_report.md
├── cypher/
│   ├── schema.cypher
│   └── demo_queries.cypher
├── app.py
├── requirements.txt
└── README.md
```

Nếu Neo4j chưa chạy được:

```text
kg_build_report.md
```

phải ghi rõ:

```text
NOT RUN
```

và lý do.

Không được giả kết quả.

---

# 25. Tiêu chí đạt

Bài được xem là đạt khi:

- toàn bộ code mới nằm trong `buoi_14/`;
- không sửa code buổi trước;
- corpus được chuẩn hóa;
- BM25 chạy được;
- Dense Retrieval chạy được;
- Hybrid Search thực sự sử dụng cả hai retriever;
- fusion không cộng raw score sai cách;
- reranker chỉ xử lý candidates;
- có thể nhìn thấy ranking trước và sau rerank;
- citation còn nguyên sau retrieval;
- có evaluation chung cho bốn cấu hình;
- có Streamlit demo dùng đúng retrieval pipeline;
- Streamlit hiển thị Top-k, citation và Before/After Rerank;
- Knowledge Graph mini chỉ dùng quan hệ có căn cứ;
- không xóa dữ liệu Neo4j của buổi trước;
- có báo cáo validation cuối.

---

# 26. Checklist nhanh trước khi kết thúc buổi

- [ ] Terminal đang làm việc trong `buoi_14/`.
- [ ] `.venv` của Buổi 14 hoạt động.
- [ ] Không sửa code các buổi trước.
- [ ] Corpus đã chuẩn hóa.
- [ ] BM25 có kết quả.
- [ ] Dense có kết quả.
- [ ] Hybrid có `bm25_rank` và `dense_rank`.
- [ ] Reranker nhận candidate từ Hybrid.
- [ ] Có Before/After Rerank.
- [ ] Citation không bị mất.
- [ ] Có evaluation report.
- [ ] Streamlit chạy được.
- [ ] Streamlit chọn được 4 retrieval method.
- [ ] Streamlit hiển thị citation.
- [ ] Streamlit hiển thị Before/After Rerank.
- [ ] Mini KG chỉ chứa quan hệ có nguồn.
- [ ] Neo4j không bị xóa toàn bộ.
- [ ] Dữ liệu graph có `lab_session = "buoi_14"`.
- [ ] Final validation báo `READY FOR DEMO: YES`.

---

# 27. Ghi nhớ

Buổi 14 không chỉ nhằm “thêm nhiều kỹ thuật hơn”.

Ý nghĩa chính là anh/chị hiểu:

```text
Dense Retrieval
không thay thế hoàn toàn keyword retrieval.

BM25
không thay thế semantic retrieval.

Hybrid
kết hợp hai tín hiệu.

Reranking
làm bước lựa chọn cuối tốt hơn.

Knowledge Graph
bổ sung quan hệ có cấu trúc mà retrieval thuần túy không thể hiện rõ.
```

Đây là nền tảng để chuyển sang các bài sau về:

```text
Graph RAG
Multi-hop Retrieval
Constraint-based Retrieval
Evidence-aware Reasoning
```
