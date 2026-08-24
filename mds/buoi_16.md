# BUỔI 16 — Đánh giá hiệu năng hệ thống RAG (RAG Evaluation) bằng Ragas

## Mục tiêu

Trong bài thực hành này, **học viên** sử dụng **AI Coding Agent / Vibe Coding** để thiết kế và cài đặt một quy trình đánh giá tự động chất lượng hệ thống Retrieval-Augmented Generation (RAG) sử dụng thư viện **Ragas**. Quy trình này được tích hợp toàn diện vào **một script duy nhất** chạy bằng **một Prompt duy nhất**, sử dụng cấu hình **2 mô hình độc lập** nhằm tách biệt vai trò sinh câu trả lời và vai trò đánh giá khách quan gọi qua Hugging Face Router API:
1. **Model Pipeline (Generator)**: Sử dụng mô hình `Qwen/Qwen3.5-9B:deepinfra` để sinh câu hỏi benchmark và sinh câu trả lời RAG dựa trên ngữ cảnh đã truy xuất.
2. **Model Judger (Evaluator)**: Sử dụng mô hình `openai/gpt-oss-20b:deepinfra` đóng vai trò làm trọng tài chấm điểm chất lượng (LLM-as-a-judge) cho các chỉ số Ragas.

Các mục tiêu cốt lõi:
1. **Tự động hóa toàn bộ quy trình qua một script**:
   - Viết tập lệnh [`evaluate_rag_pipeline.py`](file:///e:/neo4j/graph_rag_labs/buoi_15/scripts/evaluate_rag_pipeline.py) để tự động hóa: sinh câu hỏi, truy xuất ngữ cảnh bằng [`SecureRetriever`](file:///e:/neo4j/graph_rag_labs/buoi_15/src/secure_retriever.py), sinh câu trả lời, tính toán các metrics của Ragas, và viết báo cáo đánh giá.
2. **Xây dựng bộ câu hỏi & đáp án chuẩn (Golden Dataset)**:
   - Sinh tự động 20 câu hỏi và đáp án từ tệp dữ liệu [`chunks_secure.csv`](file:///e:/neo4j/graph_rag_labs/buoi_15/data/processed/chunks_secure.csv) phân chia theo use cases (Nhân sự, Rủi ro, Quy định chung) và độ khó (Easy, Medium, Hard).
3. **Đánh giá & Tính toán 4 metrics Ragas**:
   - Sử dụng mô hình trọng tài để tính điểm: *Context Precision*, *Context Recall*, *Faithfulness*, và *Answer Relevancy*.
4. **Xuất báo cáo và đề xuất tối ưu**:
   - Phân tích lỗi và tự động lưu báo cáo đánh giá chi tiết ra tệp [`ragas_evaluation_report.md`](file:///e:/neo4j/graph_rag_labs/buoi_15/outputs/ragas_evaluation_report.md).

---

# 1. Kiến thức cần hiểu trước khi thực hành

Để thực hiện đánh giá RAG hiệu quả, học viên cần nắm rõ phương pháp đánh giá không cần nhãn người dùng (LLM-as-a-judge) và ý nghĩa của 4 chỉ số cốt lõi.

## 1.1. Tại sao nên sử dụng 2 mô hình độc lập (Pipeline vs Judger)?
Khi xây dựng và đánh giá hệ thống RAG, việc tách biệt mô hình tạo nội dung (Generator) và mô hình chấm điểm (Judger) là một chuẩn công nghiệp quan trọng:
- **Tránh thiên vị bản thân (Self-preference Bias)**: Nếu một mô hình tự chấm điểm câu trả lời của chính nó, nó có xu hướng cho điểm cao hơn (do phong cách ngôn ngữ giống nhau). Sử dụng một mô hình khác để chấm điểm mang lại kết quả khách quan hơn.
- **Tối ưu hóa hiệu năng**: Mô hình sinh câu trả lời cho người dùng cuối (Qwen) có thể cần tối ưu về tốc độ phản hồi hoặc khả năng đọc tài liệu đa phương tiện, trong khi mô hình chấm điểm (GPT-OSS) cần khả năng lập luận, phân tích logic tốt nhất để tìm lỗi sai hoặc ảo tưởng.

## 1.2. Giải nghĩa 4 Metrics cốt lõi của Ragas

```text
               ┌───────────────────────┐
               │    User Question      │
               └───────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
 ┌──────────────────────┐    ┌──────────────────────┐
 │  Retrieved Context   ├────┼─►  Generated Answer  │
 └───────────┬──────────┘    │   └──────────┬───────┘
             │               │              │
             │ Context       │ Faithfulness │ Answer
             │ Recall        │              │ Relevancy
             ▼               ▼              ▼
 ┌──────────────────────┐    ┌──────────────────────┐
 │     Ground Truth     ├────┼──────────────────────┘
 └──────────────────────┘    │ Context Precision
                             └──────────────────────┘
```

1. **Context Recall (Độ phủ của ngữ cảnh)**:
   - **Định nghĩa**: Đo lường xem ngữ cảnh truy xuất được (`contexts`) có chứa toàn bộ thông tin cần thiết để đưa ra đáp án chuẩn (`ground_truth`) hay không.
   - **Ý nghĩa**: Đánh giá khả năng tìm kiếm của Retriever. Nếu chỉ số này thấp, hệ thống RAG đang bị thiếu hụt dữ liệu đầu vào.
2. **Context Precision (Độ chuẩn xác ngữ cảnh)**:
   - **Định nghĩa**: Đo lường xem các tài liệu thực sự liên quan có được xếp ở các thứ hạng đầu (Rank cao) trong kết quả truy xuất hay không.
   - **Ý nghĩa**: Rất quan trọng vì LLM Generator thường bị ảnh hưởng nhiều nhất bởi các thông tin ở đầu ngữ cảnh (Lost in the Middle effect).
3. **Faithfulness (Độ trung thực / Không ảo tưởng)**:
   - **Định nghĩa**: Đo lường xem câu trả lời được sinh ra (`answer`) có hoàn toàn dựa trên ngữ cảnh đã truy xuất (`contexts`) hay không, hay tự suy diễn/bịa đặt thông tin (Hallucination).
   - **Ý nghĩa**: Đánh giá độ tin cậy của Generator. Điểm số cao tức là LLM trả lời trung thực và không bịa chuyện.
4. **Answer Relevancy (Độ phù hợp của câu trả lời)**:
   - **Định nghĩa**: Đo lường mức độ khớp giữa câu trả lời được sinh ra (`answer`) và câu hỏi ban đầu (`question`).
   - **Ý nghĩa**: Điểm số thấp phản ánh câu trả lời bị lạc đề hoặc quá dài dòng, không đi vào trọng tâm câu hỏi.

---

# 2. Quy trình thực hành và Cấu trúc thư mục rút gọn

Học viên thực hiện toàn bộ bài lab chỉ bằng **một Prompt duy nhất** cung cấp cho AI Agent. Tập lệnh sẽ xử lý hoàn toàn luồng nghiệp vụ tự động và cấu trúc thư mục gọn gàng như sau:

- Đầu vào: [`chunks_secure.csv`](file:///e:/neo4j/graph_rag_labs/buoi_15/data/processed/chunks_secure.csv).
- Cấu hình API key được nạp từ tệp tin [`.env`](file:///e:/neo4j/graph_rag_labs/buoi_15/.env) cục bộ.

## Cấu trúc project sau khi hoàn thành buổi 16:
```text
buoi_15/
│
├── data/
│   ├── processed/
│   │   └── chunks_secure.csv
│   └── eval/                            # [NEW] Thư mục chứa dữ liệu đánh giá
│       ├── qa_dataset.csv               # Bộ câu hỏi & đáp án chuẩn (Golden Dataset)
│       └── evaluation_results.csv       # Kết quả đánh giá chi tiết từng câu hỏi từ Ragas
│
├── scripts/
│   ├── ...
│   └── evaluate_rag_pipeline.py         # [NEW] Script duy nhất tự động hóa toàn bộ quy trình RAG Eval
│
└── outputs/
    └── ragas_evaluation_report.md       # [NEW] Báo cáo đánh giá chi tiết & phương án tối ưu
```

---

# 3. Hướng dẫn thực hành

```text
Toàn bộ code và cấu hình của bài này phải được triển khai trực tiếp bên trong thư mục `buoi_15/`.
Thông tin kết nối và API Key sẽ được đọc từ file cấu hình của tôi tại `buoi_15/.env` (bao gồm HF_TOKEN).

Nhiệm vụ: Cài đặt thư viện, thiết kế và thực thi một tập lệnh duy nhất `buoi_15/scripts/evaluate_rag_pipeline.py` để tự động hóa toàn bộ quy trình đánh giá hệ thống RAG.

YÊU CẦU CHI TIẾT:
1. Cài đặt các thư viện cần thiết (`ragas`, `datasets`, `langchain-openai`, `langchain-huggingface`) bằng pip hoặc uv.
2. Viết tập lệnh `evaluate_rag_pipeline.py` thực hiện tuần tự các bước sau:
   a. Sinh bộ câu hỏi thử nghiệm (Golden Dataset):
      - Đọc dữ liệu từ `buoi_15/data/processed/chunks_secure.csv`.
      - Chọn ngẫu nhiên khoảng 10-15 chunks tiêu biểu thuộc các nhóm bảo mật khác nhau (HR, Risk, Common).
      - Tự sinh 20 câu hỏi (`question`) và đáp án chuẩn (`ground_truth`) dựa trên nội dung chunks nguồn đó.
      - Phân bổ câu hỏi theo nhóm độ khó ("easy", "medium", "hard") và loại usecase. Lưu bộ câu hỏi này ra `buoi_15/data/eval/qa_dataset.csv`.
   b. Chạy RAG Pipeline để thu thập kết quả sinh câu trả lời:
      - Đọc tệp `qa_dataset.csv` vừa tạo.
      - Với mỗi câu hỏi, gọi hàm retrieve của `SecureRetriever` trong `buoi_15/src/secure_retriever.py` (giả định vai trò người dùng có toàn quyền để lấy đủ ngữ cảnh, ví dụ: `["Admin", "HR", "Risk_Manager", "Staff"]`) để thu được danh sách văn bản ngữ cảnh (`contexts`).
      - Gửi câu hỏi kèm `contexts` đến mô hình "Qwen/Qwen3.5-9B:deepinfra" qua HF Router để sinh câu trả lời RAG (`answer`) dựa trên prompt template yêu cầu chỉ trả lời dựa trên ngữ cảnh được cung cấp, tắt reasoning của mô hình.
   c. Chạy Ragas đánh giá 4 metrics: Context Precision, Context Recall, Faithfulness, và Answer Relevancy:
      - Cấu hình LLM trọng tài (Judger) sử dụng `ChatOpenAI` gọi mô hình "openai/gpt-oss-20b:deepinfra" trỏ qua HF Router (base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"]), tắt reasoning của mô hình.
      - Chạy chấm điểm Ragas và lưu kết quả chi tiết từng câu hỏi ra `buoi_15/data/eval/evaluation_results.csv`.
   d. Viết báo cáo đánh giá tự động:
      - Phân tích điểm số từ file kết quả chấm điểm.
      - Tự động ghi nhận và xuất báo cáo chi tiết ra tệp `buoi_15/outputs/ragas_evaluation_report.md` (bao gồm: bảng tóm tắt điểm trung bình của 4 metrics, phân tích nguyên nhân lỗi đối với các câu hỏi có điểm số thấp < 0.7, và đề xuất tối ưu hóa hệ thống).
3. Thực thi tập lệnh này ngay lập tức để hoàn thành toàn bộ quy trình, in ra điểm trung bình của 4 metrics thu được và hiển thị báo cáo mẫu lên màn hình.
Cách gọi mô hình sử dụng openai sdk:
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)
```

---

# 4. Các giải pháp cải thiện hệ thống RAG dựa trên kết quả Đánh giá

Sau khi chạy đánh giá, học viên có thể gặp các lỗi phổ biến sau. Bảng dưới đây cung cấp các giải pháp kỹ thuật tương ứng để tối ưu hóa hệ thống:

| Triệu chứng (Chỉ số thấp) | Nguyên nhân phổ biến | Giải pháp kỹ thuật đề xuất |
| :--- | :--- | :--- |
| **Context Recall thấp** (< 0.7) | - Truy vấn BM25 bỏ lỡ các từ đồng nghĩa.<br>- Dense search gặp vấn đề với các từ viết tắt chuyên ngành.<br>- Chỉ số `top_k` quá nhỏ không chứa đủ thông tin. | - Tăng giá trị `top_k` (ví dụ từ 5 lên 8).<br>- Tích hợp thêm mở rộng truy vấn bằng LLM (Query Expansion).<br>- Sử dụng cấu trúc đồ thị Neo4j để lấy thêm các node lân cận (`NEXT`, `CONTAINS`). |
| **Context Precision thấp** (< 0.7) | - Các chunk không liên quan có điểm tương đồng vector cao và chiếm vị trí đầu.<br>- Cơ chế Hybrid Fusion (RRF) chưa cân bằng tốt giữa từ khóa và ngữ nghĩa. | - Cấu hình lại trọng số hoặc tham số $k$ trong RRF.<br>- Nâng cấp hoặc tinh chỉnh mô hình Cross-Encoder Reranker để xếp hạng lại chính xác hơn. |
| **Faithfulness thấp** (< 0.8) | - Generator tự ý bổ sung kiến thức ngoại lai không có trong ngữ cảnh (hallucination).<br>- Ngữ cảnh quá dài khiến LLM bị nhiễu thông tin. | - Tối ưu lại prompt hệ thống: Yêu cầu cực kỳ nghiêm ngặt chỉ trả lời dựa vào context.<br>- Áp dụng kỹ thuật sinh câu trả lời từng bước (Chain of Thought).<br>- Rút ngắn độ dài chunk hoặc lọc bớt nhiễu trước khi gửi sang Generator. |
| **Answer Relevancy thấp** (< 0.8) | - LLM trả lời chung chung, không đi thẳng vào câu hỏi.<br>- Câu trả lời quá dài dòng hoặc chứa nhiều định dạng thừa. | - Điều chỉnh prompt của Generator: Yêu cầu câu trả lời ngắn gọn, súc tích.<br>- Cung cấp một vài ví dụ mẫu (Few-shot prompting) trong prompt sinh câu trả lời. |

---

# 5. Câu hỏi thảo luận và Đánh giá năng lực của học viên

Để hoàn thành bài học, học viên cần suy nghĩ và thảo luận các vấn đề thực tế sau:

1. **Ý nghĩa của việc tách biệt mô hình chạy Pipeline và mô hình Judger**:
   * *“Tại sao chúng ta không nên sử dụng chính mô hình Qwen sinh câu trả lời để làm mô hình chấm điểm đánh giá Ragas? Hiện tượng 'Self-preference bias' sẽ làm sai lệch kết quả đánh giá như thế nào?”*
2. **Vấn đề "Data Leakage" trong quá trình chấm điểm Ragas**:
   * *“Nếu chúng ta sử dụng mô hình LLM-as-a-judge là một mô hình công cộng (như ChatGPT hoặc DeepSeek API công cộng) để đánh giá tài liệu bảo mật nội bộ doanh nghiệp, điều này có vi phạm chính sách an toàn thông tin không? Giải pháp thay thế là gì?”*
   *(Gợi ý: Cần triển khai mô hình chấm điểm nội bộ như Llama-3-Instruct hoặc Qwen-Instruct chạy offline/local hoặc trên hạ tầng đám mây bảo mật của doanh nghiệp).*
3. **Sự đánh đổi giữa Context Recall và Faithfulness**:
   * *“Tại sao khi tăng số lượng văn bản truy xuất (tăng `top_k`) để cải thiện Context Recall thì chỉ số Faithfulness đôi khi lại bị giảm xuống?”*
   *(Gợi ý: Ngữ cảnh quá dài và chứa nhiều thông tin ít liên quan có thể làm LLM Generator bị phân tâm, dẫn đến việc tổng hợp sai lệch hoặc sinh thông tin không có căn cứ).*
4. **Tính khách quan của LLM-as-a-judge**:
   * *“Làm thế nào để đảm bảo mô hình LLM đánh giá không bị thiên vị (bias) đối với định dạng câu trả lời của chính nó?”*
   *(Gợi ý: Sử dụng mô hình chấm điểm (Evaluator) khác và mạnh hơn mô hình sinh câu trả lời (Generator), ví dụ Generator dùng Qwen-35B nhưng Evaluator dùng DeepSeek-V4 hoặc Gemini-1.5-Pro).*