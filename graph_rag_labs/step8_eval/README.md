# Bước 8 — Kiểm thử Pipeline QA

Chạy 5 câu hỏi phức tạp qua toàn bộ pipeline (multi-hop retrieval + Gemini)
với 3 cấu hình `num_hops` ∈ {0, 1, 2} → ghi báo cáo so sánh vào
`output/qa_comparison.md`.

## Bộ 5 câu hỏi

| ID | Câu hỏi |
| --- | --- |
| Q1 | Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm? |
| Q2 | Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ văn bản nào, và quy định về hồ sơ, thủ tục cấp giấy phép lần đầu của ngân hàng thương mại gồm những tài liệu gì? |
| Q3 | Thông tư số 01/2025/TT-NHNN quy định về cấp giấy phép quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi văn bản nào, và những nội dung sửa đổi bổ sung chính là gì? |
| Q4 | Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào, và luật đó quy định chức năng nhiệm vụ của cơ quan nào? |
| Q5 | Hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào, và Thông tư đó có được sửa đổi bổ sung bởi văn bản nào không? |

## Chạy

```bash
cd graph_rag_labs/step8_eval
python -m src.run_eval
```

Output:
- `output/eval_results.json` — full structured log
- `output/qa_comparison.md` — bảng Markdown so sánh

## Status codes (auto-judge)

Mỗi (query, num_hops) combination được đánh dấu:

| Status | Ý nghĩa |
| --- | --- |
| `answered` | Trích xuất được nội dung từ context |
| `no-context` | Model nói "ngữ cảnh không có thông tin" |
| `no-graph-data` | Graph không có tài liệu liên quan |
| `partial` | Câu trả lời quá ngắn |
| `llm-unavailable` | Gemini quota / network error (vẫn lưu retrieval stats) |

## Kết quả thực tế (lần chạy gần nhất)

| Q | num_hops | Status | #docs | #chunks |
| - | - | - | - | - |
| Q1 | 0 | llm-unavailable | 2 | 4 |
| Q1 | 1 | llm-unavailable | 8 | 7 |
| Q1 | 2 | llm-unavailable | 9 | 7 |
| Q2 | 0 | llm-unavailable | 2 | 4 |
| Q2 | 1 | llm-unavailable | 6 | 4 |
| Q2 | 2 | llm-unavailable | 7 | 7 |
| Q3 | 0 | llm-unavailable | 1 | 4 |
| Q3 | 1 | llm-unavailable | 5 | 7 |
| Q3 | 2 | llm-unavailable | 7 | 10 |
| Q4 | 0 | llm-unavailable | 1 | 4 |
| Q4 | 1 | llm-unavailable | 5 | 7 |
| Q4 | 2 | llm-unavailable | 7 | 10 |
| Q5 | 0 | llm-unavailable | 2 | 4 |
| Q5 | 1 | llm-unavailable | 8 | 7 |
| Q5 | 2 | llm-unavailable | 9 | 7 |

> **Nhận xét:** Multi-hop expansion **có hiệu quả** — #docs tăng rõ
> rệt từ 1–2 lên 5–9 khi num_hops tăng từ 0 → 2. #chunks cũng tăng
> theo (4 → 7–10). Graph expansion phát hiện các văn bản liên quan qua
> chain CAN_CU.
>
> Gemini API đang hết quota (free tier 20 req/ngày). Khi quota reset,
> rerun `python -m src.run_eval` để có câu trả lời đầy đủ.

## File sinh ra

```
step8_eval/
├── README.md
├── output/
│   ├── eval_results.json   ← structured log
│   └── qa_comparison.md    ← báo cáo Markdown
└── src/
    ├── __init__.py
    ├── eval_suite.py        ← 5 câu hỏi + expected outline
    └── run_eval.py          ← main + markdown renderer
```

## Mở rộng

- **Evaluate retrieval quality** (không cần LLM): so sánh top-K docs
  theo precision@K, MRR, hit-rate bằng cách so sánh với ground-truth
  expected_outline.
- **Evaluate answer quality** (khi có Gemini): dùng câu trả lời +
  expected_outline để LLM-as-judge (Gemini so sánh), hoặc ROUGE-L với
  reference answer.
- **A/B test**: thay đổi prompt, num_hops, top_k, max_chunks → chạy
  eval → so sánh.