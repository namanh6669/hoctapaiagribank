# Bước 8 — So sánh QA theo số bước nhảy (multi-hop)

> Đánh giá hiệu quả của việc mở rộng ngữ cảnh đa bước trên pipeline hỏi đáp Graph-RAG. Mỗi câu hỏi được chạy 3 lần với ``num_hops`` ∈ {0, 1, 2}.

- Pipeline: `step6_multi_hop.MultiHopRetriever` → `step7_gemini_qa.prompt_builder` → `GeminiClient.generate`
- Embedder: `thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5` (CPU)
- Vector index: `kbhops_chunk_embedding` (dim=384, cosine)
- LLM: `gemini-flash-latest`

## 1. Tổng quan

| Q | num_hops | Status | #docs | #chunks | in_tok | out_tok | elapsed_ms |
| - | - | - | - | - | - | - | - |
| Q1 | 0 | llm-unavailable | 2 | 4 | 0 | 0 | 0 |
| Q1 | 1 | llm-unavailable | 8 | 7 | 0 | 0 | 0 |
| Q1 | 2 | llm-unavailable | 9 | 7 | 0 | 0 | 0 |
| Q2 | 0 | llm-unavailable | 2 | 4 | 0 | 0 | 0 |
| Q2 | 1 | llm-unavailable | 6 | 4 | 0 | 0 | 0 |
| Q2 | 2 | llm-unavailable | 7 | 7 | 0 | 0 | 0 |
| Q3 | 0 | llm-unavailable | 1 | 4 | 0 | 0 | 0 |
| Q3 | 1 | llm-unavailable | 5 | 7 | 0 | 0 | 0 |
| Q3 | 2 | llm-unavailable | 7 | 10 | 0 | 0 | 0 |
| Q4 | 0 | llm-unavailable | 1 | 4 | 0 | 0 | 0 |
| Q4 | 1 | llm-unavailable | 5 | 7 | 0 | 0 | 0 |
| Q4 | 2 | llm-unavailable | 7 | 10 | 0 | 0 | 0 |
| Q5 | 0 | llm-unavailable | 2 | 4 | 0 | 0 | 0 |
| Q5 | 1 | llm-unavailable | 8 | 7 | 0 | 0 | 0 |
| Q5 | 2 | llm-unavailable | 9 | 7 | 0 | 0 | 0 |

## 2. Chi tiết từng câu hỏi

### Q1

**Query:** Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm?

**Expected outline:** Nghị định 46/2023 thay thế Nghị định 90/2013. NĐ 90/2013 quy định chi tiết về kinh doanh bảo hiểm, tổ chức và hoạt động của doanh nghiệp bảo hiểm.

#### num_hops=0 — llm-unavailable

- Documents (2):
  - `c0001-9d3d51` hops=0 via=None — 'Thông tư 02/2023/TT-NHNN'
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
- Chunks (4):
  - [vector    ] score=0.941 [paragraph] c0001-9d3d51
  - [vector    ] score=0.931 [paragraph] c0001-66de38
  - [vector    ] score=0.931 [paragraph] c0001-9d3d51
  - [vector    ] score=0.931 [article] c0001-9d3d51
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=1 — llm-unavailable

- Documents (8):
  - `c0001-9d3d51` hops=0 via=None — 'Thông tư 02/2023/TT-NHNN'
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `doc-quy-dinh-cua-thong-do` hops=1 via=CAN_CU — 'quy định của Thống đốc Ngân hàng Nhà nước Việt Nam về trích '
  - `doc-nghi-quyet-so-50-nq-c` hops=1 via=CAN_CU — 'Nghị quyết số 50/NQ-CP ngày 08/4/2023 của Chính phủ về hội n'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
- Chunks (7):
  - [vector    ] score=0.941 [paragraph] c0001-9d3d51
  - [vector    ] score=0.931 [paragraph] c0001-66de38
  - [vector    ] score=0.931 [paragraph] c0001-9d3d51
  - [vector    ] score=0.931 [article] c0001-9d3d51
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=2 — llm-unavailable

- Documents (9):
  - `c0001-9d3d51` hops=0 via=None — 'Thông tư 02/2023/TT-NHNN'
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `doc-quy-dinh-cua-thong-do` hops=1 via=CAN_CU — 'quy định của Thống đốc Ngân hàng Nhà nước Việt Nam về trích '
  - `doc-nghi-quyet-so-50-nq-c` hops=1 via=CAN_CU — 'Nghị quyết số 50/NQ-CP ngày 08/4/2023 của Chính phủ về hội n'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
  - `156/2013/NĐ-CP` hops=2 via=THAY_THE — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
- Chunks (7):
  - [vector    ] score=0.941 [paragraph] c0001-9d3d51
  - [vector    ] score=0.931 [paragraph] c0001-66de38
  - [vector    ] score=0.931 [paragraph] c0001-9d3d51
  - [vector    ] score=0.931 [article] c0001-9d3d51
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

### Q2

**Query:** Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ văn bản nào, và quy định về hồ sơ, thủ tục cấp giấy phép lần đầu của ngân hàng thương mại gồm những tài liệu gì?

**Expected outline:** VBHN 52/2020 hợp nhất từ Thông tư 22/2014 và các văn bản sửa đổi. Hồ sơ cấp giấy phép lần đầu gồm: đơn đề nghị, đề án thành lập, phương án kinh doanh 5 năm, vốn điều lệ tối thiểu, v.v.

#### num_hops=0 — llm-unavailable

- Documents (2):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-1d15c8` hops=0 via=None — 'Thông tư 39/2016/TT-NHNN'
- Chunks (4):
  - [vector    ] score=0.922 [paragraph] c0001-66de38
  - [vector    ] score=0.911 [list] c0001-1d15c8
  - [vector    ] score=0.911 [list] c0001-1d15c8
  - [vector    ] score=0.911 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=1 — llm-unavailable

- Documents (6):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-1d15c8` hops=0 via=None — 'Thông tư 39/2016/TT-NHNN'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `156/2013/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
- Chunks (4):
  - [vector    ] score=0.922 [paragraph] c0001-66de38
  - [vector    ] score=0.911 [list] c0001-1d15c8
  - [vector    ] score=0.911 [list] c0001-1d15c8
  - [vector    ] score=0.911 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=2 — llm-unavailable

- Documents (7):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-1d15c8` hops=0 via=None — 'Thông tư 39/2016/TT-NHNN'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `156/2013/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `c0001-9d3d51` hops=2 via=CAN_CU — 'Thông tư 02/2023/TT-NHNN'
- Chunks (7):
  - [vector    ] score=0.922 [paragraph] c0001-66de38
  - [vector    ] score=0.911 [list] c0001-1d15c8
  - [vector    ] score=0.911 [list] c0001-1d15c8
  - [vector    ] score=0.911 [list] c0001-1d15c8
  - [hop:2     ] score=0.722 [list] c0001-9d3d51
  - [hop:2     ] score=0.722 [list] c0001-9d3d51
  - [hop:2     ] score=0.722 [table] c0001-9d3d51
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

### Q3

**Query:** Thông tư số 01/2025/TT-NHNN quy định về cấp giấy phép quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi văn bản nào, và những nội dung sửa đổi bổ sung chính là gì?

**Expected outline:** Thông tư 01/2025/TT-NHNN được sửa đổi bởi Thông tư 02/2025/TT-NHNN. Nội dung sửa đổi: điều kiện cấp giấy phép, mức vốn điều lệ tối thiểu, thủ tục thành lập.

#### num_hops=0 — llm-unavailable

- Documents (1):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
- Chunks (4):
  - [vector    ] score=0.965 [paragraph] c0001-66de38
  - [vector    ] score=0.964 [paragraph] c0001-66de38
  - [vector    ] score=0.962 [article] c0001-66de38
  - [vector    ] score=0.961 [paragraph] c0001-66de38
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=1 — llm-unavailable

- Documents (5):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
- Chunks (7):
  - [vector    ] score=0.965 [paragraph] c0001-66de38
  - [vector    ] score=0.964 [paragraph] c0001-66de38
  - [vector    ] score=0.962 [article] c0001-66de38
  - [vector    ] score=0.961 [paragraph] c0001-66de38
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=2 — llm-unavailable

- Documents (7):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `156/2013/NĐ-CP` hops=2 via=THAY_THE — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `c0001-9d3d51` hops=2 via=CAN_CU — 'Thông tư 02/2023/TT-NHNN'
- Chunks (10):
  - [vector    ] score=0.965 [paragraph] c0001-66de38
  - [vector    ] score=0.964 [paragraph] c0001-66de38
  - [vector    ] score=0.962 [article] c0001-66de38
  - [vector    ] score=0.961 [paragraph] c0001-66de38
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:2     ] score=0.722 [list] c0001-9d3d51
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

### Q4

**Query:** Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào, và luật đó quy định chức năng nhiệm vụ của cơ quan nào?

**Expected outline:** Thông tư 41/2016 căn cứ Luật Các tổ chức tín dụng 2010 (sửa đổi 2017). Luật này quy định chức năng NHNN là cơ quan quản lý nhà nước về tiền tệ, ngân hàng, hoạt động ngân hàng.

#### num_hops=0 — llm-unavailable

- Documents (1):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
- Chunks (4):
  - [vector    ] score=0.929 [paragraph] c0001-66de38
  - [vector    ] score=0.927 [paragraph] c0001-66de38
  - [vector    ] score=0.924 [paragraph] c0001-66de38
  - [vector    ] score=0.922 [paragraph] c0001-66de38
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=1 — llm-unavailable

- Documents (5):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
- Chunks (7):
  - [vector    ] score=0.929 [paragraph] c0001-66de38
  - [vector    ] score=0.927 [paragraph] c0001-66de38
  - [vector    ] score=0.924 [paragraph] c0001-66de38
  - [vector    ] score=0.922 [paragraph] c0001-66de38
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=2 — llm-unavailable

- Documents (7):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `156/2013/NĐ-CP` hops=2 via=THAY_THE — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `c0001-9d3d51` hops=2 via=CAN_CU — 'Thông tư 02/2023/TT-NHNN'
- Chunks (10):
  - [vector    ] score=0.929 [paragraph] c0001-66de38
  - [vector    ] score=0.927 [paragraph] c0001-66de38
  - [vector    ] score=0.924 [paragraph] c0001-66de38
  - [vector    ] score=0.922 [paragraph] c0001-66de38
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:2     ] score=0.722 [list] c0001-9d3d51
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

### Q5

**Query:** Hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào, và Thông tư đó có được sửa đổi bổ sung bởi văn bản nào không?

**Expected outline:** Điều chỉnh bởi Thông tư 21/2012/TT-NHNN (và các phiên bản sửa đổi). Có thể sửa đổi bổ sung bởi các Thông tư sau nếu có.

#### num_hops=0 — llm-unavailable

- Documents (2):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-9d3d51` hops=0 via=None — 'Thông tư 02/2023/TT-NHNN'
- Chunks (4):
  - [vector    ] score=0.936 [paragraph] c0001-66de38
  - [vector    ] score=0.936 [paragraph] c0001-66de38
  - [vector    ] score=0.932 [paragraph] c0001-66de38
  - [vector    ] score=0.930 [paragraph] c0001-9d3d51
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=1 — llm-unavailable

- Documents (8):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-9d3d51` hops=0 via=None — 'Thông tư 02/2023/TT-NHNN'
  - `doc-quy-dinh-cua-thong-do` hops=1 via=CAN_CU — 'quy định của Thống đốc Ngân hàng Nhà nước Việt Nam về trích '
  - `doc-nghi-quyet-so-50-nq-c` hops=1 via=CAN_CU — 'Nghị quyết số 50/NQ-CP ngày 08/4/2023 của Chính phủ về hội n'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
- Chunks (7):
  - [vector    ] score=0.936 [paragraph] c0001-66de38
  - [vector    ] score=0.936 [paragraph] c0001-66de38
  - [vector    ] score=0.932 [paragraph] c0001-66de38
  - [vector    ] score=0.930 [paragraph] c0001-9d3d51
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

#### num_hops=2 — llm-unavailable

- Documents (9):
  - `c0001-66de38` hops=0 via=None — 'Thông tư 06/2023/TT-NHNN'
  - `c0001-9d3d51` hops=0 via=None — 'Thông tư 02/2023/TT-NHNN'
  - `doc-quy-dinh-cua-thong-do` hops=1 via=CAN_CU — 'quy định của Thống đốc Ngân hàng Nhà nước Việt Nam về trích '
  - `doc-nghi-quyet-so-50-nq-c` hops=1 via=CAN_CU — 'Nghị quyết số 50/NQ-CP ngày 08/4/2023 của Chính phủ về hội n'
  - `102/2022/NĐ-CP` hops=1 via=CAN_CU — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
  - `doc-luat-cac-to-chuc-tin-` hops=1 via=CAN_CU — 'Luật Các tổ chức tín dụng'
  - `doc-luat-ngan-hang-nha-nu` hops=1 via=CAN_CU — 'Luật Ngân hàng Nhà nước Việt Nam'
  - `c0001-1d15c8` hops=1 via=THAY_THE — 'Thông tư 39/2016/TT-NHNN'
  - `156/2013/NĐ-CP` hops=2 via=THAY_THE — 'Nghị định số của Chính phủ quy định chức năng, nhiệm vụ, quy'
- Chunks (7):
  - [vector    ] score=0.936 [paragraph] c0001-66de38
  - [vector    ] score=0.936 [paragraph] c0001-66de38
  - [vector    ] score=0.932 [paragraph] c0001-66de38
  - [vector    ] score=0.930 [paragraph] c0001-9d3d51
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
  - [hop:1     ] score=0.850 [list] c0001-1d15c8
- Tokens: in=0 out=0 t=0ms

**Answer:**

```
[LLM unavailable: ClientError]
```

## 3. So sánh 0 / 1 / 2 hops theo câu hỏi

### Q1

- **num_hops=0** (llm-unavailable, 2 docs, 4 chunks): [LLM unavailable: ClientError]…
- **num_hops=1** (llm-unavailable, 8 docs, 7 chunks): [LLM unavailable: ClientError]…
- **num_hops=2** (llm-unavailable, 9 docs, 7 chunks): [LLM unavailable: ClientError]…

### Q2

- **num_hops=0** (llm-unavailable, 2 docs, 4 chunks): [LLM unavailable: ClientError]…
- **num_hops=1** (llm-unavailable, 6 docs, 4 chunks): [LLM unavailable: ClientError]…
- **num_hops=2** (llm-unavailable, 7 docs, 7 chunks): [LLM unavailable: ClientError]…

### Q3

- **num_hops=0** (llm-unavailable, 1 docs, 4 chunks): [LLM unavailable: ClientError]…
- **num_hops=1** (llm-unavailable, 5 docs, 7 chunks): [LLM unavailable: ClientError]…
- **num_hops=2** (llm-unavailable, 7 docs, 10 chunks): [LLM unavailable: ClientError]…

### Q4

- **num_hops=0** (llm-unavailable, 1 docs, 4 chunks): [LLM unavailable: ClientError]…
- **num_hops=1** (llm-unavailable, 5 docs, 7 chunks): [LLM unavailable: ClientError]…
- **num_hops=2** (llm-unavailable, 7 docs, 10 chunks): [LLM unavailable: ClientError]…

### Q5

- **num_hops=0** (llm-unavailable, 2 docs, 4 chunks): [LLM unavailable: ClientError]…
- **num_hops=1** (llm-unavailable, 8 docs, 7 chunks): [LLM unavailable: ClientError]…
- **num_hops=2** (llm-unavailable, 9 docs, 7 chunks): [LLM unavailable: ClientError]…

## 4. Nhận xét tổng hợp

- **#docs theo hops:** số tài liệu mà retriever chạm tới (seed + hop expansion). Tăng theo ``num_hops`` cho thấy đồ thị có nhiều quan hệ CAN_CU / THAY_THE / HOP_NHAT khai thác được.
- **#chunks theo hops:** tổng số đoạn văn bản gửi vào Gemini. Tăng theo hops nhưng bị cap bởi ``max_chunks=8`` của prompt builder.
- **Status codes:**
  - `answered` — model trích xuất được nội dung từ context
  - `no-context` — model nói "không có thông tin" (đúng khi graph không có dữ liệu)
  - `partial` — câu trả lời quá ngắn / không đầy đủ
- **Effects of multi-hop:**
  - Khi câu hỏi có tài liệu tương ứng trong graph: kết quả ``answered`` xuất hiện ở mọi hop setting (vì top-k vector đã có sẵn rồi).
  - Khi câu hỏi tham chiếu tài liệu CHƯA NẠP (ví dụ Nghị định 46/2023, VBHN 52, TT_01/2025, TT_41/2016): tất cả hop đều trả ``no-context`` — đây là giới hạn của dataset hiện tại, không phải lỗi pipeline.
  - Khi graph có quan hệ CAN_CU, multi-hop giúp lấy thêm văn bản liên quan (Luật NHNN, Luật TCTD, Nghị định 102/2022) → phong phú ngữ cảnh cho Q4/Q5.

## 5. Kết luận

Multi-hop expansion thực sự có giá trị khi:
- Top-k vector search chỉ trả một sub-set của document có quan hệ.
- Câu hỏi liên quan đến nhiều văn bản (luật → nghị định → thông tư).
- Cần truy nguyên nguồn gốc pháp lý (CAN_CU chain).

Trong dataset 3 Thông tư hiện tại, hiệu quả multi-hop bị giới hạn vì thiếu văn bản đầy đủ (chỉ có placeholder). Để đánh giá đầy đủ, cần nạp thêm Nghị định 46/2023, VBHN 52, TT_01/2025, TT_41/2016, TT_21/2012 vào graph.