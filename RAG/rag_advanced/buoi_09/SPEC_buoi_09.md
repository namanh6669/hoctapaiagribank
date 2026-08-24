# SPEC Buổi 09 - Hierarchical + Multi-Query RAG

> Trạng thái: specification và skeleton độc lập cho Buổi 09. Bước 02 chỉ tạo cấu trúc project, snapshot baseline Buổi 08 và mô tả hợp đồng kỹ thuật. Chưa triển khai hierarchy builder, multi-query retrieval, parent aggregation, rerank parent, generation flow hoặc UI.

## 1. Workspace và phạm vi ghi

- Chỉ ghi trong `rag_advanced/buoi_09/`.
- Không sửa source, test, output, `.env`, storage, cache hoặc report của Buổi 05, Buổi 06, Buổi 07, Buổi 08.
- Không sao chép `.env` thật, API key, storage, cache, report hoặc `__pycache__` từ Buổi 08.
- Không tạo/reset Chroma collection trong bước skeleton.
- Không gọi Gemini, không tải reranker, không build hierarchy store khi import module.
- `rag.py` và `advanced_rag.py` là snapshot baseline từ Buổi 08, chạy độc lập theo `Path(__file__).resolve()` trong Buổi 09 và không import runtime từ directory Buổi 08.

## 2. Mục tiêu và khác biệt Buổi 08/09

### Buổi 08

Buổi 08 cung cấp baseline flat Advanced RAG:

- BM25 lexical retrieval.
- Semantic retrieval từ Chroma/Gemini embedding baseline.
- Reciprocal Rank Fusion (RRF) giữa BM25 và semantic.
- Cross-encoder reranker trên child/fused candidates.
- Grounded answer và citation map theo metadata thật.
- CLI/UI comparison và offline evaluator.

### Buổi 09

Buổi 09 mở rộng baseline theo hai trục:

1. **Multi-query retrieval**: giữ query gốc `Q0`, tạo các variants có kiểm soát, chạy retrieval per-query rồi fuse cross-query bằng RRF có trọng số để tăng recall mà không làm mất anchor pháp lý.
2. **Parent-aware retrieval**: map child chunks sang parent documents, aggregate score từ child hits, rerank parent candidates và build context theo budget để giảm fragmentation của câu trả lời pháp lý.

Buổi 09 không thay thế baseline flat; phải so sánh bốn mode để người học thấy trade-off giữa recall, precision, latency và context size.

## 3. Pipeline mục tiêu

```text
Q0 original question
   │
   ├── QueryVariant[0] = original, weight = MULTI_QUERY_ORIGINAL_WEIGHT
   └── QueryVariant[1..N] = variants, weight = MULTI_QUERY_VARIANT_WEIGHT
            │
            ▼
   per-query hybrid retrieval
   (BM25 + semantic → per-query RRF child hits)
            │
            ▼
   cross-query RRF over child hits
   preserve query_id, variant text, child ranks, matched_by
            │
            ▼
   child-to-parent resolution
   via hierarchy registry, text heading fallback, source/order fallback
            │
            ▼
   parent aggregation
   combine top child evidence per parent, cap child count and parent chars
            │
            ▼
   optional parent rerank
   query/parent pairs through injected cross-encoder scorer
            │
            ▼
   final context budget
   TOTAL_CONTEXT_MAX_CHARS, source/page/chunk citations
            │
            ▼
   grounded generation
```

## 4. Retrieval modes

Buổi 09 phải hỗ trợ bốn mode, cùng schema output chung để compare:

1. `single_flat`
   - Chỉ dùng Q0.
   - Chạy flat baseline child retrieval/rerank từ Buổi 08.
   - Không parent aggregation.

2. `multi_flat`
   - Dùng Q0 + variants.
   - Chạy per-query flat hybrid retrieval.
   - Fuse child candidates bằng cross-query RRF.
   - Không parent aggregation.

3. `single_parent`
   - Chỉ dùng Q0.
   - Chạy child retrieval.
   - Map child-to-parent, aggregate parent candidates, optional parent rerank.

4. `multi_parent`
   - Dùng Q0 + variants.
   - Per-query hybrid retrieval.
   - Cross-query RRF over child hits.
   - Map child-to-parent, aggregate parent candidates, optional parent rerank.
   - Đây là mode advanced mặc định sau khi triển khai đầy đủ, nhưng skeleton chưa bật.

## 5. QueryVariant schema và validation

```json
{
  "query_id": "q0",
  "text": "Điều 7 quy định trách nhiệm gì?",
  "kind": "original",
  "weight": 1.5,
  "anchors": {
    "articles": ["7"],
    "clauses": [],
    "points": [],
    "sources": []
  },
  "warnings": []
}
```

Required fields:

- `query_id`: string không rỗng, unique trong một request.
- `text`: string không rỗng, length `<= MULTI_QUERY_MAX_CHARS` với variants; Q0 được validate bằng limit query baseline.
- `kind`: một trong `original`, `variant`.
- `weight`: số hữu hạn `> 0`.
- `anchors`: object chứa legal anchors nhận diện từ Q0 và/hoặc variant.
- `warnings`: list string an toàn, không chứa secret.

Validation rules:

- Query gốc Q0 luôn được giữ, không rewrite inplace.
- Variants không được làm mất anchor pháp lý quan trọng nếu Q0 có `Điều`, `Khoản`, `điểm` cụ thể; nếu thiếu anchor phải gắn warning `variant_anchor_loss`.
- Tổng số variants tối đa là `MULTI_QUERY_COUNT`; nếu generator trả nhiều hơn phải truncate có warning.
- Không gọi LLM để tạo variants trong unit test; dùng dependency injection.
- Không tạo variants rỗng, trùng exact text sau normalize hoặc quá dài.

## 6. Hierarchy registry schema

Hierarchy registry là dữ liệu đọc/build từ chunk source, lưu nội bộ Buổi 09 khi bước sau được phép triển khai. Skeleton chưa build registry.

```json
{
  "schema_version": "buoi_09_hierarchy_v1",
  "created_from": {
    "input_path": "rag_foundation/buoi_05/output/chunks/chunks.json",
    "strategy": "hierarchical",
    "source_hashes": {
      "chunks.json": "sha256:..."
    }
  },
  "parents": {
    "TT_39_2016_NHNN.pdf::article::7": {
      "parent_id": "TT_39_2016_NHNN.pdf::article::7",
      "source": "TT_39_2016_NHNN.pdf",
      "level": "article",
      "title": "Điều 7. ...",
      "page_start": 1,
      "page_end": 2,
      "child_chunk_ids": ["..."],
      "text_hash": "sha256:...",
      "metadata": {
        "chapter": "I",
        "article": "7"
      }
    }
  },
  "child_to_parent": {
    "TT_39_2016_NHNN::hierarchical::0007": "TT_39_2016_NHNN.pdf::article::7"
  },
  "warnings": []
}
```

Required top-level fields:

- `schema_version`
- `created_from`
- `parents`
- `child_to_parent`
- `warnings`

Rules:

- Registry must be deterministic for same input bytes.
- Parent IDs must include `source` and structural key to avoid cross-document collisions.
- Missing/ambiguous mapping must not be silently dropped.
- Do not assume every child has `parent_id` in source metadata.

## 7. ParentDocument schema

```json
{
  "parent_id": "TT_39_2016_NHNN.pdf::article::7",
  "source": "TT_39_2016_NHNN.pdf",
  "level": "article",
  "title": "Điều 7. ...",
  "text": "...",
  "page_start": 1,
  "page_end": 2,
  "child_chunk_ids": ["..."],
  "structure": {
    "chapter": "I",
    "article": "7"
  },
  "warnings": []
}
```

Validation:

- `parent_id`, `source`, `level`, `title`, `text`: string hợp lệ.
- `page_start`, `page_end`: integer, `page_start <= page_end`.
- `child_chunk_ids`: list string non-empty khi parent được tạo từ child chunks.
- `text` length should be capped or compressed according to `PARENT_MAX_CHARS` before generation context.
- `warnings` records truncation, ambiguous headings, or source-order fallback.

## 8. MultiQueryChildHit schema

```json
{
  "chunk_id": "TT_39_2016_NHNN::hierarchical::0007",
  "source": "TT_39_2016_NHNN.pdf",
  "page_start": 1,
  "page_end": 2,
  "text": "...",
  "query_id": "q0",
  "query_kind": "original",
  "query_weight": 1.5,
  "bm25_rank": 1,
  "bm25_score": 12.3,
  "semantic_rank": 3,
  "semantic_distance": 0.22,
  "per_query_fused_rank": 1,
  "per_query_rrf_score": 0.032,
  "cross_query_rank": 2,
  "cross_query_rrf_score": 0.041,
  "matched_by": ["bm25", "semantic"],
  "warnings": []
}
```

Required fields before cross-query fusion:

- `chunk_id`, `source`, `page_start`, `page_end`, `text`
- `query_id`, `query_kind`, `query_weight`
- At least one branch rank: `bm25_rank` or `semantic_rank` or `per_query_fused_rank`.

After cross-query fusion:

- `cross_query_rank`
- `cross_query_rrf_score`

## 9. ParentCandidate schema

```json
{
  "parent_id": "TT_39_2016_NHNN.pdf::article::7",
  "parent": {"...": "ParentDocument"},
  "child_hits": [{"...": "MultiQueryChildHit"}],
  "aggregation": {
    "method": "top_child_rrf_sum",
    "parent_score": 0.087,
    "child_limit": 3,
    "rrf_k": 60
  },
  "parent_rank": 1,
  "rerank_rank": 1,
  "rerank_score": 0.82,
  "accepted": true,
  "warnings": []
}
```

Validation:

- ParentCandidate must retain all child hit metadata needed for citations.
- Parent score must be deterministic and finite.
- If parent text is truncated, warning `parent_context_truncated` is required.
- If parent mapping is ambiguous, warning `ambiguous_parent_resolution` is required.

## 10. Hierarchy resolution và ambiguous warning

Resolution order proposed for Buổi 09 implementation:

1. Use explicit `structure` metadata if available and internally consistent.
2. Detect Markdown/legal headings at line start only, e.g. `^\s*#{0,6}\s*Điều\s+\d+` and `^\s*#{0,6}\s*Chương\s+...`.
3. Use source-local chunk order to attach continuation chunks to the nearest prior heading parent.
4. For amendment texts, distinguish top-level amended document headings from cited inner articles using line position, quote context, numbered amendment clauses and source-order context.
5. If multiple parent candidates remain plausible, choose deterministic fallback but attach warning.

Warnings:

- `missing_structure_metadata`: no usable `structure` object.
- `heading_detected_from_text`: parent inferred from text heading.
- `ambiguous_article_reference`: text contains article citation that could be mistaken for heading.
- `ambiguous_parent_resolution`: more than one parent candidate plausible.
- `orphan_child`: no parent resolved.

## 11. Cross-query RRF

Let `rank(q, c)` be the rank of child chunk `c` in query variant `q`. Let `w(q)` be variant weight.

```text
cross_query_rrf_score(c) = Σ_q w(q) / (MULTI_QUERY_RRF_K + rank(q, c))
```

Rules:

- Q0 original default weight: `MULTI_QUERY_ORIGINAL_WEIGHT=1.5`.
- Variant default weight: `MULTI_QUERY_VARIANT_WEIGHT=1.0`.
- Missing candidate in a query contributes 0.
- Tie-breaker: score desc, best Q0 rank asc, best any-query rank asc, source asc, page_start asc, chunk_id asc.
- Do not compare raw BM25 score and semantic distance directly across branches.
- Keep per-query trace so latency/cost can be explained.

## 12. Parent aggregation

For a parent `p` with child hits `H(p)`, select top `PARENT_SCORE_CHILD_LIMIT` child hits by cross-query rank/score, then aggregate:

```text
parent_score(p) = Σ_i 1 / (PARENT_RRF_K + child_cross_query_rank_i)
```

Alternative scoring may include child score sum, but must be tested and recorded in `aggregation.method`.

Rules:

- `PARENT_SCORE_CHILD_LIMIT=3` by default.
- `PARENT_CANDIDATES=10` limits parent candidates before optional parent rerank.
- `FINAL_PARENT_TOP_K=3` limits accepted parents for context/generation.
- A parent with one very strong child should not be drowned by a parent with many weak child hits; hence child limit is required.
- Parent aggregation must preserve evidence child IDs for citation.

## 13. Context budget và citation contract

Budgets:

- `PARENT_MAX_CHARS=6000`: cap per parent context before final assembly.
- `TOTAL_CONTEXT_MAX_CHARS=16000`: cap total context sent to generation.
- If budget is exceeded, trim lower-ranked parents/child spans first and attach warning.

Citation contract:

- Generated answer may only cite evidence IDs assigned by Buổi 09, e.g. `[P1]`, `[P2]`, or child evidence `[E1]` depending on final design.
- Citation mapping must resolve to real metadata: `source`, `page_start`, `page_end`, `parent_id`, and contributing `chunk_id` list.
- Never let LLM invent source, page, article, parent ID or chunk ID.
- Invalid citation labels must be removed and reported as warning.
- If answer claims a legal article/khoản not present in accepted evidence metadata/text, output should warn or refuse based on confidence gate.

## 14. Status/failure contract

Every public query/status function must return an explicit status:

- `ready`: local resources needed for requested mode exist.
- `missing_semantic_index`: Chroma collection missing or incompatible.
- `missing_hierarchy_registry`: parent-aware mode requested but registry unavailable.
- `reranker_unavailable`: reranker import/load/score failed; no silent fallback pretending rerank occurred.
- `insufficient_evidence`: retrieval ran but no evidence passed gate.
- `retrieval_only`: retrieval succeeded but generation failed or was disabled.
- `answered`: generation succeeded with mapped citations.

Failure rules:

- No API key must fail only when an API call is actually needed.
- Status/import must not create storage or call external services.
- Errors must redact API keys and stay bounded in length.
- Latency trace must include query count, embedding call count, retrieval count, rerank count and generation call count once implemented.

## 15. Testability và dependency injection

Required injection points:

- Query variant generator.
- Query embedder.
- Semantic retriever/collection adapter.
- BM25 retriever.
- Cross-query fusion input fixtures.
- Hierarchy registry loader/resolver.
- Parent rerank scorer.
- Answer generator.

Unit test rules:

- Tests must not call Gemini, Internet, real reranker download or persistent production Chroma.
- Reranker tests use deterministic fake scorer.
- Semantic tests use fake embedder and temporary Chroma/storage only when needed.
- Hierarchy tests use `tests/fixtures/hierarchical_sample.json` and minimal in-memory records.
- Import tests assert no side effects: no collection create, no model load, no hierarchy build.

## 16. Evaluation metrics và acceptance criteria

Metrics for Buổi 09 evaluator:

- Child Recall@K by `relevant_chunk_ids`.
- Parent Recall@K by `relevant_parent_ids`.
- MRR@K and nDCG@K for child and parent modes.
- Citation coverage: answer citations map to accepted evidence.
- Insufficient evidence rate for out-of-scope questions.
- Latency mean/p50/p95 by mode.
- Query expansion overhead: number of variants, embedding calls, retrieval calls, rerank pairs.
- Context budget utilization and truncation count.

Acceptance criteria before declaring Buổi 09 complete:

- All four modes return deterministic schema under fake dependencies.
- `single_flat` remains compatible with Buổi 08 baseline behavior.
- `multi_flat` preserves Q0 and legal anchors.
- `single_parent` and `multi_parent` expose parent/child trace and warnings.
- No mode silently downgrades from parent-aware to flat without status/warning.
- Unit tests compile/run offline with no API/network/model download.
- README documents commands and resource warnings.
- Evaluation reports warn when labels still have `needs_human_review=true`.

## 17. Bước 02 completion note

Bước 02 chỉ tạo skeleton và spec. Các file placeholder `hierarchical_rag.py`, `evaluate.py`, `app.py` phải import được, có docstring/TODO an toàn và chưa tuyên bố tính năng đã chạy.
