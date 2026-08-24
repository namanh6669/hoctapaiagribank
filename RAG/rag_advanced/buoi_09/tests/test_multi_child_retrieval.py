from __future__ import annotations

import unittest

import hierarchical_rag as hr


class MultiChildRetrievalTest(unittest.TestCase):
    def setUp(self):
        hr.clear_query_expansion_cache()

    def config(self) -> hr.HierarchyConfig:
        return hr.HierarchyConfig(
            gemini_embedding_model="gemini-embedding-2",
            gemini_embedding_dim=768,
            gemini_generation_model="gemini-3.5-flash-lite",
            bm25_candidates=20,
            semantic_candidates=20,
            rrf_k=60,
            rrf_bm25_weight=1.0,
            rrf_semantic_weight=1.0,
            rerank_candidates=20,
            final_top_k=5,
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_min_score=0.5,
            reranker_device="auto",
            multi_query_count=3,
            multi_query_max_chars=300,
            multi_query_temperature=0.2,
            multi_query_original_weight=1.5,
            multi_query_variant_weight=1.0,
            multi_query_rrf_k=60,
            per_query_candidates=12,
            parent_max_chars=6000,
            parent_score_child_limit=3,
            parent_rrf_k=60,
            parent_candidates=10,
            final_parent_top_k=3,
            total_context_max_chars=16000,
        )

    def generator(self, generated: list[dict[str, str]] | None = None):
        generated = generated or [
            {"text": "điều kiện vay vốn", "focus": "exact_legal_terms"},
            {"text": "trường hợp không được cho vay", "focus": "missing_aspect"},
        ]

        def _generate(question: str, config: hr.HierarchyConfig):
            return {"queries": generated}

        return _generate

    def hit(self, child_id: str, rank: int, *, text: str | None = None, source: str = "s.pdf", page: int = 1) -> dict:
        return {
            "chunk_id": child_id,
            "text": text or f"text {child_id}",
            "source": source,
            "page_start": page,
            "page_end": page,
            "bm25_rank": rank,
            "semantic_rank": None,
            "fused_rank": rank,
            "rrf_score": 1 / (60 + rank),
            "matched_by": ["bm25"],
        }

    def retriever(self, by_query: dict[str, list[dict]], calls: list[str] | None = None, errors: dict[str, Exception] | None = None):
        def _retrieve(question: str, **kwargs):
            if calls is not None:
                calls.append(question)
            if errors and question in errors:
                raise errors[question]
            return {"question": question, "candidates": by_query.get(question, []), "trace": {"semantic_embedding_call_count": 1}}

        return _retrieve

    def test_mq_rrf_formula_by_hand_and_weights(self):
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([{"text": "Q variant", "focus": "paraphrase"}]),
            hybrid_retriever_fn=self.retriever(
                {
                    "Q root": [self.hit("A", 2)],
                    "Q variant": [self.hit("A", 1)],
                }
            ),
        )
        expected = 1.5 / (60 + 2) + 1.0 / (60 + 1)
        self.assertAlmostEqual(result["children"][0]["multi_query_rrf_score"], expected)
        self.assertEqual(result["children"][0]["support_query_ids"], ["Q0", "Q1"])

    def test_deduplicate_union_and_missing_query_contribution(self):
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([{"text": "Q variant", "focus": "paraphrase"}]),
            hybrid_retriever_fn=self.retriever(
                {
                    "Q root": [self.hit("A", 1), self.hit("B", 2)],
                    "Q variant": [self.hit("A", 3)],
                }
            ),
        )
        by_id = {child["child_id"]: child for child in result["children"]}
        self.assertEqual(set(by_id), {"A", "B"})
        self.assertEqual(by_id["B"]["support_query_count"], 1)
        self.assertEqual(by_id["B"]["support_query_ids"], ["Q0"])
        self.assertAlmostEqual(by_id["B"]["multi_query_rrf_score"], 1.5 / (60 + 2))

    def test_support_query_count_ids_and_per_query_ranks(self):
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([
                {"text": "Q one", "focus": "paraphrase"},
                {"text": "Q two", "focus": "missing_aspect"},
            ]),
            hybrid_retriever_fn=self.retriever(
                {
                    "Q root": [self.hit("A", 4)],
                    "Q one": [self.hit("A", 1)],
                    "Q two": [self.hit("A", 2)],
                }
            ),
        )
        child = result["children"][0]
        self.assertEqual(child["support_query_count"], 3)
        self.assertEqual(child["support_query_ids"], ["Q0", "Q1", "Q2"])
        self.assertEqual(child["per_query_ranks"], {"Q0": 4, "Q1": 1, "Q2": 2})
        self.assertEqual(child["best_query_rank"], 1)

    def test_metadata_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "Metadata mismatch"):
            hr.multi_child_retrieve(
                "Q root",
                config=self.config(),
                query_generator_fn=self.generator([{"text": "Q variant", "focus": "paraphrase"}]),
                hybrid_retriever_fn=self.retriever(
                    {
                        "Q root": [self.hit("A", 1, text="same id old text")],
                        "Q variant": [self.hit("A", 1, text="same id new text")],
                    }
                ),
            )

    def test_deterministic_tie_break(self):
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([{"text": "Q variant", "focus": "paraphrase"}]),
            hybrid_retriever_fn=self.retriever(
                {
                    "Q root": [self.hit("B", 1), self.hit("A", 1)],
                    "Q variant": [],
                }
            ),
        )
        self.assertEqual([child["child_id"] for child in result["children"]], ["A", "B"])
        self.assertEqual([child["multi_query_rank"] for child in result["children"]], [1, 2])

    def test_each_query_calls_hybrid_once_no_reranker_generation(self):
        calls: list[str] = []
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([
                {"text": "Q one", "focus": "paraphrase"},
                {"text": "Q two", "focus": "missing_aspect"},
            ]),
            hybrid_retriever_fn=self.retriever({"Q root": [], "Q one": [], "Q two": []}, calls),
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(calls, ["Q root", "Q one", "Q two"])
        self.assertNotIn("rerank", str(result).casefold())
        self.assertNotIn("answer", str(result).casefold())

    def test_q0_failure_fails_pipeline(self):
        with self.assertRaisesRegex(RuntimeError, "q0 down"):
            hr.multi_child_retrieve(
                "Q root",
                config=self.config(),
                query_generator_fn=self.generator([{"text": "Q variant", "focus": "paraphrase"}]),
                hybrid_retriever_fn=self.retriever({}, errors={"Q root": RuntimeError("q0 down")}),
            )

    def test_generated_query_partial_status(self):
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([
                {"text": "Q one", "focus": "paraphrase"},
                {"text": "Q two", "focus": "missing_aspect"},
            ]),
            hybrid_retriever_fn=self.retriever(
                {"Q root": [self.hit("A", 1)], "Q two": [self.hit("B", 1)]},
                errors={"Q one": RuntimeError("q1 down")},
            ),
        )
        self.assertEqual(result["status"], "partial")
        self.assertIn("Q1", result["query_errors"])
        self.assertEqual(result["trace"]["query_count"]["failed"], 1)
        self.assertEqual(result["trace"]["query_count"]["executed"], 2)

    def test_all_generated_fail_multi_query_partial_status(self):
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([
                {"text": "Q one", "focus": "paraphrase"},
                {"text": "Q two", "focus": "missing_aspect"},
            ]),
            hybrid_retriever_fn=self.retriever(
                {"Q root": [self.hit("A", 1)]},
                errors={"Q one": RuntimeError("q1 down"), "Q two": RuntimeError("q2 down")},
            ),
        )
        self.assertEqual(result["status"], "multi_query_partial")
        self.assertEqual(result["children"][0]["support_query_ids"], ["Q0"])

    def test_trace_counts_latency_schema(self):
        result = hr.multi_child_retrieve(
            "Q root",
            config=self.config(),
            query_generator_fn=self.generator([{"text": "Q variant", "focus": "paraphrase"}]),
            hybrid_retriever_fn=self.retriever(
                {
                    "Q root": [self.hit("A", 1), self.hit("B", 2)],
                    "Q variant": [self.hit("A", 1), self.hit("C", 2)],
                }
            ),
        )
        trace = result["trace"]
        self.assertEqual(trace["query_count"], {"requested": 2, "valid": 2, "executed": 2, "failed": 0})
        self.assertEqual(trace["result_count_by_query"], {"Q0": 2, "Q1": 2})
        self.assertEqual(trace["union_child_count"], 3)
        self.assertEqual(trace["overlap_distribution"], {"1": 2, "2": 1})
        self.assertIn("fusion_latency_ms", trace)
        self.assertIn("query_generation_latency_ms", trace)
        self.assertEqual(trace["semantic_embedding_call_count"], 2)


if __name__ == "__main__":
    unittest.main()
