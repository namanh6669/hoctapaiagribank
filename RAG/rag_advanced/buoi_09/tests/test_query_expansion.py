from __future__ import annotations

import unicodedata
import unittest

import hierarchical_rag as hr


class QueryExpansionTest(unittest.TestCase):
    def setUp(self):
        hr.clear_query_expansion_cache()

    def config(self, *, multi_query_count: int = 3, multi_query_max_chars: int = 120) -> hr.HierarchyConfig:
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
            multi_query_count=multi_query_count,
            multi_query_max_chars=multi_query_max_chars,
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

    def generator(self, payload, counter: dict[str, int] | None = None):
        def _generate(question: str, config: hr.HierarchyConfig):
            if counter is not None:
                counter["calls"] = counter.get("calls", 0) + 1
            return payload

        return _generate

    def test_q0_first_and_preserves_original_content(self):
        question = "  Điều 7 quy định điều kiện vay vốn là gì?  "
        result = hr.expand_query(
            question,
            config=self.config(),
            query_generator_fn=self.generator({"queries": [{"text": "Điều 7 điều kiện vay vốn", "focus": "exact_legal_terms"}]}),
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["original_question"], "Điều 7 quy định điều kiện vay vốn là gì?")
        self.assertEqual(result["queries"][0], {"query_id": "Q0", "text": "Điều 7 quy định điều kiện vay vốn là gì?", "origin": "original", "focus": "original_intent"})

    def test_strict_schema_validation(self):
        result = hr.expand_query(
            "Điều 7 quy định gì?",
            config=self.config(),
            query_generator_fn=self.generator({"queries": [{"text": "Điều 7", "focus": "exact_legal_terms", "answer": "không được có"}]}),
        )
        self.assertEqual(result["status"], "query_generation_unavailable")
        self.assertIn("chỉ được chứa field", result["error"])

    def test_nfc_trim_and_max_length(self):
        decomposed = "Diều kiện vay vốn?"
        result = hr.expand_query(
            "  " + decomposed + "  ",
            config=self.config(multi_query_max_chars=20),
            query_generator_fn=self.generator({"queries": [{"text": "x" * 21, "focus": "paraphrase"}]}),
        )
        self.assertEqual(unicodedata.normalize("NFC", decomposed), result["original_question"])
        self.assertEqual(result["status"], "query_generation_unavailable")
        self.assertIn("vượt 20", result["error"])

    def test_duplicate_removal(self):
        result = hr.expand_query(
            "Điều kiện vay vốn là gì?",
            config=self.config(),
            query_generator_fn=self.generator(
                {
                    "queries": [
                        {"text": "Điều kiện vay vốn là gì?", "focus": "paraphrase"},
                        {"text": "điều kiện vay vốn", "focus": "exact_legal_terms"},
                        {"text": "Điều kiện vay vốn?", "focus": "exact_legal_terms"},
                    ]
                }
            ),
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["dropped_duplicate_count"], 2)
        self.assertEqual([query["query_id"] for query in result["queries"]], ["Q0", "Q1"])
        self.assertEqual(result["queries"][1]["text"], "điều kiện vay vốn")

    def test_legal_reference_preservation(self):
        result = hr.expand_query(
            "Điều 7 quy định gì về vay vốn?",
            config=self.config(),
            query_generator_fn=self.generator({"queries": [{"text": "Điều kiện vay vốn theo Điều 7", "focus": "exact_legal_terms"}]}),
        )
        self.assertEqual(result["status"], "ready")
        self.assertIn("Điều 7", result["queries"][1]["text"])

    def test_rejects_fabricated_article_number(self):
        result = hr.expand_query(
            "Điều 7 quy định gì về vay vốn?",
            config=self.config(),
            query_generator_fn=self.generator(
                {
                    "queries": [
                        {"text": "Điều 99 quy định điều kiện vay vốn", "focus": "exact_legal_terms"},
                        {"text": "Điều kiện vay vốn của khách hàng", "focus": "paraphrase"},
                    ]
                }
            ),
        )
        self.assertEqual(result["status"], "query_generation_unavailable")
        self.assertIn("reference pháp lý", result["error"])

    def test_deterministic_ids_after_validation(self):
        result = hr.expand_query(
            "Điều kiện vay vốn là gì?",
            config=self.config(),
            query_generator_fn=self.generator(
                {
                    "queries": [
                        {"text": "Điều kiện vay vốn là gì?", "focus": "paraphrase"},
                        {"text": "điều kiện vay vốn", "focus": "exact_legal_terms"},
                        {"text": "nhu cầu vốn được vay", "focus": "missing_aspect"},
                    ]
                }
            ),
        )
        self.assertEqual([query["query_id"] for query in result["queries"]], ["Q0", "Q1", "Q2"])
        self.assertEqual([query["origin"] for query in result["queries"]], ["original", "generated", "generated"])

    def test_one_generator_call(self):
        counter: dict[str, int] = {}
        result = hr.expand_query(
            "Điều kiện vay vốn là gì?",
            config=self.config(),
            query_generator_fn=self.generator({"queries": [{"text": "điều kiện vay vốn", "focus": "exact_legal_terms"}]}, counter),
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(counter["calls"], 1)

    def test_cache_hit_does_not_call_generator_again(self):
        counter: dict[str, int] = {}
        generator = self.generator({"queries": [{"text": "điều kiện vay vốn", "focus": "exact_legal_terms"}]}, counter)
        first = hr.expand_query("Điều kiện vay vốn là gì?", config=self.config(), query_generator_fn=generator)
        second = hr.expand_query("Điều kiện vay vốn là gì?", config=self.config(), query_generator_fn=generator)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(counter["calls"], 1)

    def test_api_error_returns_explicit_status(self):
        def failing_generator(question: str, config: hr.HierarchyConfig):
            raise RuntimeError("fake API failed")

        result = hr.expand_query("Điều kiện vay vốn là gì?", config=self.config(), query_generator_fn=failing_generator)
        self.assertEqual(result["status"], "query_generation_unavailable")
        self.assertIn("fake API failed", result["error"])
        self.assertEqual(result["queries"], [])


if __name__ == "__main__":
    unittest.main()
