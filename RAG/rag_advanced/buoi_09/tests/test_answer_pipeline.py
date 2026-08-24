from __future__ import annotations

import unittest

import hierarchical_rag as hr


class AnswerPipelineTest(unittest.TestCase):
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
            final_top_k=2,
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
            parent_score_child_limit=2,
            parent_rrf_k=60,
            parent_candidates=3,
            final_parent_top_k=2,
            total_context_max_chars=16000,
        )

    def parent_candidate(self, parent_id: str, rank: int, text: str) -> dict:
        return {
            "parent_id": parent_id,
            "source": "s.pdf",
            "page_start": 1,
            "page_end": 2,
            "structural_path": {"article": "7"},
            "text": text,
            "parent_rrf_score": 1 / (60 + rank),
            "parent_rank": rank,
            "anchor_child_id": f"c{rank}",
            "scoring_child_ids": [f"c{rank}"],
            "supporting_child_ids": [f"c{rank}"],
            "support_query_ids": ["Q0"],
            "best_child_rank": rank,
            "ambiguous": False,
            "warnings": [],
        }

    def child_hit(self, child_id: str, rank: int, text: str = "child text") -> dict:
        return {
            "child_id": child_id,
            "text": text,
            "source": "s.pdf",
            "page_start": 1,
            "page_end": 1,
            "multi_query_rrf_score": 1 / (60 + rank),
            "multi_query_rank": rank,
            "support_query_count": 1,
            "support_query_ids": ["Q0"],
            "per_query_ranks": {"Q0": rank},
            "per_query_trace": {},
        }

    def test_reranker_pair_uses_q0_and_parent_text(self):
        seen = {}

        def scorer(question, parents, config):
            seen["question"] = question
            seen["texts"] = [parent["text"] for parent in parents]
            return [1.0]

        hr.rerank_parent_candidates("original q", [self.parent_candidate("p1", 1, "parent body")], config=self.config(), scorer=scorer)
        self.assertEqual(seen, {"question": "original q", "texts": ["parent body"]})

    def test_generated_query_not_used_for_rerank_or_generation(self):
        prompts = []
        questions = []

        def child_retrieval(question, **kwargs):
            return {"status": "ready", "question": question, "queries": [{"query_id": "Q0", "text": "original q"}, {"query_id": "Q1", "text": "generated secret"}], "children": [], "parents": [self.parent_candidate("p1", 1, "parent text")], "warnings": [], "trace": {}}

        def scorer(question, parents, config):
            questions.append(question)
            return [1.0]

        def generator(prompt, config):
            prompts.append(prompt)
            return "Trả lời [P1]"

        result = hr.answer_query_buoi09("original q", mode="multi_parent", config=self.config(), child_retrieval_fn=child_retrieval, parent_rerank_scorer=scorer, answer_generator=generator)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(questions, ["original q"])
        self.assertNotIn("generated secret", prompts[0])

    def test_sort_rank_change_final_k(self):
        result = hr.rerank_parent_candidates(
            "q",
            [self.parent_candidate("p1", 1, "one"), self.parent_candidate("p2", 2, "two"), self.parent_candidate("p3", 3, "three")],
            config=self.config(),
            scorer=lambda q, parents, cfg: [0.1, 2.0, 1.0],
        )
        self.assertEqual([parent["parent_id"] for parent in result["parents"]], ["p2", "p3"])
        self.assertEqual(result["parents"][0]["parent_rerank_rank"], 1)
        self.assertEqual(result["parents"][0]["parent_rank_change"], 1)

    def test_gate_accepted_rejected(self):
        parents = hr.rerank_parent_candidates("q", [self.parent_candidate("p1", 1, "one"), self.parent_candidate("p2", 2, "two")], config=self.config(), scorer=lambda q, p, c: [2.0, -2.0])["parents"]
        accepted = hr._accepted_parent_evidence(parents, config=self.config())
        self.assertEqual([item["parent_id"] for item in accepted], ["p1"])

    def test_no_evidence_no_generation(self):
        calls = {"generation": 0}

        def generator(prompt, config):
            calls["generation"] += 1
            return "should not call"

        result = hr.answer_query_buoi09(
            "q",
            mode="multi_parent",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "text")], "warnings": [], "trace": {}},
            parent_rerank_scorer=lambda q, p, c: [-2.0],
            answer_generator=generator,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(calls["generation"], 0)

    def test_flat_and_parent_mode_routing(self):
        flat = hr.answer_query_buoi09(
            "q",
            mode="single_flat",
            config=self.config(),
            hybrid_retriever_fn=lambda q, **k: {"candidates": [{"chunk_id": "c1", "text": "child", "source": "s.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1, "matched_by": []}], "trace": {}},
            child_rerank_scorer=lambda q, c, cfg: [2.0],
            answer_generator=lambda p, c: "Flat [E1]",
        )
        parent = hr.answer_query_buoi09(
            "q",
            mode="single_parent",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "parent")], "warnings": [], "trace": {}},
            parent_rerank_scorer=lambda q, p, cfg: [2.0],
            answer_generator=lambda p, c: "Parent [P1]",
        )
        self.assertEqual(flat["status"], "answered")
        self.assertEqual(parent["status"], "answered")

    def test_multi_query_failure_status(self):
        result = hr.answer_query_buoi09(
            "q",
            mode="multi_flat",
            config=self.config(),
            query_generator_fn=lambda q, cfg: (_ for _ in ()).throw(RuntimeError("gen down")),
            hybrid_retriever_fn=lambda q, **k: {},
            child_rerank_scorer=lambda q, c, cfg: [],
        )
        self.assertEqual(result["status"], "query_generation_unavailable")

    def test_reranker_failure_no_fallback(self):
        result = hr.answer_query_buoi09(
            "q",
            mode="multi_parent",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "text")], "warnings": [], "trace": {}},
            parent_rerank_scorer=lambda q, p, cfg: (_ for _ in ()).throw(RuntimeError("rerank down")),
        )
        self.assertEqual(result["status"], "reranker_unavailable")
        self.assertEqual(result["accepted_evidence"], [])

    def test_citation_parent_anchor_child_real(self):
        result = hr.answer_query_buoi09(
            "q",
            mode="multi_parent",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "parent")], "warnings": [], "trace": {}},
            parent_rerank_scorer=lambda q, p, cfg: [2.0],
            answer_generator=lambda p, c: "Answer [P1]",
        )
        self.assertEqual(result["citations"][0]["parent_id"], "p1")
        self.assertEqual(result["citations"][0]["anchor_child_id"], "c1")
        self.assertEqual(result["citations"][0]["supporting_child_ids"], ["c1"])

    def test_citation_label_validation(self):
        result = hr.answer_query_buoi09(
            "q",
            mode="multi_parent",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "parent")], "warnings": [], "trace": {}},
            parent_rerank_scorer=lambda q, p, cfg: [2.0],
            answer_generator=lambda p, c: "Answer [P99]",
        )
        self.assertEqual(result["status"], "citation_validation_failed")
        self.assertIn("invalid_parent_citation_label", result["warnings"][0])

    def test_multi_mode_max_two_generation_api_calls_with_fakes(self):
        result = hr.answer_query_buoi09(
            "q",
            mode="multi_parent",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "parent")], "warnings": [], "trace": {"gemini_expansion_call_count": 1, "semantic_embedding_call_count": 2}},
            parent_rerank_scorer=lambda q, p, cfg: [2.0],
            answer_generator=lambda p, c: "Answer [P1]",
        )
        self.assertLessEqual(result["trace"]["api_call_counts"]["generation_calls"], 2)
        self.assertEqual(result["trace"]["api_call_counts"]["semantic_embedding_calls"], 2)

    def test_compare_no_answer_generation(self):
        result = hr.compare_buoi09(
            "q",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "parent")], "warnings": [], "trace": {}},
            hybrid_retriever_fn=lambda q, **k: {"candidates": [{"chunk_id": "c1", "text": "child", "source": "s.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1, "matched_by": []}], "trace": {}},
            parent_rerank_scorer=lambda q, p, cfg: [2.0] if p else [],
            child_rerank_scorer=lambda q, c, cfg: [2.0] * len(c),
            query_generator_fn=lambda q, cfg: {"queries": [{"text": "variant", "focus": "paraphrase"}]},
        )
        self.assertFalse(result["answer_generation_called"])
        self.assertEqual(set(result["modes"]), {"single_flat", "multi_flat", "single_parent", "multi_parent"})

    def test_trace_identity_counts(self):
        result = hr.answer_query_buoi09(
            "q",
            mode="multi_parent",
            config=self.config(),
            child_retrieval_fn=lambda *a, **k: {"status": "ready", "question": "q", "queries": [], "children": [], "parents": [self.parent_candidate("p1", 1, "parent")], "warnings": [], "trace": {"semantic_embedding_call_count": 2}},
            parent_rerank_scorer=lambda q, p, cfg: [2.0],
            answer_generator=lambda p, c: "Answer [P1]",
        )
        trace = result["trace"]
        self.assertIn("model_identity", trace)
        self.assertIn("config_identity", trace)
        self.assertIn("api_call_counts", trace)
        self.assertEqual(trace["api_call_counts"]["rerank_calls"], 1)


if __name__ == "__main__":
    unittest.main()
