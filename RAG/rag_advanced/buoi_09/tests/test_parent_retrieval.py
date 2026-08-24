from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import hierarchical_rag as hr


class ParentRetrievalTest(unittest.TestCase):
    def config(self, *, parent_score_child_limit: int = 2, parent_candidates: int = 10, total_context_max_chars: int = 1000) -> hr.HierarchyConfig:
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
            parent_score_child_limit=parent_score_child_limit,
            parent_rrf_k=60,
            parent_candidates=parent_candidates,
            final_parent_top_k=3,
            total_context_max_chars=total_context_max_chars,
        )

    def child_hit(self, child_id: str, rank: int, *, text: str | None = None, queries=None) -> dict:
        queries = queries or ["Q0"]
        return {
            "child_id": child_id,
            "text": text or f"child text {child_id}",
            "source": "s.pdf",
            "page_start": 1,
            "page_end": 1,
            "multi_query_rrf_score": 1 / (60 + rank),
            "multi_query_rank": rank,
            "support_query_count": len(queries),
            "support_query_ids": queries,
            "per_query_ranks": {query_id: rank for query_id in queries},
            "per_query_trace": {},
        }

    def parent(self, parent_id: str, text: str, *, page_start: int = 1, page_end: int = 2, warnings=None) -> dict:
        return {
            "parent_id": parent_id,
            "source": "s.pdf",
            "page_start": page_start,
            "page_end": page_end,
            "article_key": "article::7",
            "window_index": 1,
            "child_ids": [],
            "text": text,
            "char_count": len(text),
            "ambiguous_child_count": 0,
            "warnings": warnings or [],
        }

    def child_registry(self, mapping: dict[str, str]) -> dict[str, dict]:
        return {child_id: {"child_id": child_id, "parent_id": parent_id, "structural_path": {"chapter": None, "article": "7", "clause": None, "point": None}} for child_id, parent_id in mapping.items()}

    def test_child_maps_to_parent(self):
        candidates, trace = hr.aggregate_parent_candidates(
            [self.child_hit("c1", 1)],
            child_registry=self.child_registry({"c1": "p1"}),
            parents_by_id={"p1": self.parent("p1", "parent text")},
            config=self.config(),
        )
        self.assertEqual(candidates[0]["parent_id"], "p1")
        self.assertEqual(candidates[0]["supporting_child_ids"], ["c1"])
        self.assertEqual(trace["child_to_parent_mapping"][0]["parent_id"], "p1")

    def test_missing_hierarchy_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = hr.parent_retrieve("q", config=self.config(), storage_dir=tmp, child_retrieval_fn=lambda *a, **k: {})
        self.assertEqual(result["status"], "hierarchy_not_ready")
        self.assertIn("missing_hierarchy_store_files", result["warnings"][0])

    def test_stale_hierarchy_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "children.json").write_text("[]", encoding="utf-8")
            (path / "parents.json").write_text("[]", encoding="utf-8")
            (path / "manifest.json").write_text(json.dumps({"schema_version": "old", "input_file_fingerprints": [{}], "config_identity": {}}), encoding="utf-8")
            result = hr.parent_retrieve("q", config=self.config(), storage_dir=path, child_retrieval_fn=lambda *a, **k: {})
        self.assertEqual(result["status"], "hierarchy_not_ready")
        self.assertIn("stale_hierarchy_schema_version", result["warnings"])

    def test_parent_aggregation_formula_by_hand(self):
        candidates, _trace = hr.aggregate_parent_candidates(
            [self.child_hit("c1", 1), self.child_hit("c2", 4)],
            child_registry=self.child_registry({"c1": "p1", "c2": "p1"}),
            parents_by_id={"p1": self.parent("p1", "parent text")},
            config=self.config(parent_score_child_limit=2),
        )
        expected = 1 / (60 + 1) + 1 / (60 + 4)
        self.assertAlmostEqual(candidates[0]["parent_rrf_score"], expected)

    def test_child_score_cap_and_scoring_supporting_split(self):
        candidates, _trace = hr.aggregate_parent_candidates(
            [self.child_hit("c1", 1), self.child_hit("c2", 2), self.child_hit("c3", 3)],
            child_registry=self.child_registry({"c1": "p1", "c2": "p1", "c3": "p1"}),
            parents_by_id={"p1": self.parent("p1", "parent text")},
            config=self.config(parent_score_child_limit=2),
        )
        self.assertEqual(candidates[0]["scoring_child_ids"], ["c1", "c2"])
        self.assertEqual(candidates[0]["supporting_child_ids"], ["c1", "c2", "c3"])
        self.assertAlmostEqual(candidates[0]["parent_rrf_score"], 1 / 61 + 1 / 62)

    def test_parent_deduplicate(self):
        selected, trace, _warnings = hr.apply_parent_context_budget(
            [self.parent_candidate("p1", "abc"), self.parent_candidate("p1", "abc")],
            config=self.config(),
        )
        self.assertEqual([parent["parent_id"] for parent in selected], ["p1"])
        self.assertEqual(trace["used_context_chars"], 3)

    def test_sort_tie_break_deterministic(self):
        candidates, _trace = hr.aggregate_parent_candidates(
            [self.child_hit("c2", 1), self.child_hit("c1", 1)],
            child_registry=self.child_registry({"c1": "pA", "c2": "pB"}),
            parents_by_id={"pA": self.parent("pA", "A"), "pB": self.parent("pB", "B")},
            config=self.config(),
        )
        self.assertEqual([candidate["parent_id"] for candidate in candidates], ["pA", "pB"])
        self.assertEqual([candidate["parent_rank"] for candidate in candidates], [1, 2])

    def test_candidate_limit(self):
        candidates, trace = hr.aggregate_parent_candidates(
            [self.child_hit("c1", 1), self.child_hit("c2", 2), self.child_hit("c3", 3)],
            child_registry=self.child_registry({"c1": "p1", "c2": "p2", "c3": "p3"}),
            parents_by_id={"p1": self.parent("p1", "1"), "p2": self.parent("p2", "2"), "p3": self.parent("p3", "3")},
            config=self.config(parent_candidates=2),
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(trace["parents_dropped_by_candidate_limit"], ["p3"])

    def test_context_budget_parent_boundary(self):
        candidates = [self.parent_candidate("p1", "a" * 5), self.parent_candidate("p2", "b" * 6), self.parent_candidate("p3", "c" * 3)]
        selected, trace, _warnings = hr.apply_parent_context_budget(candidates, config=self.config(total_context_max_chars=10))
        self.assertEqual([parent["parent_id"] for parent in selected], ["p1", "p3"])
        self.assertEqual(trace["parents_dropped_by_context_budget"], ["p2"])

    def test_oversized_first_parent_warning(self):
        selected, _trace, warnings = hr.apply_parent_context_budget([self.parent_candidate("p1", "x" * 20)], config=self.config(total_context_max_chars=10))
        self.assertEqual([parent["parent_id"] for parent in selected], ["p1"])
        self.assertIn("first_parent_exceeds_total_context_budget", warnings)
        self.assertIn("first_parent_exceeds_total_context_budget", selected[0]["warnings"])

    def test_expansion_factor_count_trace(self):
        result = hr.parent_retrieve(
            "q",
            config=self.config(),
            storage_dir=self.ready_store(),
            child_retrieval_fn=lambda *a, **k: {
                "status": "ready",
                "question": "q",
                "queries": [],
                "children": [self.child_hit("c1", 1, text="abc")],
                "warnings": [],
            },
        )
        self.assertEqual(result["trace"]["input_child_hit_count"], 1)
        self.assertEqual(result["trace"]["unique_parent_count"], 1)
        self.assertEqual(result["trace"]["child_chars"], 3)
        self.assertEqual(result["trace"]["expanded_parent_chars"], len("parent text c1"))
        self.assertGreater(result["trace"]["context_expansion_factor"], 1.0)

    def test_no_reranker_generation(self):
        result = hr.parent_retrieve(
            "q",
            config=self.config(),
            storage_dir=self.ready_store(),
            child_retrieval_fn=lambda *a, **k: {
                "status": "ready",
                "question": "q",
                "queries": [],
                "children": [self.child_hit("c1", 1)],
                "warnings": [],
            },
        )
        lowered = str(result).casefold()
        self.assertNotIn("rerank", lowered)
        self.assertNotIn("answer", lowered)

    def test_duplicate_child_text_across_parents_is_error(self):
        with self.assertRaisesRegex(ValueError, "Duplicate child text"):
            hr.aggregate_parent_candidates(
                [self.child_hit("c1", 1, text="same"), self.child_hit("c2", 2, text="same")],
                child_registry=self.child_registry({"c1": "p1", "c2": "p2"}),
                parents_by_id={"p1": self.parent("p1", "one"), "p2": self.parent("p2", "two")},
                config=self.config(),
            )

    def parent_candidate(self, parent_id: str, text: str) -> dict:
        return {
            "parent_id": parent_id,
            "source": "s.pdf",
            "page_start": 1,
            "page_end": 1,
            "structural_path": {},
            "text": text,
            "parent_rrf_score": 1.0,
            "parent_rank": 1,
            "anchor_child_id": "c",
            "scoring_child_ids": ["c"],
            "supporting_child_ids": ["c"],
            "support_query_ids": ["Q0"],
            "best_child_rank": 1,
            "ambiguous": False,
            "warnings": [],
        }

    def ready_store(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        config = self.config()
        manifest = {"schema_version": hr.SCHEMA_VERSION, "input_file_fingerprints": [{"sha256": "x"}], "config_identity": hr._config_identity(config)}
        children = [{"child_id": "c1", "parent_id": "p1", "structural_path": {"chapter": None, "article": "7", "clause": None, "point": None}}]
        parents = [self.parent("p1", "parent text c1")]
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (path / "children.json").write_text(json.dumps(children), encoding="utf-8")
        (path / "parents.json").write_text(json.dumps(parents), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
