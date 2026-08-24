from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evaluate
import hierarchical_rag as hr


class EvaluateTest(unittest.TestCase):
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

    def questions(self):
        return [
            {
                "question_id": "QX",
                "question": "Điều 8 quy định gì?",
                "question_type": "exact",
                "relevant_child_ids": ["c1"],
                "relevant_parent_ids": ["p1"],
                "needs_human_review": True,
                "notes": "fixture",
            }
        ]

    def ready_store(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        manifest = {"schema_version": hr.SCHEMA_VERSION, "input_file_fingerprints": [{"sha256": "x"}], "config_identity": hr._config_identity(self.config())}
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (path / "children.json").write_text(json.dumps([{"child_id": "c1", "parent_id": "p1"}]), encoding="utf-8")
        (path / "parents.json").write_text(json.dumps([{"parent_id": "p1", "source": "s.pdf", "text": "parent"}]), encoding="utf-8")
        return path

    def test_metrics_math(self):
        self.assertEqual(evaluate.recall_at_k(["a", "b"], ["b", "c"], 2), 0.5)
        self.assertEqual(evaluate.mrr_at_k(["a", "b"], ["b"], 2), 0.5)
        self.assertGreater(evaluate.ndcg_at_k(["a", "b"], ["b"], 2), 0.0)

    def test_mock_evaluate_offline(self):
        report = evaluate.evaluate_questions(self.questions(), modes=["single_flat", "multi_parent"], k=3, config=self.config(), retriever=evaluate.mock_retriever, hierarchy_storage=self.ready_store())
        self.assertTrue(report["human_review_warning"])
        self.assertIn("single_flat", report["metrics"])
        self.assertEqual(report["metrics"]["single_flat"]["child_recall_at_k"], 1.0)
        self.assertEqual(report["metrics"]["multi_parent"]["parent_recall_at_k"], 1.0)

    def test_stale_parent_ids_fail(self):
        bad = self.questions()
        bad[0]["relevant_parent_ids"] = ["missing-parent"]
        with self.assertRaisesRegex(ValueError, "stale relevant_parent_ids"):
            evaluate.evaluate_questions(bad, modes=["single_flat"], k=3, config=self.config(), retriever=evaluate.mock_retriever, hierarchy_storage=self.ready_store())

    def test_save_report_latest_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {"timestamp": "2026-08-12T00:00:00+00:00", "metrics": {}, "warnings": []}
            path = evaluate.save_report(report, Path(tmp) / "report.json")
            latest = Path(tmp) / "latest_report.json"
            self.assertTrue(path.exists())
            self.assertTrue(latest.exists())
            self.assertEqual(json.loads(latest.read_text(encoding="utf-8"))["timestamp"], report["timestamp"])


if __name__ == "__main__":
    unittest.main()
