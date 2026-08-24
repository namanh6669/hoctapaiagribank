from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evaluate


class TestMetrics(unittest.TestCase):
    def test_recall_mrr_ndcg_hand_calculated(self):
        ranked = ["a", "b", "c", "d"]
        relevant = ["b", "d"]
        self.assertEqual(evaluate.recall_at_k(ranked, relevant, 3), 0.5)
        self.assertEqual(evaluate.mrr_at_k(ranked, relevant, 3), 0.5)
        expected_ndcg = (1 / evaluate._log2(3)) / (1 / evaluate._log2(2) + 1 / evaluate._log2(3))
        self.assertAlmostEqual(evaluate.ndcg_at_k(ranked, relevant, 3), expected_ndcg)

    def test_no_relevance_returns_zero(self):
        self.assertEqual(evaluate.recall_at_k(["a"], [], 1), 0.0)
        self.assertEqual(evaluate.mrr_at_k(["a"], [], 1), 0.0)
        self.assertEqual(evaluate.ndcg_at_k(["a"], [], 1), 0.0)


class TestEvaluator(unittest.TestCase):
    def test_evaluate_records_warning_and_failures(self):
        questions = [
            {"query_id": "Q1", "question": "q1", "relevant_chunk_ids": ["a"], "needs_human_review": True},
            {"query_id": "Q2", "question": "q2", "relevant_chunk_ids": ["b"], "needs_human_review": True},
        ]
        def retriever(question, mode, k, strategy):
            if question["query_id"] == "Q2":
                raise RuntimeError("boom")
            return ["a", "x"]
        report = evaluate.evaluate_questions(questions, modes=["bm25"], strategy="hierarchical", k=2, retriever=retriever)
        self.assertTrue(report["warnings"])
        self.assertIsNone(report["official_winner"])
        self.assertEqual(report["metrics"]["bm25"]["failed_count"], 1)
        self.assertEqual(report["queries"][1]["modes"]["bm25"]["status"], "failed")

    def test_mock_synthetic_command_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            questions_path = tmp_path / "questions.json"
            output_path = tmp_path / "report.json"
            questions_path.write_text(
                json.dumps([
                    {
                        "query_id": "Q1",
                        "question": "q1",
                        "relevant_chunk_ids": ["a"],
                        "needs_human_review": True,
                        "mock_ranked_chunk_ids": ["a", "z"],
                    }
                ]),
                encoding="utf-8",
            )
            exit_code = evaluate.main([
                "--questions", str(questions_path),
                "--modes", "bm25,semantic",
                "--k", "2",
                "--output", str(output_path),
                "--mock-synthetic",
            ])
            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(set(report["metrics"]), {"bm25", "semantic"})
            self.assertTrue(report["warnings"])

    def test_evaluator_does_not_call_generation(self):
        def retriever(question, mode, k, strategy):
            return ["a"]
        report = evaluate.evaluate_questions(
            [{"query_id": "Q1", "question": "q", "relevant_chunk_ids": ["a"], "needs_human_review": True}],
            modes=["bm25"],
            strategy="hierarchical",
            k=1,
            retriever=retriever,
        )
        self.assertEqual(report["metrics"]["bm25"]["recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
