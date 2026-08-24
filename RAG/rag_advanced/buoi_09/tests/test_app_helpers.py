from __future__ import annotations

import unittest

import app


class AppHelperTest(unittest.TestCase):
    def test_mode_comparison_row(self):
        row = app.mode_comparison_row(
            "multi_parent",
            {
                "parent_candidates": [
                    {
                        "evidence_id": "P1",
                        "parent_id": "p1",
                        "source": "s.pdf",
                        "page_start": 1,
                        "page_end": 2,
                        "structural_path": {"article": "7"},
                        "parent_rank": 2,
                        "parent_rerank_rank": 1,
                        "text": "parent",
                    }
                ],
                "child_hits": [{"child_id": "c1"}],
                "trace": {"retrieval_trace": {"context_expansion_factor": 3.0, "expanded_parent_chars": 6}, "api_call_counts": {"generation_calls": 0, "semantic_embedding_calls": 2}},
                "status": "answered",
                "warnings": [],
            },
        )
        self.assertEqual(row["mode"], "multi_parent")
        self.assertEqual(row["unit type"], "parent")
        self.assertEqual(row["unique sources/articles"], "1 / 1")
        self.assertEqual(row["Embedding calls"], 2)

    def test_query_child_matrix(self):
        rows = app.query_child_matrix(
            [
                {"child_id": "c2", "multi_query_rank": 2, "support_query_ids": ["Q1"], "per_query_ranks": {"Q1": 1}, "multi_query_rrf_score": 0.1},
                {"child_id": "c1", "multi_query_rank": 1, "support_query_ids": ["Q0", "Q1"], "per_query_ranks": {"Q0": 2, "Q1": 3}, "multi_query_rrf_score": 0.2},
            ]
        )
        self.assertEqual(rows[0]["child_id"], "c1")
        self.assertEqual(rows[0]["Q0"], 2)
        self.assertEqual(rows[1]["Q0"], "—")

    def test_parent_tree_data(self):
        tree = app.parent_tree_data(
            {
                "parent_candidates": [
                    {"parent_id": "p1", "evidence_id": "P1", "structural_path": {"article": "7"}, "source": "s.pdf", "page_start": 1, "page_end": 2, "parent_rank": 1, "parent_rerank_rank": 1, "parent_rrf_score": 0.1, "parent_rerank_score": 0.9, "supporting_child_ids": ["c1"], "anchor_child_id": "c1", "text": "parent", "warnings": []}
                ],
                "child_hits": [{"child_id": "c1", "text": "child snippet", "support_query_ids": ["Q0"], "per_query_ranks": {"Q0": 1}, "multi_query_rank": 1}],
                "trace": {"retrieval_trace": {"child_to_parent_mapping": [{"child_id": "c1", "support_query_ids": ["Q0"], "multi_query_rank": 1}] }},
            }
        )
        self.assertEqual(tree[0]["parent_id"], "p1")
        self.assertTrue(tree[0]["children"][0]["is_anchor"])
        self.assertEqual(tree[0]["children"][0]["query_ids"], ["Q0"])

    def test_citation_formatting(self):
        self.assertIn("parent=p1", app.citation_display({"evidence_id": "P1", "parent_id": "p1", "anchor_child_id": "c1", "source": "s.pdf", "page_start": 1, "page_end": 2}))
        self.assertIn("child=c1", app.citation_display({"evidence_id": "E1", "child_id": "c1", "source": "s.pdf", "page_start": 1, "page_end": 1}))

    def test_warning_status_mapping(self):
        self.assertEqual(app.status_guidance("hierarchy_not_ready")["level"], "error")
        self.assertEqual(app.status_guidance("multi_query_partial")["level"], "warning")
        self.assertEqual(app.status_guidance("answered")["level"], "success")


if __name__ == "__main__":
    unittest.main()
