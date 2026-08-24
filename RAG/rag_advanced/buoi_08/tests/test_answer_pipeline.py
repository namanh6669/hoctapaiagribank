from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

BUOI_08_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag


ENV = {
    "GEMINI_API_KEY": "",
    "GEMINI_EMBEDDING_MODEL": "test-embedding-model",
    "GEMINI_EMBEDDING_DIM": "128",
    "GEMINI_GENERATION_MODEL": "test-generation-model",
    "DEFAULT_TOP_K": "5",
    "RAG_MAX_DISTANCE": "0.45",
    "BM25_CANDIDATES": "3",
    "SEMANTIC_CANDIDATES": "3",
    "RRF_K": "60",
    "RRF_BM25_WEIGHT": "1.0",
    "RRF_SEMANTIC_WEIGHT": "1.0",
    "RERANK_CANDIDATES": "3",
    "FINAL_TOP_K": "3",
    "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
    "RERANKER_MAX_LENGTH": "512",
    "RERANK_BATCH_SIZE": "2",
    "RERANK_MIN_SCORE": "0.50",
    "RERANK_DEVICE": "auto",
}


def bm25_candidate(chunk_id="b1", *, distance=None):
    item = {
        "chunk_id": chunk_id,
        "text": f"text {chunk_id}",
        "source": "fixture.pdf",
        "page_start": 1,
        "page_end": 1,
        "bm25_rank": 1,
        "bm25_score": 3.0,
    }
    if distance is not None:
        item["semantic_rank"] = 1
        item["semantic_distance"] = distance
    return item


def semantic_candidate(chunk_id="s1", distance=0.1):
    return {
        "chunk_id": chunk_id,
        "text": f"text {chunk_id}",
        "source": "fixture.pdf",
        "page_start": 2,
        "page_end": 3,
        "semantic_rank": 1,
        "semantic_distance": distance,
    }


def fused_candidate(chunk_id="h1", distance=0.1):
    return {
        "chunk_id": chunk_id,
        "text": f"text {chunk_id}",
        "source": "fixture.pdf",
        "page_start": 1,
        "page_end": 1,
        "bm25_rank": 1,
        "bm25_score": 2.0,
        "semantic_rank": 1,
        "semantic_distance": distance,
        "rrf_score": 0.03,
        "fused_rank": 1,
        "matched_by": ["bm25", "semantic"],
    }


def reranked_candidate(chunk_id="r1", score=0.8, rank=1):
    item = fused_candidate(chunk_id)
    item.update({
        "rerank_raw_score": 1.0,
        "rerank_score": score,
        "rerank_rank": rank,
        "rank_change": item["fused_rank"] - rank,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "rerank_latency_ms": 1.2,
    })
    return item


class TestAnswerPipeline(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, ENV, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_gating_by_mode(self):
        calls = []
        def gen(prompt, config):
            calls.append(prompt)
            return "Trả lời [E1]"
        semantic = lambda question, *, candidate_k, strategy, storage_path: [semantic_candidate("s1", 0.9)]
        result = advanced_rag.answer_query("Q", mode="semantic", semantic_retriever=semantic, generator=gen)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(calls, [])

        result = advanced_rag.answer_query(
            "Q",
            mode="hybrid_rerank",
            bm25_retriever=lambda q, c, k: [],
            semantic_retriever=lambda question, *, candidate_k, strategy, storage_path: [],
            rerank_scorer=lambda q, candidates, config: [0.0],
            generator=gen,
        )
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_rejected_evidence_not_in_prompt(self):
        prompts = []
        def gen(prompt, config):
            prompts.append(prompt)
            return "A [E1]"
        retrieval = lambda question, *, candidate_k, strategy, storage_path: [
            semantic_candidate("good", 0.1),
            semantic_candidate("bad", 0.9),
        ]
        result = advanced_rag.answer_query("Q", mode="semantic", semantic_retriever=retrieval, generator=gen)
        self.assertEqual(result["status"], "answered")
        self.assertIn("text good", prompts[0])
        self.assertNotIn("text bad", prompts[0])

    def test_trace_counts_timings_have_keys_and_schema_all_statuses(self):
        result = advanced_rag.answer_query(
            "Q",
            mode="semantic",
            semantic_retriever=lambda question, *, candidate_k, strategy, storage_path: [semantic_candidate("s", 0.9)],
            generator=lambda p, c: "unused",
        )
        self.assertEqual(set(result), {"status", "mode", "question", "answer", "evidence", "citations", "warnings", "trace"})
        for key in advanced_rag.LATENCY_KEYS:
            self.assertIn(key, result["trace"]["latency_ms"])
        for key in ("bm25_candidates", "semantic_candidates", "overlap", "union", "reranked", "accepted", "generation_called"):
            self.assertIn(key, result["trace"])

    def test_citation_maps_real_metadata_and_removes_fake_label(self):
        result = advanced_rag.answer_query(
            "Q",
            mode="semantic",
            semantic_retriever=lambda question, *, candidate_k, strategy, storage_path: [semantic_candidate("real", 0.1)],
            generator=lambda p, c: "Câu trả lời [E1] fake [E99]",
        )
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["chunk_id"], "real")
        self.assertNotIn("E99", result["answer"])
        self.assertTrue(result["warnings"])

    def test_generation_called_at_most_once_and_empty_is_retrieval_only(self):
        calls = []
        def gen(prompt, config):
            calls.append(prompt)
            return "   "
        result = advanced_rag.answer_query(
            "Q",
            mode="semantic",
            semantic_retriever=lambda question, *, candidate_k, strategy, storage_path: [semantic_candidate("s", 0.1)],
            generator=gen,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "retrieval_only")
        self.assertTrue(result["evidence"])

    def test_compare_does_not_call_generation(self):
        with mock.patch.object(advanced_rag, "_generation_config", side_effect=AssertionError("generation must not run")):
            result = advanced_rag.compare_modes(
                "Q",
                bm25_retriever=lambda q, c, k: [bm25_candidate("b")],
                semantic_retriever=lambda question, *, candidate_k, strategy, storage_path: [semantic_candidate("s", 0.1)],
                rerank_scorer=lambda q, candidates, config: [1.0, 0.5, 0.2],
            )
        self.assertIn("rows", result)
        self.assertIn("mode_results", result)

    def test_reranker_unavailable_status(self):
        result = advanced_rag.answer_query(
            "Q",
            mode="hybrid_rerank",
            bm25_retriever=lambda q, c, k: [bm25_candidate("b")],
            semantic_retriever=lambda question, *, candidate_k, strategy, storage_path: [semantic_candidate("s", 0.1)],
            rerank_scorer=lambda q, candidates, config: (_ for _ in ()).throw(RuntimeError("model missing")),
            generator=lambda p, c: "must not run",
        )
        self.assertEqual(result["status"], "reranker_unavailable")
        self.assertFalse(result["trace"]["generation_called"])
        self.assertIn("model missing", result["warnings"][0])

    def test_bm25_generation_requires_semantic_gate(self):
        calls = []
        result = advanced_rag.answer_query(
            "Q",
            mode="bm25",
            bm25_retriever=lambda q, c, k: [bm25_candidate("b")],
            generator=lambda p, c: calls.append(p) or "A [E1]",
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(calls, [])

        result = advanced_rag.answer_query(
            "Q",
            mode="bm25",
            bm25_retriever=lambda q, c, k: [bm25_candidate("b", distance=0.1)],
            generator=lambda p, c: calls.append(p) or "A [E1]",
        )
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
