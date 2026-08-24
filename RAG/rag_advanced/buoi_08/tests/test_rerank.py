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
    "FINAL_TOP_K": "2",
    "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
    "RERANKER_MAX_LENGTH": "512",
    "RERANK_BATCH_SIZE": "2",
    "RERANK_MIN_SCORE": "0.50",
    "RERANK_DEVICE": "auto",
}


def fused(chunk_id: str, fused_rank: int, text: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text or f"text {chunk_id}",
        "source": "fixture.pdf",
        "page_start": 1,
        "page_end": 1,
        "bm25_rank": fused_rank,
        "bm25_score": float(10 - fused_rank),
        "semantic_rank": None,
        "semantic_distance": None,
        "rrf_score": 1.0 / (60 + fused_rank),
        "fused_rank": fused_rank,
        "matched_by": ["bm25"],
    }


class TestRerank(unittest.TestCase):
    def setUp(self):
        advanced_rag._RERANKER_CACHE.clear()

    def config(self, **overrides):
        with mock.patch.dict(os.environ, {**ENV, **{k: str(v) for k, v in overrides.items()}}, clear=True):
            return advanced_rag.load_config(advanced_rag.ENV_EXAMPLE_PATH)

    def test_lazy_loading(self):
        config = self.config()
        with mock.patch.object(advanced_rag, "get_reranker", side_effect=AssertionError("should not load")):
            result = advanced_rag.rerank_fused_candidates(
                "Điều 7?",
                [fused("a", 1)],
                config=config,
                scorer=lambda question, candidates, config: [1.0],
            )
        self.assertEqual(result["status"], "reranked")

    def test_one_pair_for_each_candidate_and_limit(self):
        config = self.config(RERANK_CANDIDATES=2, FINAL_TOP_K=2)
        seen = []
        def scorer(question, candidates, config):
            seen.extend((question, item["text"]) for item in candidates)
            return [0.0, 1.0]
        result = advanced_rag.rerank_fused_candidates(
            "Điều 7?",
            [fused("a", 1), fused("b", 2), fused("c", 3)],
            config=config,
            scorer=scorer,
        )
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen, [("Điều 7?", "text a"), ("Điều 7?", "text b")])
        self.assertEqual(result["reranked_count"], 2)

    def test_batch_scorer_does_not_change_count(self):
        config = self.config(RERANK_CANDIDATES=3, FINAL_TOP_K=3)
        result = advanced_rag.rerank_fused_candidates(
            "Q",
            [fused("a", 1), fused("b", 2), fused("c", 3)],
            config=config,
            scorer=lambda question, candidates, config: [0.1, 0.2, 0.3],
        )
        self.assertEqual(result["reranked_count"], 3)
        self.assertEqual(len(result["candidates"]), 3)

    def test_sigmoid_score(self):
        config = self.config(FINAL_TOP_K=1)
        result = advanced_rag.rerank_fused_candidates(
            "Q",
            [fused("a", 1)],
            config=config,
            scorer=lambda question, candidates, config: [0.0],
        )
        self.assertAlmostEqual(result["candidates"][0]["rerank_score"], 0.5)
        self.assertAlmostEqual(result["candidates"][0]["rerank_raw_score"], 0.0)

    def test_sort_and_tie_break(self):
        config = self.config(RERANK_CANDIDATES=3, FINAL_TOP_K=3)
        result = advanced_rag.rerank_fused_candidates(
            "Q",
            [fused("b", 2), fused("a", 1), fused("c", 3)],
            config=config,
            scorer=lambda question, candidates, config: [1.0, 1.0, 2.0],
        )
        self.assertEqual([item["chunk_id"] for item in result["candidates"]], ["c", "a", "b"])

    def test_rank_change(self):
        config = self.config(RERANK_CANDIDATES=3, FINAL_TOP_K=3)
        result = advanced_rag.rerank_fused_candidates(
            "Q",
            [fused("a", 1), fused("b", 2), fused("c", 3)],
            config=config,
            scorer=lambda question, candidates, config: [0.0, 2.0, 1.0],
        )
        by_id = {item["chunk_id"]: item for item in result["candidates"]}
        self.assertEqual(by_id["b"]["rerank_rank"], 1)
        self.assertEqual(by_id["b"]["rank_change"], 1)
        self.assertEqual(by_id["a"]["rank_change"], -2)

    def test_only_returns_final_top_k(self):
        config = self.config(RERANK_CANDIDATES=4, FINAL_TOP_K=2)
        result = advanced_rag.rerank_fused_candidates(
            "Q",
            [fused("a", 1), fused("b", 2), fused("c", 3), fused("d", 4)],
            config=config,
            scorer=lambda question, candidates, config: [0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["reranked_count"], 4)

    def test_model_error_not_silent_fallback(self):
        config = self.config()
        result = advanced_rag.rerank_fused_candidates(
            "Q",
            [fused("a", 1)],
            config=config,
            scorer=lambda question, candidates, config: (_ for _ in ()).throw(RuntimeError("download failed")),
        )
        self.assertEqual(result["status"], "reranker_unavailable")
        self.assertEqual(result["candidates"], [])
        self.assertIn("download failed", result["error"])

    def test_hybrid_rerank_does_not_call_generation_or_load_model_with_injection(self):
        def fake_bm25(question, chunks, candidate_k):
            return [{
                "chunk_id": "a", "text": "text a", "source": "fixture.pdf", "page_start": 1, "page_end": 1,
                "bm25_rank": 1, "bm25_score": 1.0,
            }]
        def fake_semantic(question, *, candidate_k, strategy, storage_path):
            return []
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch("rag.load_chunks", return_value=([{"chunk_id": "a", "text": "text a"}], {})):
                with mock.patch.object(advanced_rag, "get_reranker", side_effect=AssertionError("should not load")):
                    with mock.patch("rag.gemini_generate_answer", side_effect=AssertionError("generation must not run")):
                        result = advanced_rag.hybrid_rerank_retrieve(
                            "Q",
                            bm25_retriever=fake_bm25,
                            semantic_retriever=fake_semantic,
                            rerank_scorer=lambda question, candidates, config: [1.0],
                        )
        self.assertEqual(result["status"], "reranked")


if __name__ == "__main__":
    unittest.main()
