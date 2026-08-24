from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

BUOI_08_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag


def candidate(chunk_id: str, *, text: str | None = None, bm25_rank=None, semantic_rank=None) -> dict:
    base = {
        "chunk_id": chunk_id,
        "text": text or f"text {chunk_id}",
        "source": "fixture.pdf",
        "page_start": 1,
        "page_end": 1,
    }
    if bm25_rank is not None:
        base["bm25_rank"] = bm25_rank
        base["bm25_score"] = float(10 - bm25_rank)
    if semantic_rank is not None:
        base["semantic_rank"] = semantic_rank
        base["semantic_distance"] = float(semantic_rank) / 10
    return base


def chunk(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "strategy": "hierarchical",
        "source": "fixture.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": f"text {chunk_id}",
    }


ENV = {
    "GEMINI_API_KEY": "",
    "GEMINI_EMBEDDING_MODEL": "test-embedding-model",
    "GEMINI_EMBEDDING_DIM": "128",
    "GEMINI_GENERATION_MODEL": "test-generation-model",
    "DEFAULT_TOP_K": "5",
    "RAG_MAX_DISTANCE": "0.45",
    "BM25_CANDIDATES": "2",
    "SEMANTIC_CANDIDATES": "2",
    "RRF_K": "60",
    "RRF_BM25_WEIGHT": "1.0",
    "RRF_SEMANTIC_WEIGHT": "1.0",
    "RERANK_CANDIDATES": "20",
    "FINAL_TOP_K": "5",
    "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
    "RERANKER_MAX_LENGTH": "512",
    "RERANK_BATCH_SIZE": "4",
    "RERANK_MIN_SCORE": "0.50",
    "RERANK_DEVICE": "auto",
}


class TestRRFFusion(unittest.TestCase):
    def test_rrf_formula_arithmetic(self):
        fused = advanced_rag.rrf_fuse(
            [candidate("x", bm25_rank=2)],
            [candidate("x", semantic_rank=3)],
            rrf_k=60,
            bm25_weight=2.0,
            semantic_weight=1.5,
        )
        expected = 2.0 / (60 + 2) + 1.5 / (60 + 3)
        self.assertAlmostEqual(fused[0]["rrf_score"], expected)

    def test_candidate_overlap_not_duplicated(self):
        fused = advanced_rag.rrf_fuse(
            [candidate("x", bm25_rank=1)],
            [candidate("x", semantic_rank=1)],
            rrf_k=60,
            bm25_weight=1.0,
            semantic_weight=1.0,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["matched_by"], ["bm25", "semantic"])

    def test_bm25_only_candidate_is_kept(self):
        fused = advanced_rag.rrf_fuse([candidate("b", bm25_rank=1)], [], rrf_k=60, bm25_weight=1.0, semantic_weight=1.0)
        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertIsNone(fused[0]["semantic_rank"])
        self.assertEqual(fused[0]["matched_by"], ["bm25"])

    def test_semantic_only_candidate_is_kept(self):
        fused = advanced_rag.rrf_fuse([], [candidate("s", semantic_rank=1)], rrf_k=60, bm25_weight=1.0, semantic_weight=1.0)
        self.assertEqual(fused[0]["chunk_id"], "s")
        self.assertIsNone(fused[0]["bm25_rank"])
        self.assertEqual(fused[0]["matched_by"], ["semantic"])

    def test_zero_weight_removes_branch_contribution(self):
        fused = advanced_rag.rrf_fuse(
            [candidate("x", bm25_rank=1)],
            [candidate("x", semantic_rank=1)],
            rrf_k=60,
            bm25_weight=0.0,
            semantic_weight=1.0,
        )
        self.assertAlmostEqual(fused[0]["rrf_score"], 1.0 / 61)

    def test_tie_break_deterministic(self):
        fused = advanced_rag.rrf_fuse(
            [candidate("b", bm25_rank=1), candidate("a", bm25_rank=1)],
            [],
            rrf_k=60,
            bm25_weight=1.0,
            semantic_weight=1.0,
        )
        self.assertEqual([item["chunk_id"] for item in fused], ["a", "b"])
        self.assertEqual([item["fused_rank"] for item in fused], [1, 2])

    def test_metadata_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "Metadata mismatch"):
            advanced_rag.rrf_fuse(
                [candidate("x", text="bm25 text", bm25_rank=1)],
                [candidate("x", text="semantic text", semantic_rank=1)],
                rrf_k=60,
                bm25_weight=1.0,
                semantic_weight=1.0,
            )


class TestHybridRetrieval(unittest.TestCase):
    def test_trace_counts_and_each_retriever_called_once(self):
        calls = {"bm25": 0, "semantic": 0}
        chunks = [chunk("b1"), chunk("overlap"), chunk("s1")]

        def fake_load_chunks(input_path, strategy):
            return chunks, {"files_read": 1, "valid_chunks": len(chunks)}

        def fake_bm25(question, loaded_chunks, candidate_k):
            calls["bm25"] += 1
            self.assertEqual(loaded_chunks, chunks)
            self.assertEqual(candidate_k, 2)
            return [candidate("overlap", bm25_rank=1), candidate("b1", bm25_rank=2)]

        def fake_semantic(question, *, candidate_k, strategy, storage_path):
            calls["semantic"] += 1
            self.assertEqual(candidate_k, 2)
            return [candidate("overlap", semantic_rank=1), candidate("s1", semantic_rank=2)]

        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch("rag.load_chunks", side_effect=fake_load_chunks):
                result = advanced_rag.hybrid_retrieve(
                    "Điều 7?",
                    bm25_retriever=fake_bm25,
                    semantic_retriever=fake_semantic,
                )

        self.assertEqual(calls, {"bm25": 1, "semantic": 1})
        trace = result["trace"]
        self.assertEqual(trace["bm25_candidate_count"], 2)
        self.assertEqual(trace["semantic_candidate_count"], 2)
        self.assertEqual(trace["union_count"], 3)
        self.assertEqual(trace["overlap_count"], 1)
        self.assertEqual(trace["fused_count"], 3)
        for key in ("tokenize_bm25", "semantic", "fusion"):
            self.assertIn(key, trace["latency_ms"])
            self.assertGreaterEqual(trace["latency_ms"][key], 0.0)

    def test_hybrid_does_not_load_reranker_or_generation(self):
        def fake_bm25(question, loaded_chunks, candidate_k):
            return [candidate("b1", bm25_rank=1)]

        def fake_semantic(question, *, candidate_k, strategy, storage_path):
            return []

        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch("rag.load_chunks", return_value=([chunk("b1")], {"files_read": 1, "valid_chunks": 1})):
                with mock.patch.dict(sys.modules, {"transformers": None, "torch": None}):
                    with mock.patch("rag.gemini_generate_answer", side_effect=AssertionError("generation must not run")):
                        result = advanced_rag.hybrid_retrieve(
                            "Điều 7?",
                            bm25_retriever=fake_bm25,
                            semantic_retriever=fake_semantic,
                        )
        self.assertEqual(result["candidates"][0]["chunk_id"], "b1")


if __name__ == "__main__":
    unittest.main()
