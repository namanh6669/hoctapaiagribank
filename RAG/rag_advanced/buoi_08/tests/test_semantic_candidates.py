from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BUOI_08_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag
import rag


TEST_CONFIG = {
    "api_key": "test-key",
    "api_key_status": "Có",
    "embedding_model": "test-embedding-model",
    "embedding_dim": 128,
    "generation_model": "test-generation-model",
    "default_top_k": 5,
    "max_distance": 0.45,
}


def base_chunk(chunk_id: str, text: str, page_start: int = 1, page_end: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "strategy": "hierarchical",
        "source": "fixture.pdf",
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
    }


def vector(axis: int = 0, value: float = 1.0, dim: int = 128) -> list[float]:
    data = [0.0] * dim
    data[axis] = value
    return data


def fake_chunk_embedder(chunk: dict, config: dict) -> list[float]:
    if chunk["chunk_id"] == "c1":
        return vector(0, 1.0, config["embedding_dim"])
    if chunk["chunk_id"] == "c2":
        data = vector(0, 0.8, config["embedding_dim"])
        data[1] = 0.2
        return data
    return vector(1, 1.0, config["embedding_dim"])


def fake_query_embedder(question: str, config: dict) -> list[float]:
    return vector(0, 1.0, config["embedding_dim"])


class SemanticCandidateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.config_patch = mock.patch.object(rag, "load_config", return_value=dict(TEST_CONFIG))
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.tmp.cleanup()

    def index_chunks(self, chunks: list[dict]) -> Path:
        storage = self.tmp_path / "chroma"
        input_path = self.tmp_path / "chunks.json"
        import json
        input_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        rag.index_chunks(
            strategy="hierarchical",
            input_path=input_path,
            storage_path=storage,
            embedder=fake_chunk_embedder,
        )
        return storage

    def test_semantic_top_k_count_and_order(self):
        storage = self.index_chunks([
            base_chunk("c1", "Điều 7 cơ cấu nợ"),
            base_chunk("c2", "Cơ cấu lại thời hạn trả nợ"),
            base_chunk("c3", "Hội thao nội bộ"),
        ])
        candidates = advanced_rag.semantic_candidates(
            "Điều 7 quy định gì?",
            candidate_k=2,
            storage_path=storage,
            query_embedder=fake_query_embedder,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual([item["semantic_rank"] for item in candidates], [1, 2])
        self.assertEqual([item["chunk_id"] for item in candidates], ["c1", "c2"])
        self.assertLessEqual(candidates[0]["semantic_distance"], candidates[1]["semantic_distance"])

    def test_semantic_candidate_metadata_complete(self):
        storage = self.index_chunks([base_chunk("c1", "Điều 7", page_start=2, page_end=4)])
        candidate = advanced_rag.semantic_candidates(
            "Điều 7",
            candidate_k=5,
            storage_path=storage,
            query_embedder=fake_query_embedder,
        )[0]
        self.assertEqual(
            set(candidate),
            {"chunk_id", "text", "source", "page_start", "page_end", "semantic_rank", "semantic_distance"},
        )
        self.assertEqual(candidate["source"], "fixture.pdf")
        self.assertEqual(candidate["page_start"], 2)
        self.assertEqual(candidate["page_end"], 4)

    def test_collection_mismatch_is_blocked(self):
        import warnings

        import chromadb

        warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb.*")
        storage = self.tmp_path / "chroma_mismatch"
        client = chromadb.PersistentClient(path=str(storage))
        name = rag.make_collection_name("hierarchical", TEST_CONFIG["embedding_model"], TEST_CONFIG["embedding_dim"])
        client.create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"strategy": "wrong"},
            embedding_function=None,
        )
        with self.assertRaisesRegex(ValueError, "không khớp"):
            advanced_rag.semantic_candidates(
                "Điều 7",
                candidate_k=1,
                storage_path=storage,
                query_embedder=fake_query_embedder,
            )

    def test_status_does_not_create_collection(self):
        storage = self.tmp_path / "missing_chroma"
        input_path = self.tmp_path / "chunks.json"
        import json
        input_path.write_text(json.dumps([base_chunk("c1", "Điều 7")], ensure_ascii=False), encoding="utf-8")
        env = {
            "GEMINI_API_KEY": "",
            "GEMINI_EMBEDDING_MODEL": TEST_CONFIG["embedding_model"],
            "GEMINI_EMBEDDING_DIM": str(TEST_CONFIG["embedding_dim"]),
            "GEMINI_GENERATION_MODEL": TEST_CONFIG["generation_model"],
            "DEFAULT_TOP_K": "5",
            "RAG_MAX_DISTANCE": "0.45",
            "BM25_CANDIDATES": "20",
            "SEMANTIC_CANDIDATES": "20",
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
        self.config_patch.stop()
        try:
            with mock.patch.dict("os.environ", env, clear=True):
                status = advanced_rag.advanced_status(input_path=input_path, storage_path=storage)
        finally:
            self.config_patch.start()
        self.assertTrue(status["bm25_ready"])
        self.assertFalse(status["collection_exists"])
        self.assertEqual(status["collection_count"], 0)
        self.assertFalse(storage.exists())

    def test_missing_api_key_does_not_use_fake_vector(self):
        self.config_patch.stop()
        try:
            with mock.patch.object(rag, "load_config", return_value={**TEST_CONFIG, "api_key": "", "api_key_status": "Thiếu"}):
                called = []
                def fake_embedder(chunk, config):
                    called.append(chunk)
                    return vector()
                with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                    rag.index_chunks(
                        input_path=self.tmp_path / "missing.json",
                        storage_path=self.tmp_path / "chroma",
                        embedder=fake_embedder,
                    )
                self.assertEqual(called, [])
        finally:
            self.config_patch.start()

    def test_semantic_candidates_do_not_call_generation(self):
        storage = self.index_chunks([base_chunk("c1", "Điều 7")])
        with mock.patch.object(rag, "gemini_generate_answer", side_effect=AssertionError("generation must not run")):
            candidates = advanced_rag.semantic_candidates(
                "Điều 7",
                candidate_k=1,
                storage_path=storage,
                query_embedder=fake_query_embedder,
            )
        self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()
