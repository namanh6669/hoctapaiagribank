from __future__ import annotations

import sys
import unittest
from pathlib import Path

BUOI_08_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUOI_08_DIR))

import advanced_rag


def chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "strategy": "hierarchical",
        "source": "fixture.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": text,
    }


class TestVietnameseLegalTokenizer(unittest.TestCase):
    def test_tokenizer_keeps_vietnamese_diacritics(self):
        tokens = advanced_rag.tokenize_vi_legal("cơ cấu lại thời hạn trả nợ")
        self.assertIn("cơ", tokens)
        self.assertIn("cấu", tokens)
        self.assertIn("thời", tokens)
        self.assertIn("nợ", tokens)

    def test_tokenizer_keeps_article_clause_numbers(self):
        tokens = advanced_rag.tokenize_vi_legal("Điều 7, Khoản 2")
        for token in ("điều", "7", "khoản", "2"):
            self.assertIn(token, tokens)

    def test_corpus_and_query_use_same_preprocessing(self):
        chunks = [chunk("c1", "Điều 7 quy định cơ cấu lại thời hạn trả nợ")]
        corpus = advanced_rag.build_bm25_corpus(chunks)
        query_tokens = advanced_rag.tokenize_vi_legal("điều 7")
        self.assertEqual(corpus["tokenized_corpus"][0][:2], query_tokens)


class TestBM25Search(unittest.TestCase):
    def test_exact_legal_term_ranks_above_non_keyword_chunk(self):
        chunks = [
            chunk("legal", "Điều 7 Khoản 2 quy định cơ cấu lại thời hạn trả nợ"),
            chunk("other-a", "Hội thao nội bộ có lịch thi đấu bóng đá"),
            chunk("other-b", "Căng tin chuẩn bị nước uống cho vận động viên"),
        ]
        results = advanced_rag.bm25_search("Điều 7 cơ cấu nợ", chunks, candidate_k=3)
        self.assertEqual(results[0]["chunk_id"], "legal")
        self.assertGreater(results[0]["bm25_score"], results[1]["bm25_score"])

    def test_candidate_k_larger_than_corpus_still_runs(self):
        chunks = [chunk("c1", "Điều 7"), chunk("c2", "Khoản 2")]
        results = advanced_rag.bm25_search("Điều", chunks, candidate_k=20)
        self.assertEqual(len(results), 2)

    def test_empty_question_fails(self):
        chunks = [chunk("c1", "Điều 7")]
        with self.assertRaisesRegex(ValueError, "question"):
            advanced_rag.bm25_search(" ,.!? ", chunks, candidate_k=5)

    def test_tie_break_deterministic_by_chunk_id_and_keeps_zero_scores(self):
        chunks = [
            chunk("c", "không liên quan"),
            chunk("a", "ngoài phạm vi"),
            chunk("b", "tài liệu khác"),
        ]
        results = advanced_rag.bm25_search("xyzkhongtrung", chunks, candidate_k=3)
        self.assertEqual([item["chunk_id"] for item in results], ["a", "b", "c"])
        self.assertEqual([item["bm25_score"] for item in results], [0.0, 0.0, 0.0])

    def test_does_not_import_or_call_gemini_chroma_or_reranker(self):
        chunks = [chunk("c1", "Điều 7"), chunk("c2", "ngoài phạm vi")]
        blocked_modules = {
            "chromadb": None,
            "google.genai": None,
            "transformers": None,
            "torch": None,
        }
        from unittest import mock
        with mock.patch.dict(sys.modules, blocked_modules):
            results = advanced_rag.bm25_search("Điều 7", chunks, candidate_k=1)
        self.assertEqual(results[0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
