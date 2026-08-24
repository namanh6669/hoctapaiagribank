from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import hierarchical_rag as hr


class HierarchyBuilderTest(unittest.TestCase):
    def config(self, parent_max_chars: int = 1200) -> hr.HierarchyConfig:
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
            parent_max_chars=parent_max_chars,
            parent_score_child_limit=3,
            parent_rrf_k=60,
            parent_candidates=10,
            final_parent_top_k=3,
            total_context_max_chars=max(parent_max_chars, 16000),
        )

    def record(self, chunk_id: str, text: str, *, source: str = "s.pdf", page: int = 1, structure=None):
        return {
            "chunk_id": chunk_id,
            "strategy": "hierarchical",
            "source": source,
            "page_start": page,
            "page_end": page,
            "text": text,
            "structure": structure,
        }

    def write_records(self, tmp: str, records: list[dict]) -> Path:
        path = Path(tmp) / "chunks.json"
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return path

    def load_resolved(self, records: list[dict]) -> list[hr.ResolvedChild]:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_records(tmp, records)
            raw, _stats = hr.load_hierarchical_children(path)
            return hr.resolve_children(raw)

    def test_metadata_precedence(self):
        children = self.load_resolved([
            self.record("s::hierarchical::0001", "### Điều 8. Heading", structure={"article": "7", "chapter": "I"})
        ])
        self.assertEqual(children[0].resolution_method, "metadata")
        self.assertEqual(children[0].structural_path["article"], "7")
        self.assertTrue(children[0].ambiguous)
        self.assertIn("metadata_heading_conflict", children[0].warnings)

    def test_heading_inferred_at_chunk_start(self):
        children = self.load_resolved([self.record("s::hierarchical::0001", "### Điều 7. Trách nhiệm\nNội dung")])
        self.assertEqual(children[0].resolution_method, "heading_inferred")
        self.assertEqual(children[0].structural_path["article"], "7")

    def test_carry_forward_same_source(self):
        children = self.load_resolved([
            self.record("s::hierarchical::0001", "### Điều 7. Trách nhiệm", source="a.pdf"),
            self.record("s::hierarchical::0002", "1. Nội dung tiếp theo", source="a.pdf"),
        ])
        self.assertEqual(children[1].resolution_method, "carried_forward")
        self.assertEqual(children[1].structural_path["article"], "7")

    def test_no_carry_across_source(self):
        children = self.load_resolved([
            self.record("a::hierarchical::0001", "### Điều 7. A", source="a.pdf"),
            self.record("b::hierarchical::0001", "Nội dung đầu tài liệu B", source="b.pdf"),
        ])
        by_source = {child.source: child for child in children}
        self.assertEqual(by_source["b.pdf"].resolution_method, "document_fallback")
        self.assertIsNone(by_source["b.pdf"].structural_path["article"])

    def test_inline_article_not_mistaken_for_heading(self):
        children = self.load_resolved([
            self.record("s::hierarchical::0001", "Khoản này thực hiện theo Điều 7 của Thông tư này.")
        ])
        self.assertEqual(children[0].resolution_method, "document_fallback")
        self.assertIn("inline_article_reference_ignored", children[0].warnings)

    def test_conflict_sets_ambiguous_warning(self):
        children = self.load_resolved([
            self.record("s::hierarchical::0001", "### Điều 9. Heading", structure={"article": "7"})
        ])
        self.assertTrue(children[0].ambiguous)
        self.assertIn("metadata_heading_conflict", children[0].warnings)

    def test_numeric_chunk_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_records(tmp, [
                self.record("s::hierarchical::10", "### Điều 10. Ten"),
                self.record("s::hierarchical::2", "### Điều 2. Two"),
            ])
            raw, _stats = hr.load_hierarchical_children(path)
            self.assertEqual([child.chunk_id for child in raw], ["s::hierarchical::2", "s::hierarchical::10"])

    def test_stable_parent_id(self):
        records = [
            self.record("s::hierarchical::0001", "### Điều 7. A"),
            self.record("s::hierarchical::0002", "Nội dung A"),
        ]
        children1 = self.load_resolved(records)
        children2 = self.load_resolved(records)
        parents1 = hr.build_parent_documents(children1, config=self.config())
        parents2 = hr.build_parent_documents(children2, config=self.config())
        self.assertEqual([p.parent_id for p in parents1], [p.parent_id for p in parents2])

    def test_parent_split_at_child_boundary(self):
        children = self.load_resolved([
            self.record("s::hierarchical::0001", "### Điều 7. " + "A" * 700),
            self.record("s::hierarchical::0002", "B" * 700),
        ])
        parents = hr.build_parent_documents(children, config=self.config(parent_max_chars=1000))
        self.assertEqual(len(parents), 2)
        self.assertEqual(parents[0].child_ids, ["s::hierarchical::0001"])
        self.assertEqual(parents[1].child_ids, ["s::hierarchical::0002"])

    def test_oversized_child_warning(self):
        children = self.load_resolved([self.record("s::hierarchical::0001", "### Điều 7. " + "A" * 1300)])
        parents = hr.build_parent_documents(children, config=self.config(parent_max_chars=1000))
        self.assertIn("oversized_single_child", parents[0].warnings)

    def test_each_child_exactly_one_parent(self):
        children = self.load_resolved([
            self.record("s::hierarchical::0001", "### Điều 7. A"),
            self.record("s::hierarchical::0002", "Nội dung A"),
            self.record("s::hierarchical::0003", "### Điều 8. B"),
        ])
        parents = hr.build_parent_documents(children, config=self.config())
        assigned = [child_id for parent in parents for child_id in parent.child_ids]
        self.assertEqual(sorted(assigned), sorted(child.child_id for child in children))
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_parent_pages_count_text(self):
        children = self.load_resolved([
            self.record("s::hierarchical::0001", "### Điều 7. A", page=2),
            self.record("s::hierarchical::0002", "Nội dung B", page=3),
        ])
        parents = hr.build_parent_documents(children, config=self.config())
        self.assertEqual(parents[0].page_start, 2)
        self.assertEqual(parents[0].page_end, 3)
        self.assertEqual(parents[0].child_ids, ["s::hierarchical::0001", "s::hierarchical::0002"])
        self.assertEqual(parents[0].text, "### Điều 7. A\n\nNội dung B")
        self.assertEqual(parents[0].char_count, len(parents[0].text))

    def test_atomic_build_and_manifest_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = self.write_records(tmp, [self.record("s::hierarchical::0001", "### Điều 7. A")])
            storage = Path(tmp) / "store"
            result = hr.build_hierarchy_store(input_path, storage_dir=storage, config=self.config())
            self.assertTrue((storage / "children.json").exists())
            self.assertTrue((storage / "parents.json").exists())
            self.assertTrue((storage / "manifest.json").exists())
            manifest = result["manifest"]
            self.assertEqual(manifest["schema_version"], hr.SCHEMA_VERSION)
            self.assertEqual(manifest["input_file_fingerprints"][0]["sha256"], hr._fingerprint_file(input_path)["sha256"])
            self.assertEqual(manifest["counts"]["children"], 1)
            self.assertEqual(manifest["counts"]["parents"], 1)

    def test_status_does_not_create_or_modify_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "missing"
            status = hr.hierarchy_status(storage_dir=storage)
            self.assertFalse(status["exists"])
            self.assertFalse(storage.exists())

            storage.mkdir()
            manifest = storage / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": hr.SCHEMA_VERSION, "counts": {"children": 0}, "warning_counts": {}}), encoding="utf-8")
            before = manifest.stat().st_mtime_ns
            time.sleep(0.01)
            hr.hierarchy_status(storage_dir=storage)
            after = manifest.stat().st_mtime_ns
            self.assertEqual(before, after)


class ConfigValidationTest(unittest.TestCase):
    def test_config_validation_rejects_invalid_parent_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text(
                "\n".join([
                    "GEMINI_EMBEDDING_MODEL=gemini-embedding-2",
                    "GEMINI_EMBEDDING_DIM=768",
                    "GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite",
                    "BM25_CANDIDATES=20",
                    "SEMANTIC_CANDIDATES=20",
                    "RRF_K=60",
                    "RRF_BM25_WEIGHT=1.0",
                    "RRF_SEMANTIC_WEIGHT=1.0",
                    "RERANK_CANDIDATES=20",
                    "FINAL_TOP_K=5",
                    "RERANKER_MODEL=BAAI/bge-reranker-v2-m3",
                    "RERANK_MIN_SCORE=0.5",
                    "RERANK_DEVICE=auto",
                    "MULTI_QUERY_COUNT=3",
                    "MULTI_QUERY_MAX_CHARS=300",
                    "MULTI_QUERY_TEMPERATURE=0.2",
                    "MULTI_QUERY_ORIGINAL_WEIGHT=1.5",
                    "MULTI_QUERY_VARIANT_WEIGHT=1.0",
                    "MULTI_QUERY_RRF_K=60",
                    "PER_QUERY_CANDIDATES=12",
                    "PARENT_MAX_CHARS=6000",
                    "PARENT_SCORE_CHILD_LIMIT=3",
                    "PARENT_RRF_K=60",
                    "PARENT_CANDIDATES=2",
                    "FINAL_PARENT_TOP_K=3",
                    "TOTAL_CONTEXT_MAX_CHARS=16000",
                ]),
                encoding="utf-8",
            )
            old_env = os.environ.copy()
            try:
                for key in list(os.environ):
                    if key.startswith(("GEMINI_", "BM25_", "SEMANTIC_", "RRF_", "RERANK", "FINAL_", "MULTI_", "PER_", "PARENT_", "TOTAL_")):
                        os.environ.pop(key, None)
                with self.assertRaisesRegex(ValueError, "FINAL_PARENT_TOP_K"):
                    hr.load_hierarchy_config(env)
            finally:
                os.environ.clear()
                os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
