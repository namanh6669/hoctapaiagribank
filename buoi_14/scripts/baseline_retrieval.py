"""Baseline retrieval: chạy BM25 và Dense trên cùng một câu hỏi.

Cú pháp:
    python scripts/baseline_retrieval.py --query "..." --top-k 5
    python scripts/baseline_retrieval.py --query "..." --top-k 5 --csv path/to/chunks.csv
    python scripts/baseline_retrieval.py --query "..." --top-k 5 --no-dense
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép `from src...` khi chạy script trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_corpus, results_to_table
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline BM25 + Dense retrieval")
    ap.add_argument("--query", required=True, help="câu hỏi tiếng Việt")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--csv", default=None, help="đường dẫn corpus (mặc định data/processed/chunks_normalized.csv)")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()

    df = load_corpus(args.csv)
    print(f"[load] {len(df)} chunks")

    bm_results: list[dict] = []
    dense_results: list[dict] = []

    if not args.no_bm25:
        print("\n[bm25] building...")
        bm25 = BM25Retriever(df)
        bm_results = bm25.search(args.query, args.top_k)

    if not args.no_dense:
        print("\n[dense] building/loading...")
        dense = DenseRetriever(df)
        dense_results = dense.search(args.query, args.top_k)

    print("\n" + "=" * 72)
    print(f"QUERY: {args.query}")
    print("=" * 72)
    print()
    print("BM25 RESULTS")
    print("-" * 72)
    print(results_to_table(bm_results))
    print()
    print("DENSE RESULTS")
    print("-" * 72)
    print(results_to_table(dense_results))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
