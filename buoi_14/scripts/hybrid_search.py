"""Hybrid search CLI: BM25 + Dense + RRF trên cùng corpus.

Usage:
    python scripts/hybrid_search.py --query "..." --top-k 5 --candidate-k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_corpus
from src.hybrid_retriever import HybridRetriever


def _print_table(query: str, results: list[dict]) -> None:
    print()
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    print("HYBRID RESULTS")
    print("-" * 80)
    if not results:
        print("(no results)")
        return
    print(
        f"{'Rank':>4s}  {'Chunk':10s}  {'BM25':>5s}  {'Dense':>5s}  "
        f"{'RRF':>8s}  {'Citation':<60s}"
    )
    print("-" * 80)
    for r in results:
        cid_short = r["chunk_id"][:8]
        bmr = "—" if r["bm25_rank"] is None else str(r["bm25_rank"])
        dnr = "—" if r["dense_rank"] is None else str(r["dense_rank"])
        cite = r["citation"]
        if len(cite) > 60:
            cite = cite[:57] + "…"
        print(
            f"{r['final_rank']:>4d}  {cid_short:10s}  {bmr:>5s}  {dnr:>5s}  "
            f"{r['rrf_score']:>8.6f}  {cite}"
        )
    # In text snippet cho mỗi hit
    print()
    for r in results:
        snippet = (r["text"] or "")[:180].replace("\n", " ")
        print(f"  r{r['final_rank']} → {snippet}…")


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid BM25 + Dense retrieval via RRF")
    ap.add_argument("--query", required=True, help="câu hỏi tiếng Việt")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--k-rrf", type=int, default=60, help="hằng số RRF (mặc định 60)")
    args = ap.parse_args()

    df = load_corpus(args.csv)
    print(f"[load] {len(df)} chunks")

    print("[hybrid] building (BM25 + Dense on the same corpus)...")
    hyb = HybridRetriever(df, k_rrf=args.k_rrf)

    results = hyb.search(args.query, top_k=args.top_k, candidate_k=args.candidate_k)
    _print_table(args.query, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
