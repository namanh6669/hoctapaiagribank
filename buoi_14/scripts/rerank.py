"""CLI rerank: lấy candidate từ Hybrid Retriever rồi rerank qua Cross-Encoder.

Usage:
    python scripts/rerank.py --query "..." --candidate-k 20 --top-k 5
    python scripts/rerank.py --query "..." --fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_corpus
from src.reranker import RerankPipeline, DEFAULT_RERANK_MODEL


def _print_table(title: str, results: list[dict], candidates: list[dict] | None = None) -> None:
    print()
    print(title)
    print("-" * 80)
    if not results:
        print("(no results)")
        return
    print(
        f"{'rank':>4s}  {'chunk':>8s}  {'hybrid':>6s}  {'rerank':>9s}  "
        f"{'Δ':>3s}  citation"
    )
    print("-" * 80)
    for r in results:
        cid = r["chunk_id"][:8]
        hr = r.get("hybrid_rank") or "—"
        rr = r.get("rerank_score")
        rr_s = "—" if rr is None else f"{rr:+.3f}"
        try:
            delta = int(hr) - int(r["final_rank"])
            delta_s = f"{delta:+d}" if delta else " 0"
        except Exception:
            delta_s = "—"
        cite = r["citation"]
        if len(cite) > 60:
            cite = cite[:57] + "…"
        print(f"{r['final_rank']:>4d}  {cid:>8s}  {str(hr):>6s}  {rr_s:>9s}  {delta_s:>3s}  {cite}")
    # In text snippet
    for r in results:
        snippet = (r["text"] or "")[:160].replace("\n", " ")
        print(f"  r{r['final_rank']} → {snippet}…")


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid → Cross-Encoder rerank")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--csv", default=None)
    ap.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help=f"tên model HF (mặc định {DEFAULT_RERANK_MODEL})",
    )
    ap.add_argument(
        "--fallback",
        action="store_true",
        help="ép dùng FallbackReranker (không tải model neural)",
    )
    args = ap.parse_args()

    df = load_corpus(args.csv)
    print(f"[load] {len(df)} chunks")

    print(f"[pipeline] hybrid + rerank (candidate_k={args.candidate_k}, top_k={args.top_k}) ...")
    pipe = RerankPipeline(df, rerank_model=args.rerank_model, force_fallback=args.fallback)

    candidates, reranked, meta = pipe.search(args.query, top_k=args.top_k, candidate_k=args.candidate_k)

    print()
    print("=" * 80)
    print(f"QUERY: {args.query}")
    print(f"rerank method: {meta['method']}  (is_fallback={meta['is_fallback']})")
    print("=" * 80)

    # Trước rerank: hybrid (candidate_k đầy đủ, lấy top_k để so sánh).
    # Mask rerank_score (reranker đã mutate lên candidates) — tránh in nhầm.
    before_k = []
    for c in candidates[: args.top_k]:
        bc = dict(c)
        bc["rerank_score"] = None
        before_k.append(bc)

    _print_table(
        f"BEFORE RERANK  (top-{args.top_k} của {len(candidates)} candidate từ Hybrid)",
        before_k,
    )

    _print_table(
        f"AFTER RERANK   ({meta['method']})",
        reranked,
    )

    # Tóm tắt thay đổi
    before_ids = [c["chunk_id"] for c in before_k]
    after_ids = [r["chunk_id"] for r in reranked]
    promoted = [r for r in reranked if r.get("hybrid_rank") and r["hybrid_rank"] > r["final_rank"]]
    demoted = [r for r in reranked if r.get("hybrid_rank") and r["hybrid_rank"] < r["final_rank"]]
    new_top = [cid for cid in after_ids if cid not in set(before_ids)]
    print()
    print(f"Thay đổi: {len(promoted)} promoted, {len(demoted)} demoted, "
          f"{len(new_top)} mới vào top-{args.top_k} sau rerank.")
    if meta["is_fallback"]:
        print()
        print("*** FALLBACK reranker đã chạy — KHÔNG có model neural reranker nào tham gia. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
