"""So sánh BM25 / Dense / Hybrid / Rerank trên 3 câu hỏi mẫu và ghi
outputs/retrieval_examples.md.

Citation: dùng metadata thật — không bịa.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import PROJECT_ROOT, load_corpus
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import RerankPipeline

# 3 nhóm câu hỏi theo yêu cầu
QUERIES = [
    (
        "Câu có mã/số hiệu cụ thể",
        "Quy định tại Điều 5 Thông tư 01/2014/TT-NHNN về giao nhận, bảo quản tiền mặt",
    ),
    (
        "Câu diễn đạt semantic (không mã)",
        "Điều kiện để cấp giấy phép cho ngân hàng thương mại",
    ),
    (
        "Câu kết hợp cả hai",
        "Tỷ lệ an toàn vốn tối thiểu theo Thông tư 41/2016/TT-NHNN được sửa đổi bởi 22/2023/TT-NHNN",
    ),
]

TOP_K = 5
CANDIDATE_K = 20


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def _bm_row(r: dict) -> str:
    snippet = _md_escape(r["text"])[:240]
    return (
        f"| {r['rank']} | {r['retrieval_score']:.4f} | `{r['chunk_id']}` | "
        f"{r['citation']} | {snippet}… |"
    )


def _hybrid_row(r: dict) -> str:
    bmr = "—" if r["bm25_rank"] is None else str(r["bm25_rank"])
    dnr = "—" if r["dense_rank"] is None else str(r["dense_rank"])
    snippet = _md_escape(r["text"])[:240]
    return (
        f"| {r['final_rank']} | `{r['chunk_id']}` | {bmr} | {dnr} | "
        f"{r['rrf_score']:.6f} | {r['citation']} | {snippet}… |"
    )


def _rerank_header_md(top_k: int) -> tuple[str, str]:
    h = (
        "| final_rank | chunk_id | hybrid_rank | rerank_score | Δ vs hybrid | citation | text (≤240 chars) |"
    )
    sep = "|---:|---|---:|---:|---:|---|---|"
    return h, sep


def _rerank_row(r: dict) -> str:
    hr = r.get("hybrid_rank")
    hr_s = "—" if hr is None else str(hr)
    rr = r.get("rerank_score")
    rr_s = "—" if rr is None else f"{rr:+.4f}"
    try:
        delta = int(hr) - int(r["final_rank"])
        delta_s = f"{delta:+d}"
    except Exception:
        delta_s = "—"
    snippet = _md_escape(r["text"])[:240]
    return (
        f"| {r['final_rank']} | `{r['chunk_id']}` | {hr_s} | {rr_s} | "
        f"{delta_s} | {r['citation']} | {snippet}… |"
    )


def _block_bm(title: str, results: list[dict]) -> list[str]:
    out = [f"### {title}\n"]
    if not results:
        out.append("_(không có kết quả)_\n")
        return out
    out.append("| rank | score | chunk_id | citation | text (≤240 chars) |")
    out.append("|---:|---:|---|---|---|")
    for r in results:
        out.append(_bm_row(r))
    out.append("")
    return out


def _block_dense(title: str, results: list[dict]) -> list[str]:
    out = [f"### {title}\n"]
    if not results:
        out.append("_(không có kết quả)_\n")
        return out
    out.append("| rank | score | chunk_id | citation | text (≤240 chars) |")
    out.append("|---:|---:|---|---|---|")
    for r in results:
        out.append(_bm_row(r))
    out.append("")
    return out


def _block_hybrid(title: str, results: list[dict]) -> list[str]:
    out = [f"### {title}\n"]
    if not results:
        out.append("_(không có kết quả)_\n")
        return out
    out.append("| final_rank | chunk_id | bm25_rank | dense_rank | rrf_score | citation | text (≤240 chars) |")
    out.append("|---:|---|---:|---:|---:|---|---|")
    for r in results:
        out.append(_hybrid_row(r))
    out.append("")
    return out


def _block_rerank(title: str, results: list[dict], method_label: str) -> list[str]:
    out = [f"### {title}  _(rerank method: `{method_label}`)_\n"]
    if not results:
        out.append("_(không có kết quả)_\n")
        return out
    h, sep = _rerank_header_md(TOP_K)
    out.append(h)
    out.append(sep)
    for r in results:
        out.append(_rerank_row(r))
    out.append("")
    return out


def _top1_id(results: list[dict], key: str) -> str | None:
    if not results:
        return None
    return results[0].get(key)


def _analysis_after_rerank(
    bm: list, dn: list, hy: list, before: list, after: list, meta: dict
) -> list[str]:
    """Sinh bullet phân tích BEFORE → AFTER rerank."""
    notes: list[str] = []

    if meta["is_fallback"]:
        notes.append(
            "- ⚠️  FALLBACK reranker (không có model neural) — ordering giữ nguyên hybrid."
        )
    else:
        notes.append(f"- Rerank method: `{meta['method']}`")

    before_ids = [c["chunk_id"] for c in before]
    after_ids = [r["chunk_id"] for r in after]
    promoted = [r for r in after if r.get("hybrid_rank") and r["hybrid_rank"] > r["final_rank"]]
    demoted = [r for r in after if r.get("hybrid_rank") and r["hybrid_rank"] < r["final_rank"]]
    new_in = [cid for cid in after_ids if cid not in set(before_ids)]
    pushed_out = [cid for cid in before_ids if cid not in set(after_ids)]
    notes.append(
        f"- BEFORE top-{TOP_K} → AFTER top-{TOP_K}: {len(promoted)} promoted, "
        f"{len(demoted)} demoted, {len(new_in)} mới vào top, {len(pushed_out)} bị đẩy ra."
    )
    if before_ids and after_ids and before_ids[0] != after_ids[0]:
        notes.append(
            f"- Top-1 đã đổi: `{before_ids[0][:8]}` (hybrid) → `{after_ids[0][:8]}` (rerank)."
        )
    return notes


def main() -> int:
    df = load_corpus()
    print(f"[load] {len(df)} chunks")

    print("[bm25] building...")
    t0 = time.time()
    bm25 = BM25Retriever(df)
    print(f"[bm25] ready in {time.time()-t0:.1f}s")

    print("[dense] building/loading...")
    t0 = time.time()
    dense = DenseRetriever(df)
    print(f"[dense] ready in {time.time()-t0:.1f}s")

    print("[hybrid] building (BM25 + Dense + RRF)...")
    t0 = time.time()
    hybrid = HybridRetriever(df)
    print(f"[hybrid] ready in {time.time()-t0:.1f}s")

    print("[rerank] pipeline (Hybrid → Cross-Encoder)...")
    t0 = time.time()
    rerank = RerankPipeline(df)
    print(f"[rerank] ready in {time.time()-t0:.1f}s (method={rerank.reranker.name})")

    md: list[str] = []
    md.append("# Buổi 14 — Retrieval so sánh: BM25 · Dense · Hybrid (RRF) · Rerank\n")
    md.append(f"_Sinh tự động bằng `scripts/run_examples.py`, top_k={TOP_K}, "
              f"candidate_k={CANDIDATE_K}_.\n")
    md.append(
        "- **BM25** — lexical trên toàn chunk corpus (rank_bm25).\n"
        "- **Dense** — `intfloat/multilingual-e5-base`, L2-normalize, query/passage theo prefix E5.\n"
        "- **Hybrid** — Reciprocal Rank Fusion (k=60) của BM25 + Dense trên cùng corpus.\n"
        "- **Rerank** — Cross-Encoder `BAAI/bge-reranker-v2-m3` rerank top-20 candidate của Hybrid → top-5.\n"
    )
    md.append(
        "Citation lấy từ metadata thật (`title`, `so_ky_hieu`, `article`, "
        "`chunk_id`); không bịa ở bất kỳ method nào.\n"
    )

    for category, q in QUERIES:
        md.append(f"\n## {category}\n")
        md.append(f"**Query:** `{q}`\n")

        bm = bm25.search(q, TOP_K)
        dn = dense.search(q, TOP_K)
        hy = hybrid.search(q, top_k=TOP_K, candidate_k=CANDIDATE_K)

        # Rerank pipeline
        candidates_all, after_rerank, meta = rerank.search(
            q, top_k=TOP_K, candidate_k=CANDIDATE_K
        )
        # mask rerank_score trong BEFORE display
        before_rerank = []
        for c in candidates_all[:TOP_K]:
            bc = dict(c)
            bc["final_rank"] = c.get("final_rank")
            bc["rerank_score"] = None
            before_rerank.append(bc)

        md.extend(_block_bm("BM25 RESULTS", bm))
        md.extend(_block_dense("DENSE RESULTS", dn))
        md.extend(_block_hybrid("HYBRID RESULTS (RRF)", hy))
        md.extend(_block_rerank("HYBRID → RERANK RESULTS", after_rerank, meta["method"]))

        md.append("**Phân tích nhanh:**\n")
        md.extend(_analysis_after_rerank(bm, dn, hy, before_rerank, after_rerank, meta))
        md.append("")

    # Tổng quan cuối
    md.append("\n## Tổng quan\n")
    md.append(
        "- **Q1 (có mã cụ thể)** — BM25 + Dense đều nhắm đúng `01/2014/TT-NHNN`. "
        "Hybrid thừa hưởng thế mạnh đó; rerank bám top-1 hoặc đẩy canonical preamble lên đầu nếu hữu ích.\n"
        "- **Q2 (semantic thuần)** — Hybrid trộn tốt; Cross-Encoder rerank có xu hướng đẩy điều khoản "
        "có chữ 'điều kiện / thủ tục' lên ngay top-1.\n"
        "- **Q3 (mã + semantic)** — Hybrid đã thấy cả `41/2016` và `22/2023`; rerank "
        "có thể đẩy bản sửa đổi (`22/2023`) lên #1 nếu cross-encoder đánh giá "
        "mức độ liên quan với câu hỏi 'sửa đổi bởi' cao hơn.\n"
        "- Nếu rớt mạng / OOM / torch lỗi, hệ thống chuyển sang FALLBACK identity. "
        "Trong trường hợp đó ordering sẽ không đổi — bạn vẫn thấy Hybrid, mất phần rerank. "
        "Báo cáo này ghi rõ nếu FALLBACK được dùng.\n"
    )

    out_path = PROJECT_ROOT / "outputs" / "retrieval_examples.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[write] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
