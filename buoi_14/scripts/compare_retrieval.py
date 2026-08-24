"""So sánh 4 method retrieval trên bộ câu hỏi vàng, sinh
outputs/retrieval_comparison.csv và outputs/evaluation_report.md.

Đánh giá:
- Cùng corpus (data/processed/chunks_normalized.csv).
- Cùng bộ câu hỏi (data/eval/questions.csv) — gold chunk_id đã xác minh.
- Cùng protocol (cùng top_k, candidate_k).
- KHÔNG thay gold, KHÔNG bỏ query thất bại — ghi nhận lỗi.

Methods:
- BM25-only
- Dense-only (multilingual-e5-base)
- Hybrid (BM25 + Dense, RRF k=60)
- Hybrid + Rerank (Cross-Encoder BAAI/bge-reranker-v2-m3)

Metric: Hit@1, Hit@3, Hit@5, MRR.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import PROJECT_ROOT, load_corpus
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import RerankPipeline

METHODS = ["bm25", "dense", "hybrid", "rerank"]
METHOD_LABELS = {
    "bm25": "BM25-only",
    "dense": "Dense-only",
    "hybrid": "Hybrid (RRF)",
    "rerank": "Hybrid + Rerank (BGE-reranker-v2-m3)",
}


# ---------- metric helpers ----------
def _first_hit_rank(results, gold):
    if not gold or not results:
        return None
    for i, r in enumerate(results, 1):
        if str(r.get("chunk_id", "")).strip() == gold:
            return i
    return None


def _hit_at_k(rank, k):
    return 1 if rank is not None and rank <= k else 0


def _mrr(rank):
    return (1.0 / rank) if rank else 0.0


def _aggregate(rows, top_k):
    """rows: list of (rank | None) per question. Trả về dict metric."""
    if not rows:
        return {"Hit@1": 0.0, "Hit@3": 0.0, "Hit@5": 0.0, "MRR": 0.0, "n": 0}
    n = len(rows)
    return {
        "Hit@1": sum(_hit_at_k(r, 1) for r in rows) / n,
        "Hit@3": sum(_hit_at_k(r, 3) for r in rows) / n,
        "Hit@5": sum(_hit_at_k(r, top_k) for r in rows) / n,
        "MRR": sum(_mrr(r) for r in rows) / n,
        "n": n,
    }


# ---------- runner ----------
def _safe(method_obj_method, query, top_k, candidate_k=None, *, label=""):
    try:
        if candidate_k is None:
            return method_obj_method(query, top_k), None
        return method_obj_method(query, top_k, candidate_k), None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".strip()
        print(f"[ERR {label}] {msg}")
        return [], msg


def run(df, qs, top_k, candidate_k):
    print(f"[load] {len(df)} chunks, {len(qs)} questions, top_k={top_k}, candidate_k={candidate_k}")

    print("[bm25] building...")
    bm25 = BM25Retriever(df)
    print("[dense] building/loading...")
    dense = DenseRetriever(df)
    print("[hybrid] building...")
    hybrid = HybridRetriever(df)
    print("[rerank] pipeline (Hybrid → Cross-Encoder)...")
    rerank = RerankPipeline(df)
    print(f"[rerank] method: {rerank.reranker.name}  is_fallback={rerank.reranker.is_fallback}")

    per_q: list[dict] = []
    for _, q in qs.iterrows():
        qid = q["question_id"]
        qtype = q["query_type"]
        text = q["question"]
        gold = str(q["expected_chunk_id"]).strip()

        print(f"  [{qid}] ({qtype}) {text}")

        bm_results, bm_err = _safe(bm25.search, text, top_k, label=f"bm25:{qid}")
        dn_results, dn_err = _safe(dense.search, text, top_k, label=f"dense:{qid}")
        hy_results, hy_err = _safe(hybrid.search, text, top_k, candidate_k, label=f"hybrid:{qid}")
        # Rerank pipeline: returns (candidates, reranked, meta)
        try:
            _cands, reranked, rmeta = rerank.search(text, top_k=top_k, candidate_k=candidate_k)
            rk_err = None
        except Exception as e:
            print(f"[ERR rerank:{qid}] {e}")
            reranked = []
            rmeta = {"method": "?", "is_fallback": True}
            rk_err = f"{type(e).__name__}: {e}"

        per_q.append(
            {
                "question_id": qid,
                "query_type": qtype,
                "question": text,
                "expected_chunk_id": gold,
                "bm25_results": bm_results,
                "dense_results": dn_results,
                "hybrid_results": hy_results,
                "rerank_results": reranked,
                "rerank_meta": rmeta,
                "errors": {
                    "bm25": bm_err,
                    "dense": dn_err,
                    "hybrid": hy_err,
                    "rerank": rk_err,
                },
            }
        )

    return per_q, rmeta


# ---------- analysis ----------
def write_comparison_csv(per_q, path, top_k):
    """Per-question CSV: method × rank + hit indicators."""
    rows = []
    for pq in per_q:
        out = {
            "question_id": pq["question_id"],
            "query_type": pq["query_type"],
            "question": pq["question"],
            "expected_chunk_id": pq["expected_chunk_id"],
        }
        for m in METHODS:
            results = pq[f"{m}_results"]
            rank = _first_hit_rank(results, pq["expected_chunk_id"])
            out[f"{m}_rank"] = rank if rank is not None else ""
            out[f"{m}_hit_at_1"] = _hit_at_k(rank, 1)
            out[f"{m}_hit_at_3"] = _hit_at_k(rank, 3)
            out[f"{m}_hit_at_5"] = _hit_at_k(rank, top_k)
            out[f"{m}_mrr"] = f"{_mrr(rank):.4f}" if rank else "0"
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def aggregate_by_method(per_q, top_k):
    out = {}
    for m in METHODS:
        ranks = [_first_hit_rank(pq[f"{m}_results"], pq["expected_chunk_id"]) for pq in per_q]
        out[m] = _aggregate(ranks, top_k)
    return out


def aggregate_by_type_and_method(per_q, top_k):
    out: dict[str, dict[str, dict]] = {}
    types = sorted({pq["query_type"] for pq in per_q})
    for qt in types:
        subset = [pq for pq in per_q if pq["query_type"] == qt]
        out[qt] = {}
        for m in METHODS:
            ranks = [_first_hit_rank(pq[f"{m}_results"], pq["expected_chunk_id"]) for pq in subset]
            out[qt][m] = _aggregate(ranks, top_k)
    return out


def analysis_section(per_q, agg_total, agg_by_type, rerank_meta, top_k):
    """Sinh các đoạn Markdown phân tích theo yêu cầu."""
    md = []
    # BM25 mạnh ở đâu
    md.append("### BM25 mạnh ở đâu?\n")
    by_type = agg_by_type
    bm_exact = by_type.get("EXACT_KEYWORD", {}).get("bm25", {})
    bm_sem = by_type.get("SEMANTIC", {}).get("bm25", {})
    bm_mix = by_type.get("MIXED", {}).get("bm25", {})
    md.append(
        f"- EXACT_KEYWORD : Hit@5 = {bm_exact.get('Hit@5', 0):.2f} · MRR = {bm_exact.get('MRR', 0):.2f} (n={bm_exact.get('n', 0)})"
    )
    md.append(
        f"- SEMANTIC      : Hit@5 = {bm_sem.get('Hit@5', 0):.2f} · MRR = {bm_sem.get('MRR', 0):.2f} (n={bm_sem.get('n', 0)})"
    )
    md.append(
        f"- MIXED         : Hit@5 = {bm_mix.get('Hit@5', 0):.2f} · MRR = {bm_mix.get('MRR', 0):.2f} (n={bm_mix.get('n', 0)})"
    )
    md.append("")

    # Dense mạnh ở đâu
    md.append("### Dense mạnh ở đâu?\n")
    dn_exact = by_type.get("EXACT_KEYWORD", {}).get("dense", {})
    dn_sem = by_type.get("SEMANTIC", {}).get("dense", {})
    dn_mix = by_type.get("MIXED", {}).get("dense", {})
    md.append(
        f"- EXACT_KEYWORD : Hit@5 = {dn_exact.get('Hit@5', 0):.2f} · MRR = {dn_exact.get('MRR', 0):.2f} (n={dn_exact.get('n', 0)})"
    )
    md.append(
        f"- SEMANTIC      : Hit@5 = {dn_sem.get('Hit@5', 0):.2f} · MRR = {dn_sem.get('MRR', 0):.2f} (n={dn_sem.get('n', 0)})"
    )
    md.append(
        f"- MIXED         : Hit@5 = {dn_mix.get('Hit@5', 0):.2f} · MRR = {dn_mix.get('MRR', 0):.2f} (n={dn_mix.get('n', 0)})"
    )
    md.append("")

    # Hybrid có giúp không
    md.append("### Hybrid (BM25 + Dense) có giúp không?\n")
    md.append(
        "So sánh metric Hybrid với max(BM25, Dense) điểm MRR theo từng câu:"
    )
    wins_h = ties_h = losses_h = 0
    for pq in per_q:
        r_bm = _first_hit_rank(pq["bm25_results"], pq["expected_chunk_id"])
        r_dn = _first_hit_rank(pq["dense_results"], pq["expected_chunk_id"])
        r_hy = _first_hit_rank(pq["hybrid_results"], pq["expected_chunk_id"])
        m_bm = _mrr(r_bm); m_dn = _mrr(r_dn); m_hy = _mrr(r_hy)
        m_best_single = max(m_bm, m_dn)
        if m_hy > m_best_single: wins_h += 1
        elif m_hy == m_best_single: ties_h += 1
        else: losses_h += 1
    n = len(per_q)
    md.append(f"- Hybrid tốt hơn best-of-single: {wins_h}/{n}")
    md.append(f"- Hybrid ngang best-of-single : {ties_h}/{n}")
    md.append(f"- Hybrid thua best-of-single  : {losses_h}/{n}")
    md.append("")

    # Rerank có đổi không
    md.append("### Rerank có đổi ranking không?\n")
    if rerank_meta.get("is_fallback"):
        md.append(f"- ⚠️  FALLBACK reranker đã chạy: `{rerank_meta.get('method')}` — giữ nguyên hybrid.")
    else:
        md.append(f"- Rerank method: `{rerank_meta.get('method')}`")
    changed = 0
    reorder = 0
    for pq in per_q:
        r_hy = _first_hit_rank(pq["hybrid_results"], pq["expected_chunk_id"])
        r_rk = _first_hit_rank(pq["rerank_results"], pq["expected_chunk_id"])
        if r_hy is None and r_rk is None:
            continue
        if r_hy != r_rk:
            changed += 1
        # reorder: rerank top-1 chunk_id khác hybrid top-1
        hy_top1 = pq["hybrid_results"][0]["chunk_id"] if pq["hybrid_results"] else None
        rk_top1 = pq["rerank_results"][0]["chunk_id"] if pq["rerank_results"] else None
        if hy_top1 != rk_top1:
            reorder += 1
    md.append(f"- Số câu mà rerank đổi vị trí gold (so với hybrid): {changed}/{n}")
    md.append(f"- Số câu mà rerank thay đổi top-1 (so với hybrid)   : {reorder}/{n}")
    md.append("")

    # Failure cases
    md.append("### Failure cases\n")
    fail_all = []
    for pq in per_q:
        found_any = any(
            _first_hit_rank(pq[f"{m}_results"], pq["expected_chunk_id"]) is not None
            for m in METHODS
        )
        if not found_any:
            fail_all.append(pq)
    md.append(f"- Câu mà cả 4 method đều không tìm thấy gold trong top-{top_k}: {len(fail_all)}/{n}")
    if fail_all:
        for pq in fail_all:
            md.append(f"  - `{pq['question_id']}` ({pq['query_type']}): {pq['question']}")
    # Câu mà chỉ 1 method tìm ra
    md.append("")
    md.append("Chi tiết từng câu:\n")
    for pq in per_q:
        marks = []
        for m in METHODS:
            r = _first_hit_rank(pq[f"{m}_results"], pq["expected_chunk_id"])
            marks.append(f"{m}={r if r else '—'}")
        md.append(f"- `{pq['question_id']}` ({pq['query_type']}): " + ", ".join(marks))

    # Per-method error log
    err_lines = []
    for pq in per_q:
        for m, msg in pq["errors"].items():
            if msg:
                err_lines.append(f"  - `{pq['question_id']}` [{m}]: {msg}")
    if err_lines:
        md.append("")
        md.append("### Lỗi trong quá trình chạy\n")
        md.extend(err_lines)
    else:
        md.append("")
        md.append("### Lỗi trong quá trình chạy\n")
        md.append("- Không có lỗi nào.")

    return md


def write_evaluation_report(per_q, agg_total, agg_by_type, rerank_meta, top_k, path):
    md: list[str] = []
    md.append(f"# Buổi 14 — Evaluation Report\n")
    md.append(f"_Sinh tự động bằng `scripts/compare_retrieval.py`, top_k={top_k}._\n")
    md.append(
        f"- Số câu hỏi: **{len(per_q)}** ({', '.join(sorted({pq['query_type'] for pq in per_q}))})\n"
        f"- Methods so sánh: {', '.join(METHODS)}\n"
        f"- Corpus: `data/processed/chunks_normalized.csv` (chunks=1463)\n"
    )
    md.append(f"- Rerank method: `{rerank_meta.get('method')}` (is_fallback={rerank_meta.get('is_fallback')})\n")
    md.append("")
    md.append("## Metric tổng hợp\n")
    md.append("| Method | Hit@1 | Hit@3 | Hit@5 | MRR | n |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for m in METHODS:
        a = agg_total[m]
        md.append(
            f"| {METHOD_LABELS[m]} | {a['Hit@1']:.2f} | {a['Hit@3']:.2f} | "
            f"{a['Hit@5']:.2f} | {a['MRR']:.2f} | {a['n']} |"
        )
    md.append("")

    # Per query_type
    md.append("## Metric theo query_type\n")
    md.append("| Method | EXACT_KEYWORD | SEMANTIC | MIXED |")
    md.append("|---|---:|---:|---:|")
    for m in METHODS:
        cells = []
        for qt in ["EXACT_KEYWORD", "SEMANTIC", "MIXED"]:
            a = agg_by_type.get(qt, {}).get(m, {})
            if not a or a["n"] == 0:
                cells.append("—")
            else:
                cells.append(f"H@5={a['Hit@5']:.2f} MRR={a['MRR']:.2f} (n={a['n']})")
        md.append(f"| {METHOD_LABELS[m]} | {' · '.join(cells) if False else cells[0]} | {cells[1]} | {cells[2]} |")
    md.append("")
    md.append("**Cột trên 1 method 1 dòng trống — bảng dưới tách riêng từng metric để dễ đọc:**\n")
    for qt in ["EXACT_KEYWORD", "SEMANTIC", "MIXED"]:
        md.append(f"\n### {qt}\n")
        md.append("| Method | Hit@1 | Hit@3 | Hit@5 | MRR | n |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for m in METHODS:
            a = agg_by_type.get(qt, {}).get(m, {})
            md.append(
                f"| {METHOD_LABELS[m]} | {a.get('Hit@1', 0):.2f} | {a.get('Hit@3', 0):.2f} | "
                f"{a.get('Hit@5', 0):.2f} | {a.get('MRR', 0):.2f} | {a.get('n', 0)} |"
            )
    md.append("")

    # Per-question detail
    md.append("## Per-question (rank của gold trong top-5)\n")
    md.append("| question_id | type | bm25 | dense | hybrid | rerank |")
    md.append("|---|---|---:|---:|---:|---:|")
    for pq in per_q:
        cells = [pq["question_id"], pq["query_type"]]
        for m in METHODS:
            r = _first_hit_rank(pq[f"{m}_results"], pq["expected_chunk_id"])
            cells.append(str(r) if r is not None else "—")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Analysis sections
    md.extend(analysis_section(per_q, agg_total, agg_by_type, rerank_meta, top_k))

    md.append("\n## Kết luận có giới hạn\n")
    md.append(
        "- Bộ câu hỏi chỉ có **8 câu** — sai số thống kê lớn. Không nên kết luận quá mạnh.\n"
        "- Gold là chunk cụ thể, không phải toàn bộ document liên quan — một số câu có nhiều chunk đúng (ví dụ Q03 có thể ghi nhận preamble 41/2016 hoặc các chunk khác cùng doc). Đánh giá này đo *chunk-level exact hit*, không đo *document-level recall*.\n"
        "- Không đo latency/throughput; chỉ đo chất lượng ranking.\n"
        "- Reranker chỉ rerank top-candidate_k của Hybrid — nếu Hybrid miss, Rerank cũng miss (không thể phục hồi).\n"
        "- Văn bản pháp luật VN dài (max chunk ~58 KB); truncate ở 512 token khi encode dense/rerank có thể bỏ phần đầu/cuối chunk.\n"
        "- Rerank nặng (Cross-Encoder ~2.3 GB), nếu OOM hoặc torch lỗi sẽ tự động rơi vào FALLBACK — phần rerank trong bảng trên sẽ không phản ánh đúng neural rerank, mà chỉ là hybrid ordering.\n"
    )

    Path(path).write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval/questions.csv")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--csv-out", default=str(PROJECT_ROOT / "outputs" / "retrieval_comparison.csv"))
    ap.add_argument("--md-out", default=str(PROJECT_ROOT / "outputs" / "evaluation_report.md"))
    args = ap.parse_args()

    df = load_corpus()
    qs = pd.read_csv(PROJECT_ROOT / args.questions, dtype=str, keep_default_na=False)

    t0 = time.time()
    per_q, rerank_meta = run(df, qs, top_k=args.top_k, candidate_k=args.candidate_k)
    print(f"\n[run] all {len(qs)} queries done in {time.time()-t0:.1f}s")

    # Aggregate
    agg_total = aggregate_by_method(per_q, args.top_k)
    agg_by_type = aggregate_by_type_and_method(per_q, args.top_k)

    # Output CSV
    csv_df = write_comparison_csv(per_q, args.csv_out, args.top_k)
    print(f"[write] {args.csv_out}  ({len(csv_df)} rows)")

    # Output Markdown
    write_evaluation_report(per_q, agg_total, agg_by_type, rerank_meta, args.top_k, args.md_out)
    print(f"[write] {args.md_out}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("TỔNG HỢP")
    print("=" * 60)
    for m in METHODS:
        a = agg_total[m]
        print(f"  {METHOD_LABELS[m]:40s}  Hit@5={a['Hit@5']:.2f}  MRR={a['MRR']:.2f}  (n={a['n']})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
