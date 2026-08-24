"""Buổi 14 — Unified retrieval CLI.

Một hàm `retrieve(question, method, top_k)` thống nhất 4 method:
    bm25  | dense  | hybrid  | hybrid_rerank

Mỗi method trả về list các dict có cùng schema:
    rank, chunk_id, document_id, text, score, citation, retrieval_method

Với `hybrid_rerank` nếu Cross-Encoder neural reranker chạy thật (không fallback)
thì kèm thêm:
    hybrid_score (RRF), rerank_score (cross-encoder logits)

CLI usage:
    python scripts/query_demo.py --query "..." --method hybrid_rerank --top-k 5
    python scripts/query_demo.py --query "..." --method hybrid          --top-k 3
    python scripts/query_demo.py --query "..." --method bm25           --top-k 5

Cuối lệnh, in:
    GRAPH HINTS
    - document_id của các chunk retrieved
    - chunk_id
    - relations 1-hop trong Mini KG (CHỈ khi Neo4j sẵn sàng và đã nạp buoi_14)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env nếu có (Neo4j credentials)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except Exception:
    pass

from src.common import PROJECT_ROOT, load_corpus
# IMPORTANT: các retriever nặng (Dense/Hybrid/Rerank) import torch + transformers
# — không eager-load ở top-level để tránh Streamlit LocalSourcesWatcher quét
# transformers.* và lỗi ModuleNotFoundError('torchvision') cho vision modules.
# Lazy-import bên trong retrieve().
from src.bm25_retriever import BM25Retriever

# Suppress transformers' docstring-validation noise — xem app.py.
import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.models").setLevel(logging.CRITICAL)

from src._quiet_transformers import install_quiet_print_once
install_quiet_print_once()

VALID_METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")

# Hằng số Neo4j
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


# ----------------------------------------------------------------- format
def _shape_score(r: dict) -> str:
    s = r.get("score")
    return f"{float(s):+.4f}" if s is not None else "—"


def _chunk_short(cid: str) -> str:
    return cid[:8] if cid else "—"


# ----------------------------------------------------------------- core
def _to_unified(rank: int, row: dict, method: str) -> dict:
    """Chuyển 1 row từ retriever nội bộ -> schema thống nhất."""
    return {
        "rank": int(rank),
        "chunk_id": row.get("chunk_id", ""),
        "document_id": row.get("document_id", ""),
        "text": row.get("text", ""),
        "score": float(row.get("retrieval_score")
                       or row.get("rrf_score")
                       or row.get("rerank_score")
                       or row.get("hybrid_score")
                       or 0.0),
        "citation": row.get("citation", ""),
        "retrieval_method": method,
        # extras cho hybrid / hybrid_rerank
        "hybrid_rrf_score": row.get("rrf_score"),
        "hybrid_bm25_rank": row.get("bm25_rank"),
        "hybrid_dense_rank": row.get("dense_rank"),
        "hybrid_score": row.get("hybrid_score"),
        "hybrid_rank": row.get("hybrid_rank"),
        "rerank_score": row.get("rerank_score"),
        "is_reranker_fallback": row.get("is_fallback"),
    }


def retrieve(
    question: str,
    method: str,
    top_k: int = 5,
    df=None,
    candidate_k: int = 20,
) -> tuple[list[dict], dict]:
    """Unified retrieval. Returns (results, meta).

    `meta` chứa thông tin phụ trợ (vd. label reranker thật, FALLBACK, v.v.).
    """
    meta: dict = {"method_requested": method}
    if not question or not question.strip():
        meta["note"] = "empty query"
        return [], meta

    method = method.strip().lower()
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method: {method!r}. Must be one of {VALID_METHODS}")

    if df is None:
        df = load_corpus()

    if method == "bm25":
        retriever = BM25Retriever(df)
        raw = retriever.search(question, top_k)
        return [_to_unified(r["rank"], r, "bm25") for r in raw], meta

    if method == "dense":
        from src.dense_retriever import DenseRetriever
        retriever = DenseRetriever(df)
        raw = retriever.search(question, top_k)
        return [_to_unified(r["rank"], r, "dense") for r in raw], meta

    if method == "hybrid":
        from src.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(df)
        raw = retriever.search(question, top_k=top_k, candidate_k=candidate_k)
        return [_to_unified(r["final_rank"], r, "hybrid") for r in raw], meta

    if method == "hybrid_rerank":
        from src.reranker import RerankPipeline
        pipe = RerankPipeline(df)
        _cands, reranked, rmeta = pipe.search(
            question, top_k=top_k, candidate_k=candidate_k
        )
        meta["rerank_method"] = rmeta.get("method", "?")
        meta["rerank_is_fallback"] = bool(rmeta.get("is_fallback", False))
        # Schema row từ Reranker: final_rank, chunk_id, document_id,
        #   hybrid_score, hybrid_rank, rerank_score, text, citation, is_fallback
        results = [_to_unified(r["final_rank"], r, "hybrid_rerank") for r in reranked]
        return results, meta

    return [], meta


# ----------------------------------------------------------------- printing
def _print_result_row(r: dict, *, method: str) -> str:
    lines = []
    rno = r["rank"]
    cid = _chunk_short(r["chunk_id"])
    doc = r["document_id"][:18] if r["document_id"] else "—"
    score = _shape_score(r)
    cite = r["citation"]
    if len(cite) > 65:
        cite = cite[:62] + "…"
    snippet = (r.get("text") or "")[:180].replace("\n", " ")
    method_label = method

    header = f"#{rno:<2d}  {cid:>8s}  doc={doc:<18s}  score={score:>9s}  method={method_label}"
    parts = []
    if method == "hybrid":
        # Hiển thị 2 đóng góp + RRF để học viên thấy vì sao RRF hoạt động
        if r.get("hybrid_bm25_rank") is not None:
            parts.append(f"bm25_rank={r['hybrid_bm25_rank']}")
        if r.get("hybrid_dense_rank") is not None:
            parts.append(f"dense_rank={r['hybrid_dense_rank']}")
        if r.get("hybrid_rrf_score") is not None:
            parts.append(f"rrf_score={r['hybrid_rrf_score']:+.4f}")
    elif method == "hybrid_rerank":
        # Hybrid score (RRF) + rerank score (cross-encoder)
        if r.get("hybrid_score") is not None:
            parts.append(f"hybrid_score={r['hybrid_score']:+.4f}")
        if r.get("rerank_score") is not None:
            parts.append(f"rerank_score={r['rerank_score']:+.4f}")
        if r.get("is_reranker_fallback") is True:
            parts.append("FALLBACK-reranker")
    if parts:
        header += "  " + " · ".join(parts)
    lines.append(header)
    lines.append(f"     cite: {cite}")
    lines.append(f"     text: {snippet}…")
    return "\n".join(lines)


def print_results(args, results: list[dict], meta: dict) -> None:
    bar = "=" * 80
    print(bar)
    print(f"  BUỔI 14  ·  UNIFIED RETRIEVAL  ·  method={args.method.upper()}")
    print(f"  query  : {args.query}")
    print(f"  top_k  : {args.top_k}")
    if args.method == "hybrid_rerank":
        rm = meta.get("rerank_method", "?")
        fb = "FALLBACK (no neural)" if meta.get("rerank_is_fallback") else "neural reranker"
        print(f"  rerank : {rm}  ·  {fb}")
    print(bar)
    print()

    if not results:
        print("(no results)\n")
        return

    show_extra = (args.method == "hybrid")
    for r in results:
        print(_print_result_row(r, method=args.method))
        print()


# ----------------------------------------------------------------- GRAPH HINTS
def _neo4j_driver_or_none():
    if not (NEO4J_URI and NEO4J_PASSWORD):
        return None
    try:
        from neo4j import GraphDatabase
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception:
        return None


def _collect_1hop_relations(driver, doc_ids: list[str], database: str) -> list[dict]:
    """Chạy 1-hop MATCH (v:VanBan)-[r]->(n) WHERE v.id IN $ids.
    Chỉ lấy trong lab_session='buoi_14' để tránh đụng các node khác."""
    out: list[dict] = []
    if not doc_ids:
        return out
    query = (
        "MATCH (v:VanBan {lab_session:'buoi_14'})-[r]->(n) "
        "WHERE v.id IN $ids AND r.lab_session='buoi_14' "
        "RETURN v.id AS src, v.so_ky_hieu AS src_so, "
        "       labels(n)[0] AS n_label, "
        "       coalesce(n.so_ky_hieu, n.name, n.text, '(unknown)') AS target, "
        "       type(r) AS rel, "
        "       coalesce(r.confidence, 1.0) AS conf "
        "ORDER BY src_so, rel"
    )
    with driver.session(database=database) as session:
        for row in session.run(query, ids=doc_ids):
            out.append({
                "src_id": row["src"],
                "src_so": row["src_so"] or "—",
                "rel": row["rel"],
                "n_label": row["n_label"] or "—",
                "target": row["target"] or "—",
                "conf": float(row["conf"]) if row["conf"] is not None else 1.0,
            })
    return out


def print_graph_hints(args, results: list[dict]) -> None:
    print()
    print("─" * 80)
    print("  GRAPH HINTS  (chỉ tham khảo; KHÔNG phải Graph RAG đầy đủ)")
    print("─" * 80)
    print()

    doc_ids: list[str] = []
    chunk_ids: list[str] = []
    seen_docs: set[str] = set()
    for r in results:
        cid = r.get("chunk_id", "")
        did = r.get("document_id", "")
        if cid:
            chunk_ids.append(cid)
        if did and did not in seen_docs:
            seen_docs.add(did)
            doc_ids.append(did)

    print("Document IDs (của các chunk retrieved):")
    if not doc_ids:
        print("  (none)")
    else:
        for d in doc_ids:
            print(f"  - {d}")
    print()
    print("Chunk IDs:")
    for c in chunk_ids:
        print(f"  - {_chunk_short(c)}")
    print()

    driver = _neo4j_driver_or_none()
    if driver is None:
        print("Mini KG trong Neo4j:")
        print("  - Không thể kết nối Neo4j (thiếu NEO4J_URI/NEO4J_PASSWORD trong .env)")
        print("    hoặc driver không import được. Bỏ qua 1-hop query.")
        print()
        return

    try:
        rels = _collect_1hop_relations(driver, doc_ids, NEO4J_DATABASE)
        print(f"Direct relations (1-hop, từ VanBan trong Neo4j, lab='buoi_14'):")
        if not rels:
            print("  - Không tìm được edge nào (chưa nạp buổi 14, hoặc doc_ids không trùng node).")
        else:
            for r in rels:
                print(
                    f"  - (VanBan {r['src_so']}) -[:{r['rel']} "
                    f"conf={r['conf']:.2f}]-> ({r['n_label']} {r['target']})"
                )
        print()
    except Exception as e:
        print(f"  - Lỗi khi truy vấn Neo4j: {e}")
    finally:
        driver.close()


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unified retrieval demo (bm25/dense/hybrid/hybrid_rerank)"
    )
    ap.add_argument("--query", required=True, help="câu hỏi tiếng Việt")
    ap.add_argument("--method", required=True, choices=VALID_METHODS,
                    help="retrieval method")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20,
                    help="chỉ dùng cho hybrid/hybrid_rerank")
    args = ap.parse_args()

    df = load_corpus()
    results, meta = retrieve(args.query, args.method, args.top_k,
                             df=df, candidate_k=args.candidate_k)

    print_results(args, results, meta)
    print_graph_hints(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
