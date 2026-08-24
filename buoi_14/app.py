"""Buổi 14 — Streamlit demo cho hệ thống retrieval Hybrid Search + Reranking.

Chạy:    streamlit run app.py
Dừng:    Ctrl+C trong terminal đang chạy

Sử dụng trực tiếp các retriever / pipeline của buổi 14
(src.bm25_retriever / src.dense_retriever / src.hybrid_retriever / src.reranker),
KHÔNG viết lại pipeline khác cho Streamlit.

KHÔNG hardcode credentials Neo4j / API key — đọc qua .env (python-dotenv) hoặc
biến môi trường; nếu thiếu, Graph hints chỉ ghi "(chưa kết nối)".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Đảm bảo `from src...` chạy được dù chạy từ bất kỳ cwd
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env nếu có
try:
    from dotenv import load_dotenv
    _env = PROJECT_ROOT / ".env"
    if _env.exists():
        load_dotenv(_env)
except Exception:
    pass

import pandas as pd
import streamlit as st

# Suppress transformers' docstring-validation noise (`[ERROR] 'foo' is part of
# XKwargs, but not documented`). Đây là log nội bộ của transformers v5+ khi
# nó quét các image processor kwargs của các vision submodule mà ta không dùng.
# Bật 1 lần ở app startup để suppress toàn bộ ERROR+ từ thư viện transformers.
import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.models").setLevel(logging.CRITICAL)

# Patch builtins.print để nuốt noise đến từ transformers' lazy-scan vision
# submodule (transformers.utils.auto_docstring dùng print(), không qua logger).
from src._quiet_transformers import install_quiet_print_once
install_quiet_print_once()

from src.common import PROJECT_ROOT as _PROJECT_ROOT, load_corpus
from src.bm25_retriever import BM25Retriever
# Các retriever nặng (Dense / Hybrid / Rerank) import torch + transformers,
# gây noise từ LocalSourcesWatcher của Streamlit do transformers lazy-scan
# các vision modules. Để KHÔNG eager-load, ta lazy-import bên trong _get_*().
from scripts.query_demo import (
    retrieve,
    print_results,            # không dùng (giữ cho tương thích)
    _chunk_short,
    _print_result_row,
)

# Cố gắng import helper cho 1-hop KG query từ query_demo (không bắt buộc)
try:
    from scripts.query_demo import (
        _neo4j_driver_or_none,
        _collect_1hop_relations,
    )
except Exception:
    _neo4j_driver_or_none = None
    _collect_1hop_relations = None


# ----------------------------------------------------------------------------
# Streamlit config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Hybrid Search — Buổi 14",
    page_icon="🔍",
    layout="wide",
)

st.title("RAG Hybrid Search — Buổi 14")
st.caption(
    "Demo UI cho pipeline retrieval: BM25 · Dense · Hybrid (RRF) · Hybrid + Rerank. "
    "Mini KG (VanBan / DieuKhoan) load qua `scripts/load_mini_kg.py`, query 1-hop "
    "qua `cypher/demo_queries.cypher`."
)

METHOD_OPTIONS = {
    "BM25": "bm25",
    "Dense": "dense",
    "Hybrid": "hybrid",
    "Hybrid + Rerank": "hybrid_rerank",
}


# ----------------------------------------------------------------------------
# Resource cache — build retrievers 1 lần / session
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang build BM25 retriever (cache)...")
def _get_bm25():
    df = load_corpus()
    return BM25Retriever(df), df

@st.cache_resource(show_spinner="Đang build Dense retriever (load embedding cache)...")
def _get_dense():
    from src.dense_retriever import DenseRetriever
    df = load_corpus()
    return DenseRetriever(df), df

@st.cache_resource(show_spinner="Đang build Hybrid retriever...")
def _get_hybrid():
    from src.hybrid_retriever import HybridRetriever
    df = load_corpus()
    return HybridRetriever(df), df

@st.cache_resource(show_spinner="Đang build Rerank pipeline (Cross-Encoder)...")
def _get_rerank():
    from src.reranker import RerankPipeline
    df = load_corpus()
    return RerankPipeline(df), df


# ----------------------------------------------------------------------------
# Sidebar — controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Tìm kiếm")
    query = st.text_area("Câu hỏi", height=100,
                         placeholder="Nhập câu hỏi tiếng Việt...")

    method_label = st.selectbox(
        "Method",
        list(METHOD_OPTIONS.keys()),
        index=3,                       # mặc định Hybrid + Rerank
        help="Chọn retrieval pipeline.",
    )

    top_k = st.slider("Top-k", min_value=1, max_value=20, value=5, step=1)

    candidate_k = st.slider("Candidate-k (Hybrid path)", min_value=10,
                            max_value=50, value=20, step=5,
                            help="Số candidate Hybrid lấy trước khi rerank (chỉ Hybrid/Hybrid + Rerank).")

    run_clicked = st.button("Tìm kiếm", type="primary", width="stretch")


# ----------------------------------------------------------------------------
# Main — chạy retrieval khi click
# ----------------------------------------------------------------------------
method = METHOD_OPTIONS[method_label]

# BEFORE / AFTER cho hybrid_rerank: cần truy cập cả candidates (BEFORE) và reranked (AFTER)
def _rerank_with_before_after(query_text: str, top_k_: int, cand_k: int):
    from src.reranker import RerankPipeline   # lazy để không trigger torch/transformers ở startup
    _, df = _get_dense()
    pipe = RerankPipeline(df)
    candidates, reranked, meta = pipe.search(
        query_text, top_k=top_k_, candidate_k=cand_k,
    )
    # Chuẩn hoá BEFORE sang dict có các field "score"/"citation" để hiển thị
    before_rows = []
    for c in candidates[:top_k_]:
        before_rows.append({
            "rank": c.get("final_rank"),
            "chunk_id": c.get("chunk_id"),
            "document_id": c.get("document_id"),
            "text": c.get("text", ""),
            "score": c.get("rrf_score"),
            "citation": c.get("citation", ""),
            "retrieval_method": "hybrid",
            "hybrid_bm25_rank": c.get("hybrid_bm25_rank"),
            "hybrid_dense_rank": c.get("hybrid_dense_rank"),
            "hybrid_rrf_score": c.get("rrf_score"),
            "rerank_score": None,
            "is_reranker_fallback": None,
        })
    return reranked, before_rows, meta


if run_clicked:
    if not query or not query.strip():
        st.warning("Vui lòng nhập câu hỏi.")
        st.stop()

    with st.spinner(f"Đang chạy `{method_label}`..."):
        try:
            if method == "hybrid_rerank":
                reranked, before_rows, rmeta = _rerank_with_before_after(
                    query, top_k, candidate_k,
                )
                # Ánh xạ reranked -> dict thống nhất (dùng helper từ retrieve())
                from scripts.query_demo import _to_unified
                results = [_to_unified(r["final_rank"], r, "hybrid_rerank") for r in reranked]
                meta = {
                    "method": method_label,
                    "rerank_method": rmeta.get("method", "?"),
                    "is_fallback": bool(rmeta.get("is_fallback", False)),
                }
            else:
                if method == "bm25":
                    _, df = _get_bm25()
                elif method == "dense":
                    _, df = _get_dense()
                else:
                    _, df = _get_hybrid()
                results, _meta = retrieve(
                    query, method, top_k, df=df, candidate_k=candidate_k,
                )
                meta = {"method": method_label}

            # Top-5 doc/chunk for Graph hints
            doc_ids = []
            seen = set()
            for r in results:
                d = r.get("document_id", "")
                if d and d not in seen:
                    seen.add(d)
                    doc_ids.append(d)
            chunk_ids = [r.get("chunk_id", "") for r in results if r.get("chunk_id")]

        except Exception as e:
            st.error(f"Lỗi khi chạy retrieval: {type(e).__name__}: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            st.stop()

    # ------------------------------------------------------------------------
    # BEFORE / AFTER Rerank (chỉ cho hybrid_rerank)
    # ------------------------------------------------------------------------
    if method == "hybrid_rerank":
        st.subheader("Before / After Rerank")
        st.caption(
            f"Rerank method: `{meta.get('rerank_method', '?')}` · "
            + ("🟡 FALLBACK (không neural)" if meta.get("is_fallback") else "🟢 neural reranker")
        )
        before_map = {b["chunk_id"]: b for b in before_rows}
        rows = []
        # BEFORE — điểm hybrid_score, "—" cho rerank_score
        for b in before_rows:
            h = b.get("hybrid_rrf_score")
            rows.append({
                "stage": "BEFORE",
                "rank": b.get("rank"),
                "chunk": _chunk_short(b["chunk_id"]),
                "hybrid_score": f"{h:+.4f}" if h is not None else "—",
                "rerank_score": "—",
                "citation": (b.get("citation") or "")[:60],
            })
        # AFTER — rerank_score, kèm hybrid_score lấy từ BEFORE map
        for a in results:
            b = before_map.get(a["chunk_id"]) or {}
            b_rank = b.get("rank") if b else None
            h = b.get("hybrid_rrf_score") if b else None
            r = a.get("rerank_score")
            delta_s = ""
            if b_rank is not None and isinstance(a.get("rank"), int):
                d = b_rank - a["rank"]
                delta_s = f"{d:+d}" if d != 0 else "0"
            cite = (a.get("citation") or "")[:60]
            if delta_s:
                cite = cite + f"   Δ={delta_s}"
            rows.append({
                "stage": "AFTER" + ("  FALLBACK" if a.get("is_reranker_fallback") else ""),
                "rank": a.get("rank"),
                "chunk": _chunk_short(a["chunk_id"]),
                "hybrid_score": f"{h:+.4f}" if h is not None else "—",
                "rerank_score": f"{r:+.4f}" if r is not None else "—",
                "citation": cite,
            })

        st.dataframe(
            rows,
            column_config={
                "stage":       st.column_config.TextColumn("stage", width="small"),
                "rank":        st.column_config.TextColumn("rank", width="small"),
                "chunk":       st.column_config.TextColumn("chunk", width="medium"),
                "hybrid_score":st.column_config.TextColumn("hybrid (RRF)", width="medium"),
                "rerank_score":st.column_config.TextColumn("rerank", width="medium"),
                "citation":    st.column_config.TextColumn("citation", width="large"),
            },
            hide_index=True,
            width="stretch",
        )

    # ------------------------------------------------------------------------
    # Kết quả chính
    # ------------------------------------------------------------------------
    st.subheader(f"Kết quả · method = {method_label} · top_k = {top_k}")
    if not results:
        st.info("Không có kết quả.")
    else:
        for r in results:
            rno = r["rank"]
            cid = _chunk_short(r["chunk_id"])
            doc = (r["document_id"] or "")[:18]
            score = r["score"]
            score_str = f"{score:+.4f}" if score is not None else "—"
            extras = []
            if method == "hybrid":
                if r.get("hybrid_bm25_rank") is not None:
                    extras.append(f"bm25_rank={r['hybrid_bm25_rank']}")
                if r.get("hybrid_dense_rank") is not None:
                    extras.append(f"dense_rank={r['hybrid_dense_rank']}")
                if r.get("hybrid_rrf_score") is not None:
                    extras.append(f"rrf={r['hybrid_rrf_score']:+.4f}")
            elif method == "hybrid_rerank":
                if r.get("hybrid_score") is not None:
                    extras.append(f"hybrid={r['hybrid_score']:+.4f}")
                if r.get("rerank_score") is not None:
                    extras.append(f"rerank={r['rerank_score']:+.4f}")
                if r.get("is_reranker_fallback") is True:
                    extras.append("FALLBACK-reranker")
            extra_str = " · ".join(extras)
            with st.container(border=True):
                h = f"**#{rno}** · `chunk={cid}` · `doc={doc}` · score={score_str}"
                if extra_str:
                    h += f"  \n<sub>{extra_str}</sub>"
                st.markdown(h)
                st.markdown(f"**Citation:** `{r.get('citation','')}`")
                snippet = (r.get("text") or "")[:600].replace("\n", "  \n")
                st.markdown(f"**Text:**")
                st.markdown(f"> {snippet}{'…' if len(r.get('text') or '') > 600 else ''}")

    # ------------------------------------------------------------------------
    # Graph hints
    # ------------------------------------------------------------------------
    st.subheader("Graph hints")
    if not doc_ids:
        st.write("—")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Document IDs** (của các chunk retrieved):")
            for d in doc_ids:
                st.code(d, language="text")
        with col2:
            st.markdown("**Chunk IDs:**")
            for c in chunk_ids:
                st.code(_chunk_short(c), language="text")

        st.markdown("**Direct relations (1-hop trong Mini KG):**")
        if _neo4j_driver_or_none is None:
            st.info("Không query được Neo4j (helper hoặc driver lỗi).")
        else:
            driver = _neo4j_driver_or_none()
            if driver is None:
                st.info(
                    "Neo4j chưa sẵn sàng — thiếu `NEO4J_URI` / `NEO4J_PASSWORD` "
                    "trong `.env`. Hệ thống retrieval vẫn chạy, Graph hints được lược bỏ."
                )
            else:
                try:
                    db = os.environ.get("NEO4J_DATABASE", "neo4j")
                    rels = _collect_1hop_relations(driver, doc_ids, db)
                    if not rels:
                        st.write(
                            "_(không có edge 1-hop nào — chưa nạp buổi 14 vào Neo4j "
                            "hoặc `doc_ids` không trùng node)_"
                        )
                    else:
                        for r in rels:
                            st.code(
                                f"(VanBan {r['src_so']}) -[:{r['rel']} "
                                f"conf={r['conf']:.2f}]-> ({r['n_label']} {r['target']})",
                                language="cypher",
                            )
                except Exception as e:
                    st.warning(f"Lỗi khi truy vấn Neo4j: {e}")
                finally:
                    driver.close()


# ----------------------------------------------------------------------------
# Footer (always visible)
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Để dừng Streamlit: Ctrl+C trong terminal. "
    "Khôi phục cache: `rm -rf .venv` rồi `pip install -r requirements.txt`."
)
