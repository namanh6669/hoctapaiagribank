"""Streamlit UI Buổi 08 - Advanced RAG comparison workspace.

The UI calls public functions from rag.py and advanced_rag.py instead of
reimplementing retrieval logic. Opening the app is read-only: it does not index,
download reranker models, or call generation APIs until the user clicks an
action button.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

import advanced_rag
import rag


STRATEGIES = ["hierarchical", "semantic", "fixed-size"]
MODES = ["bm25", "semantic", "hybrid", "hybrid_rerank"]
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"


st.set_page_config(page_title="Buổi 08 - Advanced RAG", page_icon="🧭", layout="wide")


if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer_result" not in st.session_state:
    st.session_state.last_answer_result = None
if "last_compare_result" not in st.session_state:
    st.session_state.last_compare_result = None
if "last_trace_result" not in st.session_state:
    st.session_state.last_trace_result = None


@st.cache_resource(show_spinner=False)
def cached_bm25_corpus(strategy: str, input_path: str | None) -> dict[str, Any]:
    """Cache BM25 corpus by strategy/input path for the small workshop corpus."""
    chunks, stats = rag.load_chunks(input_path or rag.BUOI_05_CHUNKS_DIR, strategy=strategy)
    corpus = advanced_rag.build_bm25_corpus(chunks)
    return {"corpus": corpus, "stats": stats, "chunk_count": len(chunks)}


@st.cache_data(show_spinner=False)
def read_report_json(path: str) -> dict[str, Any]:
    """Read one evaluation report JSON; never triggers evaluation."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        message = message.replace(api_key, "[secret]")
    return message[:800]


def page_text(page_start: int, page_end: int) -> str:
    return f"tr. {page_start}" if page_start == page_end else f"tr. {page_start}-{page_end}"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_row(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label, value if value is not None else "—", help=help_text)


def show_status_badge(status: str) -> None:
    if status == "answered":
        st.success("answered — đã tạo câu trả lời từ accepted evidence.")
    elif status == "insufficient_evidence":
        st.warning("insufficient_evidence — chưa có evidence đạt gate; không gọi generation.")
    elif status == "retrieval_only":
        st.info("retrieval_only — retrieval có kết quả nhưng generation lỗi/rỗng.")
    elif status == "reranker_unavailable":
        st.error("reranker_unavailable — reranker chưa sẵn sàng, không giả vờ dùng RRF như đã rerank.")
        st.caption(
            "Hãy chạy command rerank khi chủ động muốn tải model, hoặc kiểm tra Internet/disk/RAM/cache: "
            "`python -m streamlit run ...` không tự tải model khi mở trang."
        )
    else:
        st.warning(f"Trạng thái chưa biết: {status}")


def evidence_card(item: dict[str, Any], index: int) -> None:
    accepted = "✅ accepted" if item.get("accepted") else "⛔ rejected"
    title = f"{item.get('evidence_id', 'E?')} · {accepted} · {item['chunk_id']} · {item['source']} {page_text(item['page_start'], item['page_end'])}"
    with st.expander(title, expanded=index <= 3 or item.get("accepted", False)):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BM25 rank", fmt(item.get("bm25_rank")))
        c1.metric("BM25 score ↑", fmt(item.get("bm25_score")))
        c2.metric("Semantic rank", fmt(item.get("semantic_rank")))
        c2.metric("Cosine distance ↓", fmt(item.get("semantic_distance")))
        c3.metric("Fused rank", fmt(item.get("fused_rank")))
        c3.metric("RRF score ↑", fmt(item.get("rrf_score"), 6))
        c4.metric("Rerank rank", fmt(item.get("rerank_rank")))
        c4.metric("Rerank score ↑", fmt(item.get("rerank_score")))
        st.write(
            {
                "source": item["source"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "chunk_id": item["chunk_id"],
                "rerank_raw_score": item.get("rerank_raw_score"),
                "rank_change": item.get("rank_change"),
                "accepted": item.get("accepted"),
            }
        )
        st.text_area("Nội dung evidence", item.get("text", ""), height=180, key=f"evidence_{index}_{item['chunk_id']}")


def compare_rows_to_table(compare_result: dict[str, Any]) -> list[dict[str, Any]]:
    table = []
    for row in compare_result.get("rows", []):
        ranks = row.get("ranks", {})
        table.append(
            {
                "chunk_id": row["chunk_id"],
                "bm25_rank": ranks.get("bm25"),
                "semantic_rank": ranks.get("semantic"),
                "fused_rank": ranks.get("hybrid"),
                "rerank_rank": ranks.get("hybrid_rerank"),
                "rank_change": row.get("rank_movement"),
                "final modes": ", ".join(row.get("modes", [])),
            }
        )
    return table


def candidate_panel(mode: str, mode_result: dict[str, Any]) -> None:
    st.markdown(f"#### {mode}")
    if mode_result.get("status") == "reranker_unavailable":
        st.error("Reranker unavailable")
        st.caption(mode_result.get("error", ""))
        return
    candidates = mode_result.get("candidates", [])[:5]
    if not candidates:
        st.info("Không có candidate.")
        return
    rows = []
    for item in candidates:
        rows.append(
            {
                "chunk_id": item["chunk_id"],
                "bm25": item.get("bm25_rank"),
                "semantic": item.get("semantic_rank"),
                "fused": item.get("fused_rank"),
                "rerank": item.get("rerank_rank"),
                "score/distance": item.get("rerank_score") or item.get("rrf_score") or item.get("semantic_distance") or item.get("bm25_score"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def trace_metrics(trace: dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BM25 candidates", trace.get("bm25_candidates", 0))
    c2.metric("Semantic candidates", trace.get("semantic_candidates", 0))
    c3.metric("Union / Overlap", f"{trace.get('union', 0)} / {trace.get('overlap', 0)}")
    c4.metric("Reranked", trace.get("reranked", 0))
    c5.metric("Accepted", trace.get("accepted", 0))


def latency_table(trace: dict[str, Any]) -> list[dict[str, Any]]:
    latency = trace.get("latency_ms", {})
    return [{"stage": key, "latency_ms": latency.get(key, 0.0)} for key in advanced_rag.LATENCY_KEYS]


def report_metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metrics = report.get("metrics", report.get("results", {}))
    if isinstance(metrics, dict):
        for mode, values in metrics.items():
            if isinstance(values, dict):
                rows.append(
                    {
                        "mode": mode,
                        "Recall@K": values.get("recall_at_k") or values.get("Recall@K") or values.get("recall"),
                        "MRR@K": values.get("mrr_at_k") or values.get("MRR@K") or values.get("mrr"),
                        "nDCG@K": values.get("ndcg_at_k") or values.get("nDCG@K") or values.get("ndcg"),
                        "latency_avg_ms": values.get("latency_avg_ms") or values.get("avg_latency_ms"),
                        "latency_p50_ms": values.get("latency_p50_ms") or values.get("p50_latency_ms"),
                    }
                )
    return rows


def report_needs_review(report: dict[str, Any]) -> bool:
    questions = report.get("questions") or report.get("eval_questions") or []
    if isinstance(questions, list) and any(q.get("needs_human_review") is True for q in questions if isinstance(q, dict)):
        return True
    return bool(report.get("needs_human_review"))


with st.sidebar:
    st.title("🧭 Advanced RAG")
    strategy = st.selectbox("Strategy", STRATEGIES, index=0, key="sidebar_strategy")
    retrieval_mode = st.selectbox("Retrieval mode", MODES, index=MODES.index("hybrid_rerank"), key="sidebar_mode")

    try:
        config = advanced_rag.load_config(advanced_rag.ENV_EXAMPLE_PATH if not advanced_rag.ENV_PATH.exists() else advanced_rag.ENV_PATH)
    except Exception as error:  # noqa: BLE001
        config = None
        st.error(f"Không đọc được config: {safe_error(error)}")

    try:
        status = advanced_rag.advanced_status(strategy=strategy)
    except Exception as error:  # noqa: BLE001
        status = None
        st.error(f"Không đọc được status: {safe_error(error)}")

    try:
        bm25_cache_info = cached_bm25_corpus(strategy, None)
    except Exception as error:  # noqa: BLE001
        bm25_cache_info = None
        st.error(f"Không cache được BM25 corpus: {safe_error(error)}")

    if config:
        st.subheader("Config retrieval")
        st.write("**Final top-k:**", config["final_top_k"])
        st.write("**BM25 candidates:**", config["bm25_candidates"])
        st.write("**Semantic candidates:**", config["semantic_candidates"])
        st.write("**RRF k:**", config["rrf_k"])
        st.write("**RRF weights:**", f"BM25={config['rrf_bm25_weight']} · Semantic={config['rrf_semantic_weight']}")
        st.subheader("Reranker")
        st.write("**Model:**", config["reranker_model"])
        st.write("**Device:**", config["rerank_device"])
        st.write("**Rerank candidates:**", config["rerank_candidates"])
        st.write("**Min score:**", config["rerank_min_score"])
    if status:
        st.write("**Cache exists:**", "Có" if status["reranker_cache_exists"] else "Chưa")
        st.caption(status["reranker_cache_path"])
        st.subheader("Semantic index")
        st.write("**Collection:**", status["semantic_collection_name"])
        st.write("**Exists/count:**", f"{status['collection_exists']} / {status['collection_count']}")
        st.write("**Embedding:**", f"{status['embedding_model']} · dim={status['embedding_dim']}")
        st.write("**BM25 ready:**", "Có" if status["bm25_ready"] else "Chưa")
    if bm25_cache_info:
        st.write("**BM25 cache chunks:**", bm25_cache_info["chunk_count"])
    if config:
        st.write("**API key:**", config["api_key_status"])
    if st.button("Clear UI cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


st.title("Buổi 08 — Advanced RAG nhiều tầng")
st.caption("BM25 → Semantic → RRF → Rerank → Grounded answer/citation. App không tự index hoặc tải model khi mở trang.")

answer_tab, compare_tab, trace_tab, eval_tab = st.tabs([
    "Hỏi đáp Advanced RAG",
    "So sánh Retrieval",
    "Pipeline Trace",
    "Đánh giá",
])

with answer_tab:
    st.header("1. Hỏi đáp Advanced RAG")
    q1 = st.text_area("Question", value=st.session_state.last_question, height=120, key="answer_question")
    mode = st.selectbox("Mode cho query", MODES, index=MODES.index(retrieval_mode), key="answer_mode")
    if st.button("Chạy query", type="primary"):
        st.session_state.last_question = q1
        if not q1.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang retrieve, gate và generation nếu đủ evidence..."):
                try:
                    result = advanced_rag.answer_query(q1, mode=mode, strategy=strategy)
                    st.session_state.last_answer_result = result
                    st.session_state.last_trace_result = result
                except Exception as error:  # noqa: BLE001
                    st.error(f"Query lỗi: {safe_error(error)}")
                    if "Chưa có Chroma storage" in str(error) or "chưa tồn tại" in str(error):
                        st.info("Thiếu semantic index. Hãy chạy `advanced_rag.py prepare-semantic --strategy hierarchical` khi đã có GEMINI_API_KEY.")

    result = st.session_state.last_answer_result
    if result:
        show_status_badge(result["status"])
        st.subheader("Answer")
        st.write(result["answer"])
        if result["warnings"]:
            st.warning("Warnings")
            for warning in result["warnings"]:
                st.write(f"- {warning}")
        st.subheader("Citations")
        if result["citations"]:
            st.json(result["citations"])
        else:
            st.info("Chưa có citation được map từ accepted evidence.")
        st.subheader("Evidence cards")
        for idx, item in enumerate(result["evidence"], start=1):
            evidence_card(item, idx)

with compare_tab:
    st.header("2. So sánh Retrieval")
    q2 = st.text_area("Question để compare", value=st.session_state.last_question, height=100, key="compare_question")
    st.caption("Compare chạy BM25, Semantic, Hybrid RRF, Hybrid + Rerank; không gọi generation.")
    if st.button("Chạy compare"):
        st.session_state.last_question = q2
        if not q2.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang chạy retrieval/rerank comparison..."):
                try:
                    compare_result = advanced_rag.compare_modes(q2, strategy=strategy)
                    st.session_state.last_compare_result = compare_result
                    st.session_state.last_trace_result = compare_result
                except Exception as error:  # noqa: BLE001
                    st.error(f"Compare lỗi: {safe_error(error)}")
                    if "Chưa có Chroma storage" in str(error) or "chưa tồn tại" in str(error):
                        st.info("Thiếu semantic index. Hãy chạy prepare-semantic trước khi compare semantic/hybrid.")

    compare_result = st.session_state.last_compare_result
    if compare_result:
        st.subheader("Bảng rank chung")
        st.dataframe(compare_rows_to_table(compare_result), use_container_width=True, hide_index=True)
        st.subheader("Top-k theo từng mode")
        cols = st.columns(4)
        for col, mode_name in zip(cols, MODES):
            with col:
                candidate_panel(mode_name, compare_result["mode_results"].get(mode_name, {}))

with trace_tab:
    st.header("3. Pipeline Trace")
    trace_source = st.session_state.last_answer_result
    if trace_source:
        trace = trace_source["trace"]
        trace_metrics(trace)
        st.subheader("Latency")
        st.dataframe(latency_table(trace), use_container_width=True, hide_index=True)
        st.subheader("Trace JSON")
        st.json(trace)
    else:
        st.info("Chưa có query result. Chạy tab Hỏi đáp để xem trace đầy đủ.")
    st.markdown(
        """
        **Chú thích score**

        - BM25 score cao hơn thường tốt hơn trong nhánh lexical.
        - Cosine distance thấp hơn tốt hơn trong semantic retrieval.
        - RRF score và rerank score cao hơn tốt hơn.
        - Rerank score là sigmoid(logit) của model, **không phải xác suất câu trả lời đúng**.
        """
    )

with eval_tab:
    st.header("4. Đánh giá")
    st.caption("Tab này chỉ đọc report JSON có sẵn trong reports/. Không tự chạy evaluation hàng loạt hoặc gọi API khi mở trang.")
    report_files = sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.exists() else []
    if not report_files:
        st.info("Chưa có report JSON hợp lệ trong reports/. Không kết luận winner.")
    else:
        selected = st.selectbox("Report JSON", [str(path.relative_to(BASE_DIR)) for path in report_files])
        path = BASE_DIR / selected
        try:
            report = read_report_json(str(path))
            if report_needs_review(report):
                st.warning("Gold labels còn needs_human_review=true; không xem đây là kết quả đã được chuyên gia pháp lý duyệt.")
            rows = report_metric_rows(report)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.warning("Report chưa có metrics Recall@K/MRR@K/nDCG@K theo mode ở schema UI nhận diện được.")
            st.caption("Không kết luận winner nếu report/gold labels chưa hợp lệ hoặc còn cần human review.")
            with st.expander("Raw report JSON"):
                st.json(report)
        except Exception as error:  # noqa: BLE001
            st.error(f"Không đọc được report: {safe_error(error)}")
