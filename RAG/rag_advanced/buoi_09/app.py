"""Streamlit UI for Buổi 09 - Multi-query & Parent-Child Retrieval.

Importing this module is safe for unit tests: Streamlit page setup and backend actions
only run inside ``main()`` when Streamlit executes the script.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import hierarchical_rag as hr


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
MODES = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
PARENT_MODES = {"single_parent", "multi_parent"}
FLAT_MODES = {"single_flat", "multi_flat"}


def page_text(page_start: int | None, page_end: int | None) -> str:
    if page_start is None or page_end is None:
        return "—"
    return f"tr. {page_start}" if page_start == page_end else f"tr. {page_start}-{page_end}"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def safe_error(error: Exception | str) -> str:
    message = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        message = message.replace(api_key, "[secret]")
    return message[:800]


def status_guidance(status: str) -> dict[str, str]:
    mapping = {
        "answered": ("success", "Đã trả lời từ accepted evidence."),
        "ready": ("success", "Retrieval/rerank hoàn tất."),
        "partial": ("warning", "Một số generated query lỗi; xem query_errors/trace."),
        "multi_query_partial": ("warning", "Q0 thành công nhưng toàn bộ generated query retrieval lỗi."),
        "hierarchy_not_ready": ("error", "Hierarchy store thiếu/stale. Chạy build-hierarchy bằng action riêng trước khi dùng parent mode."),
        "collection_not_ready": ("error", "Semantic collection chưa sẵn sàng. Chạy prepare semantic bằng action riêng."),
        "query_generation_unavailable": ("error", "Không sinh được query variants; kiểm tra API key/schema/lỗi validation."),
        "reranker_unavailable": ("error", "Reranker lỗi/thiếu model; không fallback im lặng."),
        "insufficient_evidence": ("warning", "Không có evidence đạt gate; không gọi answer generation."),
        "generation_error": ("error", "Generation lỗi; giữ retrieval trace để debug."),
        "citation_validation_failed": ("error", "Answer có citation label không khớp evidence thật."),
    }
    level, message = mapping.get(status, ("warning", f"Trạng thái: {status}"))
    return {"level": level, "message": message}


def citation_display(citation: dict[str, Any]) -> str:
    evidence_id = citation.get("evidence_id", "?")
    source = citation.get("source", "?")
    pages = page_text(citation.get("page_start"), citation.get("page_end"))
    if "parent_id" in citation:
        return f"[{evidence_id}] {source}, {pages}, parent={citation.get('parent_id')}, anchor={citation.get('anchor_child_id')}"
    return f"[{evidence_id}] {source}, {pages}, child={citation.get('child_id')}"


def query_cards(result: dict[str, Any]) -> list[dict[str, Any]]:
    queries = result.get("query_set") or result.get("queries") or []
    retrieval_latency = _nested(result, ["trace", "retrieval_trace", "query_retrieval_latency_ms"], {}) or _nested(result, ["trace", "query_retrieval_latency_ms"], {})
    result_counts = _nested(result, ["trace", "retrieval_trace", "result_count_by_query"], {}) or _nested(result, ["trace", "result_count_by_query"], {})
    errors = result.get("query_errors", {})
    cards = []
    for query in queries:
        query_id = query.get("query_id")
        cards.append(
            {
                "query_id": query_id,
                "text": query.get("text", ""),
                "origin": query.get("origin", "original" if query_id == "Q0" else "generated"),
                "focus": query.get("focus", "—"),
                "validation_status": "error" if query_id in errors else "ok",
                "result_count": result_counts.get(query_id, 0),
                "retrieval_latency_ms": retrieval_latency.get(query_id, 0.0),
                "weight": query.get("weight"),
            }
        )
    return cards


def query_child_matrix(child_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_ids = sorted({qid for child in child_hits for qid in child.get("per_query_ranks", {})}, key=_query_id_sort_key)
    rows = []
    for child in sorted(child_hits, key=lambda item: item.get("multi_query_rank", item.get("child_rerank_rank", 10**9))):
        row = {
            "child_id": child.get("child_id") or child.get("chunk_id"),
            "support_query_count": child.get("support_query_count", len(child.get("support_query_ids", []))),
            "MQ-RRF": child.get("multi_query_rrf_score"),
            "rerank": child.get("child_rerank_rank"),
            "source/pages": f"{child.get('source', '—')} {page_text(child.get('page_start'), child.get('page_end'))}",
        }
        ranks = child.get("per_query_ranks", {})
        for query_id in query_ids:
            row[query_id] = ranks.get(query_id, "—")
        rows.append(row)
    return rows


def parent_tree_data(result: dict[str, Any]) -> list[dict[str, Any]]:
    child_by_id = {child.get("child_id") or child.get("chunk_id"): child for child in result.get("child_hits", result.get("children", []))}
    mapping_rows = _nested(result, ["trace", "retrieval_trace", "child_to_parent_mapping"], []) or _nested(result, ["trace", "child_to_parent_mapping"], [])
    mapping_by_child = {row.get("child_id"): row for row in mapping_rows if isinstance(row, dict)}
    parents = result.get("parent_candidates") or result.get("parents") or []
    nodes = []
    for parent in parents:
        children = []
        for child_id in parent.get("supporting_child_ids", []):
            child = child_by_id.get(child_id, {})
            mapping = mapping_by_child.get(child_id, {})
            children.append(
                {
                    "child_id": child_id,
                    "query_ids": mapping.get("support_query_ids") or child.get("support_query_ids", []),
                    "query_ranks": child.get("per_query_ranks", {}),
                    "multi_query_rank": mapping.get("multi_query_rank") or child.get("multi_query_rank"),
                    "anchor_snippet": " ".join(str(child.get("text", "")).split())[:180],
                    "is_anchor": child_id == parent.get("anchor_child_id"),
                }
            )
        nodes.append(
            {
                "parent_id": parent.get("parent_id"),
                "evidence_id": parent.get("evidence_id"),
                "structural_path": parent.get("structural_path", {}),
                "source": parent.get("source"),
                "pages": page_text(parent.get("page_start"), parent.get("page_end")),
                "parent_rank": parent.get("parent_rank"),
                "parent_rerank_rank": parent.get("parent_rerank_rank"),
                "parent_rrf_score": parent.get("parent_rrf_score"),
                "parent_rerank_score": parent.get("parent_rerank_score"),
                "text": parent.get("text", ""),
                "ambiguous": parent.get("ambiguous", False),
                "warnings": parent.get("warnings", []),
                "children": children,
            }
        )
    return nodes


def mode_comparison_row(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    if mode in PARENT_MODES and "retrieval" in payload:
        retrieval = payload.get("retrieval", {})
        rerank = payload.get("rerank", {})
        parent_candidates = rerank.get("parents") or retrieval.get("parents", [])
        child_hits = retrieval.get("children", [])
        trace = retrieval.get("trace", {})
        status = rerank.get("status") or retrieval.get("status")
        warnings = retrieval.get("warnings", []) + ([rerank.get("error")] if rerank.get("error") else [])
    elif mode in PARENT_MODES:
        parent_candidates = payload.get("parent_candidates") or payload.get("parents", [])
        child_hits = payload.get("child_hits") or payload.get("children", [])
        trace = payload.get("trace", {}).get("retrieval_trace", payload.get("trace", {}))
        status = payload.get("status")
        warnings = payload.get("warnings", [])
    else:
        parent_candidates = []
        child_hits = payload.get("child_hits") or payload.get("children", [])
        trace = payload.get("trace", {})
        status = payload.get("status")
        warnings = payload.get("warnings", [])
    evidence = payload.get("accepted_evidence") or parent_candidates or child_hits
    sources = sorted({item.get("source") for item in evidence if item.get("source")})
    articles = sorted({str((item.get("structural_path") or {}).get("article")) for item in evidence if (item.get("structural_path") or {}).get("article")})
    api_counts = payload.get("trace", {}).get("api_call_counts", {})
    return {
        "mode": mode,
        "status": status,
        "final evidence IDs": ", ".join(str(item.get("evidence_id") or item.get("parent_id") or item.get("child_id")) for item in evidence[:5]),
        "unit type": "parent" if mode in PARENT_MODES else "child",
        "rank fields": _rank_summary(evidence[:3]),
        "source/pages": "; ".join(f"{item.get('source')} {page_text(item.get('page_start'), item.get('page_end'))}" for item in evidence[:3]),
        "unique sources/articles": f"{len(sources)} / {len(articles)}",
        "retrieved child count": len(child_hits) or trace.get("input_child_hit_count", 0),
        "expanded parent count": len(parent_candidates) or trace.get("selected_parent_count", 0),
        "context chars": trace.get("expanded_parent_chars", sum(len(str(item.get("text", ""))) for item in evidence)),
        "expansion factor": trace.get("context_expansion_factor", 0.0),
        "latency": _latency_total(payload),
        "Generation calls": api_counts.get("generation_calls", 0),
        "Embedding calls": api_counts.get("semantic_embedding_calls", trace.get("semantic_embedding_call_count", 0)),
        "warnings": "; ".join(str(w) for w in warnings if w),
    }


def report_metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("metrics") or report.get("results") or report.get("mode_metrics") or {}
    rows = []
    if isinstance(metrics, dict):
        for mode, values in metrics.items():
            if isinstance(values, dict):
                rows.append(
                    {
                        "mode": mode,
                        "Child Recall@K": values.get("child_recall_at_k") or values.get("Child Recall@K") or values.get("recall_at_k"),
                        "Parent Recall@K": values.get("parent_recall_at_k") or values.get("Parent Recall@K"),
                        "MRR@K": values.get("mrr_at_k") or values.get("MRR@K") or values.get("mrr"),
                        "nDCG@K": values.get("ndcg_at_k") or values.get("nDCG@K") or values.get("ndcg"),
                        "latency_ms": values.get("latency_ms") or values.get("latency_avg_ms") or values.get("p50_latency_ms"),
                        "context_chars": values.get("context_chars") or values.get("expanded_parent_chars"),
                    }
                )
    return rows


def report_needs_review(report: dict[str, Any]) -> bool:
    questions = report.get("questions") or report.get("eval_questions") or []
    return bool(report.get("needs_human_review")) or any(isinstance(item, dict) and item.get("needs_human_review") is True for item in questions)


def latest_report_path(reports_dir: Path = REPORTS_DIR) -> Path | None:
    if not reports_dir.exists():
        return None
    files = sorted(reports_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _query_id_sort_key(query_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(query_id) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(query_id))


def _rank_summary(items: list[dict[str, Any]]) -> str:
    parts = []
    for item in items:
        if "parent_id" in item:
            parts.append(f"{item.get('parent_id')}: parent {item.get('parent_rank')}→{item.get('parent_rerank_rank')}")
        else:
            parts.append(f"{item.get('child_id')}: mq {item.get('multi_query_rank')} rerank {item.get('child_rerank_rank')}")
    return "; ".join(parts)


def _latency_total(payload: dict[str, Any]) -> float:
    trace = payload.get("trace", {})
    stage = trace.get("stage_latencies_ms", {})
    if "total" in stage:
        return stage["total"]
    latency = trace.get("latency_ms", {})
    if isinstance(latency, dict):
        return latency.get("total", sum(value for value in latency.values() if isinstance(value, (int, float))))
    return 0.0


def _runtime_config(base: hr.HierarchyConfig, values: dict[str, Any]) -> hr.HierarchyConfig:
    parent_candidates = int(values.get("parent_candidates", base.parent_candidates))
    final_parent_top_k = min(int(values.get("final_parent_top_k", base.final_parent_top_k)), parent_candidates)
    return replace(
        base,
        multi_query_count=int(values.get("multi_query_count", base.multi_query_count)),
        per_query_candidates=int(values.get("per_query_candidates", base.per_query_candidates)),
        parent_candidates=parent_candidates,
        final_parent_top_k=final_parent_top_k,
        reranker_min_score=float(values.get("reranker_min_score", base.reranker_min_score)),
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval",
        page_icon="🧬",
        layout="wide",
    )
    for key, value in {
        "last_answer_result": None,
        "last_compare_result": None,
        "last_parent_result": None,
        "last_question": "",
    }.items():
        st.session_state.setdefault(key, value)

    base_config = _load_config_for_ui()
    runtime_values = _sidebar(st, base_config)
    config = _runtime_config(base_config, runtime_values)

    st.title("RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval")
    st.caption("Query fan-out → Hybrid per query → Cross-query RRF → Parent expansion → Parent rerank")

    tabs = st.tabs(["Ask Advanced RAG", "Query Fan-out", "Parent–Child Explorer", "Mode Comparison", "Evaluation"])
    with tabs[0]:
        _tab_ask(st, config)
    with tabs[1]:
        _tab_query_fanout(st)
    with tabs[2]:
        _tab_parent_child(st)
    with tabs[3]:
        _tab_compare(st, config)
    with tabs[4]:
        _tab_evaluation(st)


def _load_config_for_ui() -> hr.HierarchyConfig:
    try:
        return hr.load_hierarchy_config()
    except Exception:
        return hr.load_hierarchy_config(hr.ENV_EXAMPLE_PATH)


def _sidebar(st: Any, config: hr.HierarchyConfig) -> dict[str, Any]:
    with st.sidebar:
        st.header("⚙️ Buổi 09 controls")
        st.selectbox("Mode", MODES, index=MODES.index("multi_parent"), key="sidebar_mode")
        values = {
            "multi_query_count": st.number_input("MULTI_QUERY_COUNT", min_value=1, max_value=5, value=config.multi_query_count),
            "per_query_candidates": st.number_input("PER_QUERY_CANDIDATES", min_value=1, max_value=100, value=config.per_query_candidates),
            "parent_candidates": st.number_input("PARENT_CANDIDATES", min_value=1, max_value=100, value=config.parent_candidates),
            "final_parent_top_k": st.number_input("FINAL_PARENT_TOP_K", min_value=1, max_value=100, value=config.final_parent_top_k),
            "reranker_min_score": st.slider("RERANK_MIN_SCORE", min_value=0.0, max_value=1.0, value=float(config.reranker_min_score), step=0.01),
        }
        st.write("**strategy:** hierarchical")
        st.write("**Gemini key:**", "Có" if os.getenv("GEMINI_API_KEY", "").strip() else "Không")
        st.write("**Embedding:**", f"{config.gemini_embedding_model} · dim={config.gemini_embedding_dim}")
        st.write("**Generation:**", config.gemini_generation_model)
        st.write("**Reranker:**", config.reranker_model)
        _sidebar_status(st, config)
        _action_buttons(st, config)
        return values


def _sidebar_status(st: Any, config: hr.HierarchyConfig) -> None:
    st.subheader("Status")
    h_status = hr.hierarchy_status()
    h_ready = hr.load_hierarchy_store(config=config)
    st.write("**Hierarchy:**", h_ready["status"])
    st.write("**children / parents:**", f"{h_status.get('counts', {}).get('children', '—')} / {h_status.get('counts', {}).get('parents', '—')}")
    st.write("**ambiguous:**", h_status.get("counts", {}).get("ambiguous_children", "—"))
    if h_ready.get("warnings"):
        st.warning("; ".join(h_ready["warnings"][:3]))
    try:
        import rag

        collection = rag.collection_status(strategy="hierarchical")
    except Exception as error:  # noqa: BLE001
        st.write("**Collection:**", f"unknown — {safe_error(error)}")
    else:
        st.write("**Collection:**", f"exists={collection.get('exists')} count={collection.get('count')} compatible={collection.get('compatible')}")


def _action_buttons(st: Any, config: hr.HierarchyConfig) -> None:
    st.subheader("Explicit actions")
    confirm = st.checkbox("Tôi hiểu action có thể ghi storage/gọi API/tải model", key="confirm_actions")
    c1, c2 = st.columns(2)
    if c1.button("Build hierarchy", disabled=not confirm):
        with st.spinner("Building hierarchy store..."):
            try:
                result = hr.build_hierarchy_store(config=config)
                st.success(f"Built: children={result['manifest']['counts']['children']} parents={result['manifest']['counts']['parents']}")
            except Exception as error:  # noqa: BLE001
                st.error(safe_error(error))
    if c2.button("Prepare semantic", disabled=not confirm):
        with st.spinner("Preparing semantic index..."):
            try:
                import advanced_rag

                result = advanced_rag.prepare_semantic(strategy="hierarchical")
                st.success(f"Prepared collection {result.get('collection_name')} count={result.get('count')}")
            except Exception as error:  # noqa: BLE001
                st.error(safe_error(error))


def _tab_ask(st: Any, config: hr.HierarchyConfig) -> None:
    st.header("Ask Advanced RAG")
    question = st.text_area("Câu hỏi", value=st.session_state.last_question, height=120, key="ask_question")
    mode = st.selectbox("Mode", MODES, index=MODES.index(st.session_state.get("sidebar_mode", "multi_parent")), key="ask_mode")
    if st.button("Run query", type="primary"):
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            st.session_state.last_question = question
            with st.spinner("Running Buổi 09 pipeline..."):
                try:
                    result = hr.answer_query_buoi09(question, mode=mode, config=config)
                    st.session_state.last_answer_result = result
                    st.session_state.last_parent_result = result if mode in PARENT_MODES else None
                except Exception as error:  # noqa: BLE001
                    st.error(safe_error(error))
    result = st.session_state.last_answer_result
    if result:
        guidance = status_guidance(result["status"])
        getattr(st, guidance["level"])(guidance["message"])
        counts = result.get("trace", {}).get("api_call_counts", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total latency ms", fmt(_latency_total(result)))
        c2.metric("Generation calls", counts.get("generation_calls", 0))
        c3.metric("Embedding calls", counts.get("semantic_embedding_calls", 0))
        st.subheader("Answer")
        st.write(result.get("answer") or "—")
        if result.get("warnings"):
            st.warning("\n".join(f"- {w}" for w in result["warnings"]))
        st.subheader("Citations")
        for citation in result.get("citations", []):
            st.write(citation_display(citation))
        with st.expander("Raw result"):
            st.json(result)


def _tab_query_fanout(st: Any) -> None:
    st.header("Query Fan-out")
    result = st.session_state.last_answer_result
    if not result:
        st.info("Chạy Ask Advanced RAG trước để xem query fan-out.")
        return
    cards = query_cards(result)
    if not cards:
        st.info("Mode hiện tại không có query_set trace.")
    cols = st.columns(max(1, min(4, len(cards))))
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            label = "🟦 Q0 original" if card["query_id"] == "Q0" else "🟨 generated"
            st.markdown(f"### {label}: {card['query_id']}")
            st.write(card["text"])
            st.caption(f"focus={card['focus']} · status={card['validation_status']} · results={card['result_count']} · latency={fmt(card['retrieval_latency_ms'])}ms")
    st.subheader("Query × Child rank matrix")
    matrix = query_child_matrix(result.get("child_hits", []))
    if matrix:
        st.dataframe(matrix, use_container_width=True, hide_index=True)
    else:
        st.info("Không có child hit matrix cho result này.")


def _tab_parent_child(st: Any) -> None:
    st.header("Parent–Child Explorer")
    result = st.session_state.last_parent_result or st.session_state.last_answer_result
    if not result:
        st.info("Chạy parent mode trước để xem cây parent–child.")
        return
    nodes = parent_tree_data(result)
    if not nodes:
        st.info("Không có parent candidate trong result hiện tại.")
        return
    trace = result.get("trace", {}).get("retrieval_trace", result.get("trace", {}))
    st.metric("Context expansion factor", fmt(trace.get("context_expansion_factor")))
    for node in nodes:
        title = f"Parent {node.get('evidence_id') or node['parent_id']} · rank {node['parent_rank']} → {node['parent_rerank_rank']} · {node['source']} {node['pages']}"
        with st.expander(title, expanded=bool(node.get("ambiguous") or node.get("warnings"))):
            if node.get("ambiguous") or node.get("warnings"):
                st.warning(f"Ambiguous/warnings: {node.get('warnings')}")
            c1, c2 = st.columns(2)
            c1.write({"structural_path": node["structural_path"], "parent_rrf_score": node["parent_rrf_score"]})
            c2.write({"parent_rerank_score": node["parent_rerank_score"], "children": len(node["children"])})
            st.markdown("**Supporting children**")
            for child in node["children"]:
                st.write(f"- `{child['child_id']}` queries={child['query_ids']} ranks={child['query_ranks']} anchor={child['is_anchor']}")
                st.caption(child["anchor_snippet"])
            with st.expander("Parent text", expanded=False):
                st.text(node["text"])


def _tab_compare(st: Any, config: hr.HierarchyConfig) -> None:
    st.header("Mode Comparison")
    question = st.text_area("Question", value=st.session_state.last_question, height=100, key="compare_q")
    st.caption("Chạy bốn mode retrieval/rerank; không gọi answer generation và không tuyên bố mode thắng khi không có gold labels.")
    if st.button("Run mode comparison"):
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Comparing four modes..."):
                try:
                    st.session_state.last_compare_result = hr.compare_buoi09(question, config=config)
                except Exception as error:  # noqa: BLE001
                    st.error(safe_error(error))
    result = st.session_state.last_compare_result
    if result:
        rows = [mode_comparison_row(mode, payload) for mode, payload in result["modes"].items()]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.info("Không kết luận mode thắng nếu chưa có gold labels đã duyệt.")


def _tab_evaluation(st: Any) -> None:
    st.header("Evaluation")
    st.caption("Chỉ đọc latest report; không tự chạy evaluator khi render.")
    path = latest_report_path()
    if path is None:
        st.info("Chưa có report JSON trong reports/.")
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        st.error(safe_error(error))
        return
    st.write(f"Latest report: `{path.name}`")
    if report_needs_review(report):
        st.warning("Gold labels còn needs_human_review=true; không tuyên bố mode thắng chính thức.")
    rows = report_metric_rows(report)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Report chưa có metric schema nhận diện được.")
    with st.expander("Raw report"):
        st.json(report)


if __name__ == "__main__":
    main()
