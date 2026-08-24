"""app_secure.py — Buổi 15.

Streamlit Web App minh họa **kiểm soát truy cập theo vai trò (RBAC)**
trên hệ thống RAG lai (BM25 + Dense + Hybrid + Rerank).

Khác biệt so với ``buoi_14/app.py``:
  • Sidebar có Multiselect **Vai trò của bạn (Your Roles)** — chọn 1+ role.
  • Mọi retrieval (BM25 / Dense / Graph / Hybrid / Rerank) đều truyền
    user_roles vào :class:`SecureRetriever`.
  • Mỗi kết quả hiển thị rõ nhãn bảo mật (``security_label``) +
    danh sách roles được phép xem.
  • Stats panel: "Đã lọc bỏ X kết quả do không đủ quyền".
  • Graph hints (1-hop) cũng bị filter bởi RBAC — không leak doc ID
    của tài liệu cấm.

Chạy:   streamlit run app_secure.py
Dừng:   Ctrl+C trong terminal.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env ngay khi khởi động (trước cả import secure_retriever).
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
except Exception:
    pass

import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.models").setLevel(logging.CRITICAL)

# Silence transformers' lazy-scan vision-module noise.
try:
    from src._quiet_transformers import install_quiet_print_once  # type: ignore
    install_quiet_print_once()
except Exception:
    pass

import streamlit as st

from src.config import ROLE_LIST, VALID_ROLES
from src.secure_retriever import SecureRetriever, normalize_roles


# ----------------------------------------------------------------------------
# Streamlit config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Secure (RBAC) — Buổi 15",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 RAG Secure (RBAC) — Buổi 15")
st.caption(
    "Demo minh họa kiểm soát truy cập theo vai trò: BM25 · Dense · Hybrid (RRF) · "
    "Hybrid + Rerank — **mọi kết quả đều được lọc theo role bạn chọn ở sidebar**. "
    "Graph Cypher cũng áp dụng `WHERE any(role IN node.allowed_roles WHERE role IN $user_roles)`."
)

METHOD_OPTIONS = {
    "BM25": "bm25",
    "Dense": "dense",
    "Hybrid": "hybrid",
    "Hybrid + Rerank": "hybrid_rerank",
}


# ----------------------------------------------------------------------------
# Cache SecureRetriever theo user_roles (Streamlit cache_resource)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang khởi tạo SecureRetriever...")
def _get_secure_retriever(tuple_roles: tuple[str, ...], use_dense: bool, use_rerank: bool):
    """Cache theo (sorted user_roles, use_dense, use_rerank)."""
    return SecureRetriever(
        user_roles=list(tuple_roles),
        use_dense=use_dense,
        use_rerank=use_rerank,
        force_fallback=False,
    )


def _color_for_label(label: str) -> str:
    """Mỗi security_label có 1 màu riêng."""
    return {
        "General": "🟢",
        "Risk": "�",
        "HR": "🔴",
        "Restricted": "🟣",
    }.get(label, "⚪")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔐 Vai trò của bạn (Your Roles)")
    st.caption(
        "Chọn 1 hoặc nhiều role. Hệ thống sẽ chỉ trả về chunks mà **ít nhất 1** "
        "role của bạn có trong ``allowed_roles``."
    )
    chosen_roles = st.multiselect(
        "Chọn role",
        options=list(ROLE_LIST),
        default=["Guest"],
        help=(
            "Admin       → thấy tất cả\n"
            "HR_Manager  → Confidential trở xuống\n"
            "Risk_Officer→ Confidential trở xuống\n"
            "Employee    → Internal trở xuống\n"
            "Guest       → chỉ Public"
        ),
    )
    if not chosen_roles:
        st.warning("⚠️ Chọn ít nhất 1 role để bắt đầu tìm kiếm.")
        st.stop()

    normalized = normalize_roles(chosen_roles)
    if not normalized:
        st.error(f"Role không hợp lệ: {chosen_roles}. Hợp lệ: {list(ROLE_LIST)}")
        st.stop()
    st.success(f"Đang tìm với roles: **{', '.join(normalized)}**")

    st.divider()
    st.header("⚙️ Cấu hình retrieval")
    query = st.text_area(
        "Câu hỏi",
        height=100,
        placeholder="Nhập câu hỏi tiếng Việt... ví dụ: 'quy định về hạn mức tín dụng'",
    )
    method_label = st.selectbox(
        "Method",
        list(METHOD_OPTIONS.keys()),
        index=2,  # mặc định Hybrid
        help="Chọn retrieval pipeline.",
    )
    top_k = st.slider("Top-k", min_value=1, max_value=20, value=5, step=1)
    candidate_k = st.slider(
        "Candidate-k (Hybrid path)",
        min_value=10, max_value=50, value=20, step=5,
        help="Số candidate Hybrid lấy trước khi rerank.",
    )

    use_dense = st.checkbox(
        "Bật Dense retrieval",
        value=True,
        help="Tắt để chỉ dùng BM25 (nhanh hơn, không cần tải embedding model).",
    )
    use_rerank = st.checkbox(
        "Bật Rerank (Cross-Encoder)",
        value=False,
        help="Tốn ~2.3 GB tải về lần đầu.",
    )

    run_clicked = st.button("🔍 Tìm kiếm", type="primary", width="stretch")


# ----------------------------------------------------------------------------
# Main — chạy retrieval khi click
# ----------------------------------------------------------------------------
if not run_clicked:
    st.info(
        "👈 Chọn **role** ở sidebar, nhập câu hỏi, rồi bấm **Tìm kiếm**.\n\n"
        "Tip: đổi role rồi bấm tìm lại cùng câu hỏi để thấy kết quả thay đổi."
    )
    st.stop()

if not query or not query.strip():
    st.warning("Vui lòng nhập câu h�i.")
    st.stop()

method = METHOD_OPTIONS[method_label]

with st.spinner(f"Đang chạy `{method_label}` với role {normalized}..."):
    try:
        sr = _get_secure_retriever(
            tuple(sorted(normalized)),
            use_dense=use_dense,
            use_rerank=use_rerank,
        )
        total_corpus = len(sr.df)
        allowed_corpus = len(sr.bm25.df_allowed)
        filtered_out_corpus = total_corpus - allowed_corpus

        # Run the chosen method
        if method == "bm25":
            results = sr.search_bm25(query, top_k=top_k)
            meta = {"method": method_label}
        elif method == "dense":
            if not use_dense:
                st.warning("Dense đang tắt — chuyển sang BM25.")
                results = sr.search_bm25(query, top_k=top_k)
                meta = {"method": "BM25 (fallback)"}
            else:
                results = sr.search_dense(query, top_k=top_k)
                meta = {"method": method_label}
        elif method == "hybrid":
            results = sr.search_hybrid(query, top_k=top_k, candidate_k=candidate_k)
            meta = {"method": method_label}
        else:  # hybrid_rerank
            if not use_rerank:
                st.warning("Rerank đang tắt — chuyển sang Hybrid.")
                results = sr.search_hybrid(query, top_k=top_k, candidate_k=candidate_k)
                meta = {"method": "Hybrid (fallback — rerank disabled)"}
            else:
                cand, results, meta = sr.search_with_rerank(
                    query, top_k=top_k, candidate_k=candidate_k
                )
                meta = {"method": method_label, **meta}

        # Đếm số candidate bị filter ra khỏi "top-k hiển thị" (post-RBAC pre-cap).
        # Lưu ý: secure_retriever đã lọc ở mọi tầng, nên con số này là các chunk
        # có điểm cao nhưng KHÔNG nằm trong top-k (không phải bị filter bởi RBAC).
        # Để đếm filter-out do RBAC, ta so với "kết quả nếu là Admin".
        if run_clicked and method != "hybrid_rerank":
            try:
                sr_admin = _get_secure_retriever(
                    ("Admin",), use_dense=use_dense, use_rerank=use_rerank
                )
                if method == "bm25":
                    res_admin = sr_admin.search_bm25(query, top_k=top_k)
                elif method == "dense":
                    res_admin = sr_admin.search_dense(query, top_k=top_k)
                else:
                    res_admin = sr_admin.search_hybrid(query, top_k=top_k, candidate_k=candidate_k)
                visible_user = {r["chunk_id"] for r in results}
                visible_admin = {r["chunk_id"] for r in res_admin}
                hidden_due_to_rbac = visible_admin - visible_user
                n_hidden = len(hidden_due_to_rbac)
            except Exception:
                n_hidden = 0
        else:
            n_hidden = 0

    except Exception as e:
        st.error(f"Lỗi khi chạy retrieval: {type(e).__name__}: {e}")
        import traceback
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()


# ----------------------------------------------------------------------------
# Stats panel — góc phải màn hình
# ----------------------------------------------------------------------------
col_stats, _ = st.columns([1, 3])
with col_stats:
    if n_hidden > 0:
        st.warning(
            f"🔒 Đã lọc b� **{n_hidden}** kết quả do không đủ quyền truy cập "
            f"(so với role Admin)."
        )
    else:
        st.success(
            f"✅ Vai trò **{', '.join(normalized)}** xem được tất cả "
            f"{len(results)} kết quả top-k."
        )
    st.caption(
        f"Corpus: **{allowed_corpus}/{total_corpus}** chunks được phép truy cập "
        f"({filtered_out_corpus} bị ẩn bởi RBAC ở mọi tầng retrieval)."
    )


# ----------------------------------------------------------------------------
# Kết quả chính
# ----------------------------------------------------------------------------
st.subheader(
    f"Kết quả · method = **{method_label}** · roles = **{', '.join(normalized)}** · top_k = {top_k}"
)

if not results:
    st.info(
        f"Không có kết quả nào cho câu hỏi này với role **{', '.join(normalized)}**. "
        f"Thử chọn role có nhiều quyền hơn (Admin / HR_Manager / Risk_Officer)."
    )
else:
    for r in results:
        rno = r.get("final_rank") or r.get("rank")
        cid_short = (r.get("chunk_id") or "")[:8]
        doc = (r.get("document_id") or "")[:18]
        score = r.get("retrieval_score") or r.get("rrf_score") or r.get("rerank_score")
        score_str = f"{score:+.4f}" if score is not None else "—"

        label = r.get("security_label", "?")
        roles_seen = r.get("allowed_roles", [])
        icon = _color_for_label(label)

        # Sub-info
        extras: list[str] = []
        if r.get("bm25_rank") is not None:
            extras.append(f"bm25={r['bm25_rank']}")
        if r.get("dense_rank") is not None:
            extras.append(f"dense={r['dense_rank']}")
        if r.get("graph_rank") is not None:
            extras.append(f"graph={r['graph_rank']}")
        if r.get("rrf_score") is not None:
            extras.append(f"rrf={r['rrf_score']:+.4f}")
        if r.get("rerank_score") is not None:
            extras.append(f"rerank={r['rerank_score']:+.4f}")
        method_tag = r.get("retrieval_method", method_label)
        extras.append(f"method={method_tag}")
        extra_str = " · ".join(extras)

        with st.container(border=True):
            header = (
                f"{icon} **#{rno}** · `chunk={cid_short}` · `doc={doc}` · "
                f"score={score_str} · "
                f"**Quyền xem:** `[{', '.join(roles_seen) if roles_seen else '—'}]` · "
                f"label=`{label}`"
            )
            st.markdown(header)
            if extra_str:
                st.markdown(f"<sub>{extra_str}</sub>", unsafe_allow_html=True)
            cite = r.get("citation") or ""
            st.markdown(f"**Citation:** `{cite}`")
            text = (r.get("text") or "").replace("\n", "  \n")
            st.markdown(f"**Text:**")
            st.markdown(
                f"> {text[:600]}{'…' if len(text) > 600 else ''}"
            )


# ----------------------------------------------------------------------------
# Graph hints — RBAC-aware: chỉ liệt kê doc_ids user được thấy
# ----------------------------------------------------------------------------
st.divider()
st.subheader("🕸️ Graph hints (1-hop) — RBAC filtered")
doc_ids: list[str] = []
seen: set[str] = set()
for r in results:
    d = r.get("document_id") or ""
    if d and d not in seen:
        seen.add(d)
        doc_ids.append(d)

chunk_ids = [r.get("chunk_id") for r in results if r.get("chunk_id")]

if not doc_ids:
    st.write("—")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Document IDs (đã lọc theo role):**")
        for d in doc_ids:
            st.code(d, language="text")
    with col2:
        st.markdown("**Chunk IDs:**")
        for c in chunk_ids:
            st.code((c or "")[:8] + "…", language="text")

    # Truy vấn Cypher 1-hop có RBAC filter (chỉ liệt kê các cung lộ ra cho user)
    cfg_uri = os.environ.get("NEO4J_URI", "")
    cfg_pwd = os.environ.get("NEO4J_PASSWORD", "")
    if not (cfg_uri and cfg_pwd):
        st.info(
            "🔌 Neo4j chưa cấu hình trong `.env` — graph hints bị lược bỏ."
        )
    else:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                cfg_uri,
                auth=(os.environ.get("NEO4J_USER", "neo4j"), cfg_pwd),
            )
            with driver.session(
                database=os.environ.get("NEO4J_DATABASE", "neo4j")
            ) as s:
                # Cypher 1-hop với RBAC WHERE clause trên VanBan (start node)
                # và target node. CASE WHEN chọn field hiển thị theo label để
                # tránh Neo4j warning "property/label không tồn tại".
                cypher = (
                    "MATCH (vb:VanBan {lab_session:'buoi_15'})-[r]->(n) "
                    "WHERE vb.id IN $ids "
                    "  AND any(role IN vb.allowed_roles WHERE role IN $roles) "
                    "  AND (n.allowed_roles IS NULL OR "
                    "       any(role IN n.allowed_roles WHERE role IN $roles)) "
                    "RETURN vb.id AS src_id, "
                    "       CASE "
                    "         WHEN vb.so_ky_hieu IS NOT NULL THEN vb.so_ky_hieu "
                    "         WHEN vb.title IS NOT NULL THEN vb.title "
                    "         ELSE vb.id "
                    "       END AS src_label, "
                    "       labels(n)[0] AS n_label, "
                    "       CASE labels(n)[0] "
                    "         WHEN 'NguoiKy' THEN coalesce(n.canonical_name, n.id) "
                    "         WHEN 'CoQuan' THEN coalesce(n.canonical_name, n.id) "
                    "         WHEN 'LinhVuc' THEN coalesce(n.canonical_name, n.id) "
                    "         WHEN 'DoiTuongApDung' THEN coalesce(n.canonical_name, n.id) "
                    "         WHEN 'Document' THEN coalesce(n.title, n.id) "
                    "         WHEN 'VanBan' THEN coalesce(n.title, n.id) "
                    "         WHEN 'Chunk' THEN left(coalesce(n.text, n.id), 40) "
                    "         ELSE coalesce(n.id, '(unknown)') "
                    "       END AS target, "
                    "       type(r) AS rel, "
                    "       coalesce(r.confidence, 1.0) AS conf "
                    "ORDER BY src_label, rel"
                )
                rows = list(s.run(cypher, ids=doc_ids, roles=normalized))
            driver.close()

            if not rows:
                st.write(
                    "_Không có edge 1-hop nào cho các doc này trong Neo4j "
                    "(hoặc các edge đều dẫn tới node mà role hiện tại không được xem)._"
                )
            else:
                st.markdown(
                    f"**{len(rows)} edge 1-hop** (đã lọc — chỉ liệt kê các edge "
                    f"mà role `{', '.join(normalized)}` được phép xem):"
                )
                for row in rows[:30]:  # cap hiển thị
                    st.code(
                        f"(VanBan {row['src_label']}) -[:{row['rel']} "
                        f"conf={row['conf']:.2f}]-> ({row['n_label']} {row['target']})",
                        language="cypher",
                    )
        except Exception as e:
            st.warning(f"Lỗi khi truy vấn Neo4j: {e}")


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "🔐 **RBAC**: mọi candidate phải vượt qua filter ở cấp retriever "
    "(BM25 pre-filter, Dense post-filter, Cypher WHERE) trước khi vào RRF / rerank. "
    "Đổi role ở sidebar → cache được rebuild theo key mới."
)
