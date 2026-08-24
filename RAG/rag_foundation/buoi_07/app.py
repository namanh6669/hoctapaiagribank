"""Streamlit UI Buổi 07 cho demo RAG."""

from __future__ import annotations

import streamlit as st

import rag


STRATEGIES = ["hierarchical", "semantic", "fixed-size"]


st.set_page_config(page_title="Buổi 07 - RAG", page_icon="🔎", layout="wide")


if "last_index_result" not in st.session_state:
    st.session_state.last_index_result = None
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None


def safe_error(error: Exception, config: dict | None = None) -> str:
    message = f"{type(error).__name__}: {error}"
    api_key = (config or {}).get("api_key", "")
    if api_key:
        message = message.replace(api_key, "[secret]")
    return message[:500]


def page_text(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"tr. {page_start}"
    return f"tr. {page_start}-{page_end}"


def evidence_title(item: dict) -> str:
    return f"{item['source']} – {page_text(item['page_start'], item['page_end'])} – {item['chunk_id']}"


def status_badge(status: str) -> None:
    if status == "answered":
        st.success("answered - Đã tạo câu trả lời từ evidence đạt ngưỡng.")
    elif status == "insufficient_evidence":
        st.warning("insufficient_evidence - Không tìm thấy đủ thông tin liên quan.")
    elif status == "retrieval_only":
        st.info("retrieval_only - Đã retrieve được nguồn nhưng chưa thể tạo câu trả lời tổng hợp.")
    else:
        st.warning(f"Trạng thái chưa biết: {status}")


def render_evidence(evidence: list[dict]) -> None:
    st.subheader("Nguồn tham khảo")
    st.caption(
        "Distance thấp hơn thường liên quan hơn trong demo cosine distance; "
        "đây không phải xác suất hay độ tin cậy tuyệt đối."
    )
    if not evidence:
        st.info("Chưa có evidence.")
        return

    for item in evidence:
        accepted_text = "Đạt confidence gate" if item.get("accepted") else "Không đạt confidence gate"
        title = f"{item['evidence_id']} · {evidence_title(item)} · {item['distance']:.4f} · {accepted_text}"
        with st.expander(title, expanded=item.get("accepted", False)):
            st.write("**evidence_id:**", item["evidence_id"])
            st.write("**source:**", item["source"])
            st.write("**page:**", page_text(item["page_start"], item["page_end"]))
            st.write("**chunk_id:**", item["chunk_id"])
            st.write("**distance:**", f"{item['distance']:.6f}")
            if item.get("accepted"):
                st.success("accepted: đạt confidence gate và có thể được dùng trong generation prompt.")
            else:
                st.warning("accepted: không đạt confidence gate, không được dùng để tạo answer.")
            st.text_area("Nội dung chunk", item.get("text", ""), height=240, key=f"evidence_{item['evidence_id']}_{item['chunk_id']}")


try:
    config = rag.load_config()
except Exception as error:  # noqa: BLE001 - UI hiển thị lỗi gọn cho người học
    config = None
    st.error(f"Không đọc được cấu hình: {safe_error(error)}")

with st.sidebar:
    st.header("Cấu hình")
    strategy = st.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index(rag.DEFAULT_STRATEGY))
    top_k = st.number_input("Top-k", min_value=1, max_value=10, value=5, step=1)

status = None
if config:
    try:
        status = rag.collection_status(strategy=strategy)
    except Exception as error:  # noqa: BLE001
        st.sidebar.error(f"Không đọc được status: {safe_error(error, config)}")

with st.sidebar:
    st.divider()
    st.subheader("Trạng thái hệ thống")
    if config:
        st.write("**API key:**", config["api_key_status"])
        st.write("**Embedding model:**", config["embedding_model"])
        st.write("**Embedding dimension:**", config["embedding_dim"])
        st.write("**Generation model:**", config["generation_model"])
        st.write("**RAG_MAX_DISTANCE:**", config["max_distance"])
    else:
        st.write("**API key:** Không đọc được cấu hình")

    st.write("**Strategy đang chọn:**", strategy)
    if status:
        st.write("**Collection:**", status["collection_name"])
        st.write("**Collection tồn tại:**", "Có" if status["exists"] else "Chưa")
        st.write("**Số chunk:**", status["count"])
        if status.get("warning"):
            st.caption(status["warning"])

st.title("🔎 Buổi 07 - RAG với Gemini Embedding và ChromaDB")
st.caption("UI tiếng Việt cho người mới học RAG. App chỉ gọi các hàm có sẵn trong rag.py.")

st.markdown("## 1. Index dữ liệu")
reset = st.checkbox("Reset collection trước khi index", value=False)

if st.button("Index dữ liệu"):
    if not config:
        st.error("Chưa đọc được cấu hình. Hãy kiểm tra file .env.")
    elif config["api_key_status"] == "Thiếu":
        st.warning("Thiếu GEMINI_API_KEY. Hãy điền key vào rag_foundation/buoi_07/.env rồi chạy lại.")
    else:
        before_count = status["count"] if status else 0
        with st.spinner("Đang tạo embedding và index vào ChromaDB..."):
            try:
                result = rag.index_chunks(strategy=strategy, reset=reset)
                result["before_count"] = before_count
                st.session_state.last_index_result = result
                st.session_state.last_query_result = None
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(f"Index lỗi: {safe_error(error, config)}")

if st.session_state.last_index_result:
    result = st.session_state.last_index_result
    st.success("Index hoàn tất")
    cols = st.columns(5)
    cols[0].metric("Strategy", result["strategy"])
    cols[1].metric("Chunk trước", result.get("before_count", 0))
    cols[2].metric("Chunk sau", result["count"])
    cols[3].metric("Text rỗng bỏ qua", result["load_stats"].get("empty_text_skipped", 0))
    cols[4].metric("Valid chunks", result["load_stats"].get("valid_chunks", 0))
    st.write("**Collection:**", result["collection_name"])

st.markdown("## 2. Đặt câu hỏi")
question = st.text_area("Nhập câu hỏi", placeholder="Ví dụ: Điều kiện cơ cấu lại thời hạn trả nợ là gì?", height=120)

if st.button("Gửi câu hỏi"):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi trước khi gửi.")
    elif not config:
        st.error("Chưa đọc được cấu hình. Hãy kiểm tra file .env.")
    elif config["api_key_status"] == "Thiếu":
        st.warning("Thiếu GEMINI_API_KEY. Hãy điền key vào rag_foundation/buoi_07/.env rồi chạy lại.")
    elif not status or not status["exists"]:
        st.warning("Collection chưa tồn tại. Hãy index dữ liệu trước.")
    elif status["count"] < 1:
        st.warning("Collection đang rỗng. Hãy index dữ liệu trước.")
    else:
        with st.spinner("Đang retrieval và tạo câu trả lời..."):
            try:
                st.session_state.last_query_result = rag.answer_question(
                    question,
                    top_k=int(top_k),
                    strategy=strategy,
                )
            except Exception as error:  # noqa: BLE001
                st.error(f"Query lỗi: {safe_error(error, config)}")

result = st.session_state.last_query_result
if result:
    st.markdown("## 3. Answer")
    status_badge(result["status"])

    if result["warnings"]:
        st.warning("Cảnh báo:")
        for warning in result["warnings"]:
            st.write(f"- {warning}")

    if result["status"] == "insufficient_evidence":
        st.info(result["answer"])
    elif result["status"] == "retrieval_only":
        st.info(result["answer"])
    else:
        st.write(result["answer"])

    st.markdown("### Citations")
    if result["citations"]:
        st.json(result["citations"])
    else:
        st.info("Không có citation inline được map. Xem Nguồn tham khảo bên dưới.")

    render_evidence(result["evidence"])
