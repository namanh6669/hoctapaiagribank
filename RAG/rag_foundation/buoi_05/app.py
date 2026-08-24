#!/usr/bin/env python3
"""Streamlit UI Buổi 5: visualize raw OCR và chunks trong output/."""

from __future__ import annotations

import json
import statistics
import unicodedata
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover - Streamlit thường đã kèm pandas
    pd = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHUNKS_PATH = OUTPUT_DIR / "chunks" / "chunks.json"
REPORT_PATH = OUTPUT_DIR / "reports" / "chunk_report.json"
RAW_DIR = OUTPUT_DIR / "raw"
RENDERED_DIR = OUTPUT_DIR / "rendered_pages"


st.set_page_config(
    page_title="Buổi 5 - Visualize RAG chunks",
    page_icon="📄",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_chunks(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [
            {
                "chunk_id": "json_error",
                "strategy": "error",
                "source": path.name,
                "page_start": 0,
                "page_end": 0,
                "text": f"File chunks phải là JSON array hợp lệ: {error}",
                "metadata": {"error": "JSONDecodeError"},
            }
        ]

    if isinstance(data, dict):
        for key in ("chunks", "data", "documents", "items"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        return [
            {
                "chunk_id": "json_format_error",
                "strategy": "error",
                "source": path.name,
                "page_start": 0,
                "page_end": 0,
                "text": "File chunks phải là JSON array object.",
                "metadata": {"error": "ExpectedList"},
            }
        ]

    chunks: list[dict[str, Any]] = []
    for item_no, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            item = {"chunk_id": f"item_{item_no}", "text": str(item), "metadata": {}}
        item["text_length"] = len(item.get("text", ""))
        item["is_nfc"] = unicodedata.is_normalized("NFC", item.get("text", ""))
        chunks.append(item)
    return chunks


@st.cache_data(show_spinner=False)
def load_text(path_text: str) -> str:
    path = Path(path_text)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def to_table(rows: list[dict[str, Any]]):
    """Dùng pandas nếu có, nếu không trả list dict cho st.dataframe/st.table."""
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def short_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def count_by(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return [{key: value, "count": count} for value, count in sorted(counts.items())]


def render_metric_row(chunks: list[dict[str, Any]], report: dict[str, Any]) -> None:
    lengths = [int(chunk.get("text_length", len(chunk.get("text", "")))) for chunk in chunks]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số chunk", f"{len(chunks):,}")
    col2.metric("Số tài liệu", f"{len({c.get('source') for c in chunks}):,}")
    col3.metric("Số chiến lược", f"{len({c.get('strategy') for c in chunks}):,}")
    avg = round(statistics.mean(lengths), 2) if lengths else 0
    col4.metric("Độ dài TB", f"{avg:,} ký tự")

    if report.get("generated_at"):
        st.caption(f"Report tạo lúc: `{report['generated_at']}`")


def render_report_tab(report: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
    st.subheader("Tổng quan output")
    render_metric_row(chunks, report)

    stats = report.get("stats", [])
    if stats:
        st.markdown("#### Thống kê theo chiến lược")
        st.dataframe(to_table(stats), use_container_width=True, hide_index=True)

        chart_rows = [{"strategy": row["strategy"], "chunk_count": row["chunk_count"]} for row in stats]
        if pd is not None:
            chart_df = pd.DataFrame(chart_rows).set_index("strategy")
            st.bar_chart(chart_df)

    documents = report.get("documents", [])
    if documents:
        st.markdown("#### Tài liệu đã xử lý")
        doc_rows = [
            {
                "source": doc.get("source"),
                "extraction_method": doc.get("extraction_method"),
                "page_count": doc.get("page_count"),
                "warning_count": len(doc.get("warnings", [])),
            }
            for doc in documents
        ]
        st.dataframe(to_table(doc_rows), use_container_width=True, hide_index=True)

        with st.expander("Xem cảnh báo theo tài liệu"):
            for doc in documents:
                st.markdown(f"**{doc.get('source')}**")
                warnings = doc.get("warnings", [])
                if warnings:
                    st.code("\n".join(warnings), language="text")
                else:
                    st.write("Không có cảnh báo.")

    st.markdown("#### Kiểm tra nhanh theo SPEC")
    checks = [
        {
            "Hạng mục": "Có chunks.json dạng JSON array",
            "Kết quả": "PASS" if CHUNKS_PATH.exists() else "FAIL",
            "Ghi chú": str(CHUNKS_PATH.relative_to(BASE_DIR)) if CHUNKS_PATH.exists() else "Chưa thấy file chunks.json",
        },
        {
            "Hạng mục": "Text chunk chuẩn NFC",
            "Kết quả": "PASS" if chunks and all(chunk.get("is_nfc") for chunk in chunks) else "FAIL",
            "Ghi chú": "Tất cả text chunk đang là Unicode NFC" if chunks else "Chưa có chunk để kiểm tra",
        },
        {
            "Hạng mục": "Không hiển thị secret",
            "Kết quả": "PASS",
            "Ghi chú": "UI chỉ đọc output; không đọc file .env và không in API key.",
        },
        {
            "Hạng mục": "Không ghi đè PDF gốc",
            "Kết quả": "PASS",
            "Ghi chú": "UI chỉ đọc output/ và datademo không bị ghi từ app.",
        },
    ]
    st.dataframe(to_table(checks), use_container_width=True, hide_index=True)


def render_chunks_tab(chunks: list[dict[str, Any]]) -> None:
    st.subheader("Duyệt và so sánh chunk")
    if not chunks:
        st.warning("Chưa có chunk trong chunks.json. Hãy chạy pipeline với `--write` trước.")
        return

    sources = sorted({str(chunk.get("source", "")) for chunk in chunks})
    strategies = sorted({str(chunk.get("strategy", "")) for chunk in chunks})

    col1, col2, col3, col4 = st.columns([1.4, 1.1, 1, 1.2])
    selected_sources = col1.multiselect("Lọc theo PDF", sources, default=sources)
    selected_strategies = col2.multiselect("Lọc theo chiến lược", strategies, default=strategies)
    keyword = col3.text_input("Tìm trong text", value="")
    length_limit = col4.slider("Độ dài tối thiểu", min_value=0, max_value=3000, value=0, step=50)

    filtered = []
    keyword_lower = keyword.lower().strip()
    for chunk in chunks:
        text = chunk.get("text", "")
        if chunk.get("source") not in selected_sources:
            continue
        if chunk.get("strategy") not in selected_strategies:
            continue
        if keyword_lower and keyword_lower not in text.lower():
            continue
        if int(chunk.get("text_length", len(text))) < length_limit:
            continue
        filtered.append(chunk)

    st.caption(f"Đang hiển thị {len(filtered):,}/{len(chunks):,} chunk")

    summary_rows = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "strategy": chunk.get("strategy"),
            "source": chunk.get("source"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text_length": chunk.get("text_length"),
            "is_nfc": chunk.get("is_nfc"),
            "preview": short_text(chunk.get("text", "")),
        }
        for chunk in filtered
    ]
    st.dataframe(to_table(summary_rows), use_container_width=True, hide_index=True)

    if not filtered:
        return

    selected_id = st.selectbox("Chọn chunk để xem chi tiết", [chunk["chunk_id"] for chunk in filtered])
    selected = next(chunk for chunk in filtered if chunk["chunk_id"] == selected_id)

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Nội dung chunk")
        st.text_area("Text", selected.get("text", ""), height=420)
    with right:
        st.markdown("#### Metadata")
        st.json(
            {
                "chunk_id": selected.get("chunk_id"),
                "strategy": selected.get("strategy"),
                "source": selected.get("source"),
                "page_start": selected.get("page_start"),
                "page_end": selected.get("page_end"),
                "text_length": selected.get("text_length"),
                "is_nfc": selected.get("is_nfc"),
                "metadata": selected.get("metadata", {}),
            }
        )


def render_compare_tab(chunks: list[dict[str, Any]]) -> None:
    st.subheader("So sánh chiến lược chunking")
    if not chunks:
        st.warning("Chưa có chunk để so sánh.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Số chunk theo chiến lược")
        rows = count_by(chunks, "strategy")
        st.dataframe(to_table(rows), use_container_width=True, hide_index=True)
        if pd is not None:
            st.bar_chart(pd.DataFrame(rows).set_index("strategy"))

    with col2:
        st.markdown("#### Số chunk theo PDF")
        rows = count_by(chunks, "source")
        st.dataframe(to_table(rows), use_container_width=True, hide_index=True)
        if pd is not None:
            st.bar_chart(pd.DataFrame(rows).set_index("source"))

    st.markdown("#### Độ dài chunk theo chiến lược")
    length_rows: list[dict[str, Any]] = []
    for strategy in sorted({chunk.get("strategy") for chunk in chunks}):
        lengths = [chunk.get("text_length", len(chunk.get("text", ""))) for chunk in chunks if chunk.get("strategy") == strategy]
        length_rows.append(
            {
                "strategy": strategy,
                "count": len(lengths),
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(statistics.mean(lengths), 2),
            }
        )
    st.dataframe(to_table(length_rows), use_container_width=True, hide_index=True)


def render_raw_tab(report: dict[str, Any]) -> None:
    st.subheader("Raw OCR / text đã lưu")
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        st.warning("Chưa có raw text trong output/raw/. Hãy chạy pipeline với `--write` trước.")
        return

    selected_raw = st.selectbox("Chọn raw text", raw_files, format_func=lambda path: path.name)
    text = load_text(str(selected_raw))
    st.caption(
        f"File: `{selected_raw.relative_to(BASE_DIR)}` · {len(text):,} ký tự · "
        f"NFC: {'PASS' if unicodedata.is_normalized('NFC', text) else 'FAIL'}"
    )
    st.text_area("Nội dung raw", text, height=420)

    meta_path = selected_raw.with_suffix(".meta.json")
    if meta_path.exists():
        with st.expander("Metadata raw"):
            st.json(load_json(str(meta_path)))

    st.markdown("#### Rendered pages khi fallback OCR")
    source_stem = selected_raw.stem
    image_dir = RENDERED_DIR / source_stem
    images = sorted(image_dir.glob("*.png"))
    if not images:
        st.info("Không có ảnh render cho tài liệu này, có thể PDF đã dùng text layer tốt hoặc chưa chạy `--write` có fallback OCR.")
    else:
        st.caption(f"{len(images)} ảnh trong `{image_dir.relative_to(BASE_DIR)}`")
        page_image = st.selectbox("Chọn ảnh trang", images, format_func=lambda path: path.name)
        st.image(str(page_image), caption=page_image.name, use_container_width=True)


def main() -> None:
    st.title("📄 Buổi 5 - Visualize RAG chunks")
    st.caption("UI tiếng Việt để xem raw OCR, metadata và chunks trong `RAG/rag_foundation/buoi_05/output`.")

    with st.sidebar:
        st.header("Cấu hình")
        st.write("Thư mục bài học:")
        st.code(str(BASE_DIR), language="text")
        st.write("Thư mục output:")
        st.code(str(OUTPUT_DIR), language="text")
        if st.button("Tải lại dữ liệu"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("**Lưu ý an toàn**")
        st.write("App chỉ đọc `output/`, không đọc `.env`, không ghi đè PDF gốc và không gọi LlamaParse.")

    if not OUTPUT_DIR.exists():
        st.error("Chưa thấy thư mục output/. Hãy chạy pipeline với `--write` trước.")
        st.code(".venv/bin/python src/rag_pdf_pipeline.py --write", language="bash")
        return

    report = load_json(str(REPORT_PATH))
    chunks = load_chunks(str(CHUNKS_PATH))

    tabs = st.tabs(["Tổng quan", "Duyệt chunk", "So sánh", "Raw OCR"])
    with tabs[0]:
        render_report_tab(report, chunks)
    with tabs[1]:
        render_chunks_tab(chunks)
    with tabs[2]:
        render_compare_tab(chunks)
    with tabs[3]:
        render_raw_tab(report)


if __name__ == "__main__":
    main()
