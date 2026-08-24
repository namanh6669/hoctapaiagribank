import os

import streamlit as st
from dotenv import load_dotenv

import rag


load_dotenv(rag.ENV_PATH)


st.set_page_config(page_title="RAG Buổi 06", page_icon="🔎", layout="wide")


def postgres_status():
    store = rag._postgres_store()
    if not store:
        return "Không kết nối được"
    store.connection.close()
    return "Đang chạy"


def chromadb_status():
    return rag._chroma_mode()


def gemini_status():
    return "Có" if os.getenv("GEMINI_API_KEY") else "Thiếu"


def search_top_k(question, k):
    collection = rag._chroma_collection()
    gemini = rag._gemini_client()

    if gemini:
        question_embedding = rag._embed(question, gemini)
        results = collection.query(query_embeddings=[question_embedding], n_results=k)
    else:
        results = collection.query(query_texts=[question], n_results=k)

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    text_store = rag._text_store()
    rows = text_store.get_many(ids)
    by_id = {row["id"]: row for row in rows}

    top_k = []
    for position, chunk_id in enumerate(ids, start=1):
        row = by_id.get(chunk_id)
        if not row:
            continue
        distance = distances[position - 1] if position - 1 < len(distances) else None
        top_k.append(
            {
                "rank": position,
                "id": chunk_id,
                "source": row.get("source", ""),
                "text": row.get("text", ""),
                "distance": distance,
            }
        )
    return top_k


with st.sidebar:
    st.header("Trạng thái")
    st.write("**PostgreSQL:**", postgres_status())
    st.write("**ChromaDB:**", chromadb_status())
    st.write("**Gemini API Key:**", gemini_status())

    try:
        current_status = rag.status()
        st.divider()
        st.write("**Documents:**", current_status.get("documents", 0))
        st.write("**Chunks:**", current_status.get("chunks", 0))
    except Exception as error:
        st.warning(f"Chưa đọc được status: {type(error).__name__}")


st.title("RAG Demo - Buổi 06")
st.caption("Pipeline: Question ➔ Top-k ➔ Gemini ➔ Answer")

if st.button("Index"):
    progress_box = st.empty()
    progress_bar = st.progress(0)

    def show_progress(message):
        progress_box.write(message)
        if "Đã index" in message:
            current, total = message.split("Đã index ", 1)[1].split(" chunks", 1)[0].split("/")
            progress_bar.progress(int(current) / int(total))

    with st.spinner("Đang index dữ liệu..."):
        try:
            result = rag.index(progress=show_progress)
            progress_bar.progress(1.0)
            st.success("Index xong")
            st.json(result)
        except Exception as error:
            st.error(f"Index lỗi: {type(error).__name__}: {error}")

st.subheader("Question")
question = st.text_area("Nhập câu hỏi", placeholder="Ví dụ: Nội dung chính của tài liệu là gì?")
k = st.number_input("Top-k", min_value=1, max_value=10, value=3, step=1)

if st.button("Ask"):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        st.subheader("Top-k")
        with st.spinner("Đang retrieval..."):
            try:
                top_k = search_top_k(question, int(k))
            except Exception as error:
                st.error(f"Retrieval lỗi: {type(error).__name__}: {error}")
                top_k = []

        if top_k:
            for item in top_k:
                title = f"#{item['rank']} - {item['source']}"
                if item["distance"] is not None:
                    title += f" - distance: {item['distance']:.4f}"
                with st.expander(title, expanded=item["rank"] == 1):
                    st.write(item["text"])
        else:
            st.info("Chưa tìm thấy kết quả phù hợp. Hãy chạy Index trước.")

        st.subheader("Answer")
        if gemini_status() == "Thiếu":
            st.info("Thiếu GEMINI_API_KEY nên chỉ hiển thị Retrieval, không gọi Gemini.")
        elif top_k:
            with st.spinner("Đang gọi Gemini..."):
                try:
                    answer = rag.ask(question, k=int(k))
                    st.write(answer)
                except Exception as error:
                    st.error(f"Gemini lỗi: {type(error).__name__}: {error}")
