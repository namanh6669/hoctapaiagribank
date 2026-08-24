from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import chromadb
import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types
from psycopg import sql
from psycopg.types.json import Jsonb


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = BASE_DIR.parent / "buoi_05" / "output" / "chunks"
ENV_PATH = BASE_DIR / ".env"
STORAGE_DIR = BASE_DIR / "storage"
LOCAL_DB_PATH = STORAGE_DIR / "rag.db"
CHROMA_PATH = STORAGE_DIR / "chroma"

COLLECTION_NAME = "buoi_06_rag"
EMBEDDING_MODEL = "gemini-embedding-2"
ANSWER_MODEL = "gemini-flash-lite-latest"
EMBEDDING_DIMENSIONS = 384


load_dotenv(ENV_PATH)


def index(progress=None):
    """Đọc chunks JSON, lưu text và embedding vào các storage cần thiết."""
    chunks = _load_chunks()
    text_store = _text_store()
    chroma = _chroma_collection()
    gemini = _gemini_client()

    text_store.setup()
    total = len(chunks)

    _log(f"Bắt đầu index {total} chunks", progress)
    _log(f"Text store: {text_store.name}", progress)
    _log(f"ChromaDB: {_chroma_mode()}", progress)

    for position, chunk in enumerate(chunks, start=1):
        text_store.save(chunk)

        embedding = _embed_with_quota_wait(chunk["text"], gemini, position, total, progress) if gemini else None
        metadata = {
            "document_id": chunk["document_id"],
            "source": chunk["source"],
        }

        if embedding:
            chroma.upsert(
                ids=[chunk["id"]],
                documents=[chunk["text"]],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        else:
            chroma.upsert(
                ids=[chunk["id"]],
                documents=[chunk["text"]],
                metadatas=[metadata],
            )

        if position == total or position % 10 == 0:
            _log(f"Đã index {position}/{total} chunks", progress)

    _log("Index hoàn tất", progress)
    return {
        "documents": len({chunk["document_id"] for chunk in chunks}),
        "chunks": len(chunks),
        "text_store": text_store.name,
        "chromadb": _chroma_mode(),
    }


def ask(question, k=5):
    """Truy vấn top-k chunks và trả lời bằng Gemini nếu có API key."""
    if not question or not str(question).strip():
        return "Vui lòng nhập câu hỏi."

    text_store = _text_store()
    chroma = _chroma_collection()
    gemini = _gemini_client()
    k = int(k or 5)

    if gemini:
        question_embedding = _embed(question, gemini)
        results = chroma.query(query_embeddings=[question_embedding], n_results=k)
    else:
        results = chroma.query(query_texts=[question], n_results=k)

    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    rows = text_store.get_many(ids)
    by_id = {row["id"]: row for row in rows}

    contexts = []
    for index_no, chunk_id in enumerate(ids, start=1):
        row = by_id.get(chunk_id)
        if not row:
            continue
        source = row.get("source") or _metadata_value(metadatas, index_no - 1, "source")
        contexts.append(f"[{index_no}] Nguồn: {source}\n{row['text']}")

    if not contexts:
        return "Chưa tìm thấy dữ liệu phù hợp. Hãy chạy index() trước."

    context_text = "\n\n".join(contexts)

    if not gemini:
        return (
            "Chưa có GEMINI_API_KEY nên chỉ thực hiện retrieval, không gọi LLM.\n\n"
            f"Các đoạn liên quan:\n\n{context_text}"
        )

    prompt = f"""Bạn là trợ lý RAG cho workshop người mới.
Chỉ trả lời dựa trên ngữ cảnh bên dưới. Nếu ngữ cảnh không đủ, hãy nói chưa đủ thông tin.

Ngữ cảnh:
{context_text}

Câu hỏi: {question}

Trả lời ngắn gọn, dễ hiểu bằng tiếng Việt.
"""
    response = gemini.models.generate_content(model=ANSWER_MODEL, contents=prompt)
    return response.text or "Không nhận được câu trả lời từ Gemini."


def status():
    """Trả về số lượng document và chunk đang lưu trong text store."""
    text_store = _text_store()
    text_store.setup()
    counts = text_store.counts()
    counts["text_store"] = text_store.name
    counts["chromadb"] = _chroma_mode()
    return counts


class _PostgresStore:
    name = "PostgreSQL"

    def __init__(self, connection):
        self.connection = connection

    def setup(self):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source TEXT,
                    text TEXT NOT NULL,
                    metadata JSONB
                )
                """
            )
        self.connection.commit()

    def save(self, chunk):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO rag_chunks (id, document_id, source, text, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    source = EXCLUDED.source,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata
                """,
                (
                    chunk["id"],
                    chunk["document_id"],
                    chunk["source"],
                    chunk["text"],
                    Jsonb(chunk["metadata"]),
                ),
            )
        self.connection.commit()

    def get_many(self, ids):
        if not ids:
            return []

        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, document_id, source, text FROM rag_chunks WHERE id = ANY(%s)",
                (list(ids),),
            )
            rows = cursor.fetchall()

        return [
            {"id": row[0], "document_id": row[1], "source": row[2], "text": row[3]}
            for row in rows
        ]

    def counts(self):
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(DISTINCT document_id), COUNT(*) FROM rag_chunks")
            documents, chunks = cursor.fetchone()
        return {"documents": documents, "chunks": chunks}


class _LocalStore:
    name = "Local .db"

    def __init__(self, db_path):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path)

    def setup(self):
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                source TEXT,
                text TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self.connection.commit()

    def save(self, chunk):
        self.connection.execute(
            """
            INSERT INTO rag_chunks (id, document_id, source, text, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document_id = excluded.document_id,
                source = excluded.source,
                text = excluded.text,
                metadata = excluded.metadata
            """,
            (
                chunk["id"],
                chunk["document_id"],
                chunk["source"],
                chunk["text"],
                json.dumps(chunk["metadata"], ensure_ascii=False),
            ),
        )
        self.connection.commit()

    def get_many(self, ids):
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        cursor = self.connection.execute(
            f"SELECT id, document_id, source, text FROM rag_chunks WHERE id IN ({placeholders})",
            list(ids),
        )
        return [
            {"id": row[0], "document_id": row[1], "source": row[2], "text": row[3]}
            for row in cursor.fetchall()
        ]

    def counts(self):
        cursor = self.connection.execute(
            "SELECT COUNT(DISTINCT document_id), COUNT(*) FROM rag_chunks"
        )
        documents, chunks = cursor.fetchone()
        return {"documents": documents, "chunks": chunks}


def _load_chunks():
    chunks = []
    for json_file in sorted(CHUNKS_DIR.glob("*.json")):
        for item_no, item in enumerate(_read_json_items(json_file), start=1):
            text = _text_from_item(item)
            if not text:
                continue

            metadata = item if isinstance(item, dict) else {"value": item}
            source = _source_from_item(item, json_file)
            document_id = _document_id_from_item(item, json_file)
            chunk_id = _chunk_id_from_item(item, json_file, item_no)

            chunks.append(
                {
                    "id": chunk_id,
                    "document_id": document_id,
                    "source": source,
                    "text": text,
                    "metadata": metadata,
                }
            )
    return chunks


def _read_json_items(json_file):
    text = json_file.read_text(encoding="utf-8")
    try:
        return _items_from_json(json.loads(text))
    except json.JSONDecodeError as error:
        raise ValueError(f"{json_file} phải là JSON array/object hợp lệ, không dùng JSONL.") from error


def _items_from_json(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("chunks", "data", "documents", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return [data]


def _text_from_item(item):
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()

    for key in ("text", "content", "chunk", "page_content", "body"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_from_item(item, json_file):
    if isinstance(item, dict):
        for key in ("source", "file", "filename", "path", "document"):
            value = item.get(key)
            if value:
                return str(value)
    return json_file.name


def _document_id_from_item(item, json_file):
    if isinstance(item, dict):
        for key in ("document_id", "doc_id", "source", "file", "filename"):
            value = item.get(key)
            if value:
                return str(value)
    return json_file.stem


def _chunk_id_from_item(item, json_file, item_no):
    if isinstance(item, dict):
        for key in ("id", "chunk_id"):
            value = item.get(key)
            if value:
                return str(value)
    return f"{json_file.stem}-{item_no}"


def _gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _embed(text, client):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )

    if getattr(response, "embeddings", None):
        return list(response.embeddings[0].values)
    if getattr(response, "embedding", None):
        return list(response.embedding.values)
    raise ValueError("Gemini không trả về embedding.")


def _embed_with_quota_wait(text, client, position, total, progress=None):
    while True:
        try:
            return _embed(text, client)
        except Exception as error:
            if not _is_quota_error(error):
                raise
            _log(f"Gặp quota 429 tại chunk {position}/{total}. Đợi 60 giây rồi chạy tiếp...", progress)
            time.sleep(60)


def _is_quota_error(error):
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _log(message, progress=None):
    print(message, flush=True)
    if progress:
        progress(message)


def _text_store():
    postgres = _postgres_store()
    if postgres:
        return postgres
    return _LocalStore(LOCAL_DB_PATH)


def _postgres_store():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "rag_db")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")

    try:
        connection = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=3,
        )
    except Exception:
        return None

    return _PostgresStore(connection)


def _chroma_client():
    if _is_port_open("127.0.0.1", 8000):
        return chromadb.HttpClient(host="localhost", port=8000)
    if _is_port_open("127.0.0.1", 8001):
        return chromadb.HttpClient(host="localhost", port=8001)

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def _chroma_collection():
    return _chroma_client().get_or_create_collection(name=COLLECTION_NAME)


def _chroma_mode():
    if _is_port_open("127.0.0.1", 8000):
        return "Server"
    if _is_port_open("127.0.0.1", 8001):
        return "Server"
    return "Embedded Local"


def _is_port_open(host, port):
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _metadata_value(metadatas, index_no, key):
    if not metadatas or index_no >= len(metadatas):
        return ""
    metadata = metadatas[index_no] or {}
    return metadata.get(key, "")
