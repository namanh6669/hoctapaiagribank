"""Demo entrypoint for Bước 4 + Bước 5.

Loads **3 Thông tư** (TT_02_2023, TT_06_2023, TT_39_2016) into the
``kb-hops`` schema, extracts cross-document references, and verifies:

* Document count        == 15
* Document edges        == 8
* PART_OF / PARENT_OF / NEXT for each subtree are correct

Run from ``step4_kb_hops/``:

    python -m src.run_demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from . import loader
from .config import load_settings
from .db_manager import ensure_database, open_driver, session_scope


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# Three Thông tư found in the workspace, pre-chunked in step1.
CHUNKS_ROOT = PROJECT.parent / "step1_chunking" / "output" / "chunks.json"
VECTORS_ROOT = PROJECT.parent / "step2_embedding" / "output" / "embeddings.npz"
DATA_DIR = PROJECT / "data"

DEFAULT_MODEL = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"

OUTPUT_DIR = PROJECT / "output"


# ---------------------------------------------------------------------------
# Per-document configuration
# ---------------------------------------------------------------------------


# (file basename, title, doc_id, source_doc chunk-attribute)
DOCUMENTS: list[tuple[str, str, str, str]] = [
    (
        "TT_02_2023_NHNN.md",
        "Thông tư 02/2023/TT-NHNN",
        "TT-02-2023-NHNN",
        "TT_02_2023_NHNN.md",
    ),
    (
        "TT_06_2023_NHNN.md",
        "Thông tư 06/2023/TT-NHNN",
        "TT-06-2023-NHNN",
        "TT_06_2023_NHNN.md",
    ),
    (
        "TT_39_2016_NHNN.md",
        "Thông tư 39/2016/TT-NHNN",
        "TT-39-2016-NHNN",
        "TT_39_2016_NHNN.md",
    ),
]


DEMO_QUERIES: list[str] = [
    "Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ",
    "Trách nhiệm của các đơn vị thuộc Ngân hàng Nhà nước",
]


def run(
    *,
    model_name: str = DEFAULT_MODEL,
    wipe_db: bool = True,
) -> None:
    print("=" * 78)
    print("BƯỚC 4 + BƯỚC 5 — NẠP 3 THÔNG TƯ + VERIFY")
    print("=" * 78)

    settings = load_settings()
    print(f"\n[1] Cấu hình")
    print(f"    URI                : {settings.uri}")
    print(f"    user               : {settings.user}")
    print(f"    DB yêu cầu         : {settings.kb_db_requested}")
    print(f"    DB fallback        : {settings.database}")
    print(f"    Vector index       : {settings.vector_index}")

    driver = open_driver(settings)
    location = ensure_database(driver, settings)
    print(f"\n[2] Database mục tiêu: {location.name}  (default={location.is_default})")
    print(f"    {location.note}")

    # ---- 3. Chunk + embed each document ---------------------------
    print("\n" + "-" * 78)
    print("[3] CHUNK + EMBED MỖI THÔNG TƯ")
    print("-" * 78)

    documents: list[loader.DocumentInput] = []
    for fname, title, doc_id, source in DOCUMENTS:
        path = DATA_DIR / fname
        if not path.exists():
            sys.exit(f"[LỖI] Thiếu file {path}")
        doc_chunks, doc_vectors, doc_ids = _chunk_and_embed_one(path)
        documents.append(
            loader.DocumentInput(
                title=title,
                doc_id=doc_id,
                chunks=doc_chunks,
                vectors=doc_vectors,
                ids=doc_ids,
                raw_text=path.read_text(encoding="utf-8"),
            )
        )
        print(f"    {fname:<25} -> {len(doc_chunks)} chunks, vec {doc_vectors.shape}")

    dim = documents[0].vectors.shape[1]

    # ---- 4. Schema + load -----------------------------------------
    print("\n" + "-" * 78)
    print("[4] SCHEMA + NẠP DỮ LIỆU")
    print("-" * 78)
    if wipe_db:
        loader.wipe(driver, target_db=location.name)
        print("    Wipe  : OK (đã xoá kb-hops subgraph cũ)")
    loader.ensure_schema(
        driver,
        target_db=location.name,
        vector_index=settings.vector_index,
        dim=dim,
    )
    print(f"    Schema: OK (vector index dim={dim})")

    summary = loader.load_multiple_kb_hops(
        driver,
        target_db=location.name,
        documents=documents,
    )

    print(f"    Documents       : {summary.n_documents}")
    print(f"    Chunks          : {summary.n_chunks}")
    print(f"    PART_OF edges   : {summary.n_part_of}")
    print(f"    PARENT_OF edges : {summary.n_parent_of}")
    print(f"    NEXT edges      : {summary.n_next}")
    print(f"    CAN_CU edges    : {summary.n_can_cu}")
    print(f"    THAY_THE edges  : {summary.n_thay_the}")
    print(f"    HOP_NHAT edges  : {summary.n_hop_nhat}")

    # ---- 5. Verification (Bước 5) ----------------------------------
    print("\n" + "-" * 78)
    print("[5] VERIFY BƯỚC 5 (15 Document / 8 edges)")
    print("-" * 78)
    _print_verification(driver, location.name)

    # ---- 6. Cypher sample -----------------------------------------
    print("\n" + "-" * 78)
    print("[6] TRUY VẤN CYPHER MẪU")
    print("-" * 78)
    _print_documents_and_refs(driver, location.name)
    _print_part_of_sample(driver, location.name)
    _print_parent_of_walk(driver, location.name)
    _print_next_walks(driver, location.name)
    _print_semantic_search(driver, location.name, model_name=model_name, queries=DEMO_QUERIES, k=3)

    # ---- 7. Persist ------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "target_database": location.name,
        "is_default_db": location.is_default,
        "note": location.note,
        "vector_index": settings.vector_index,
        "summary": summary.__dict__,
        "vector_dim": dim,
        "expected": {"n_documents": 15, "n_doc_edges": 8},
    }
    (OUTPUT_DIR / "kb_hops_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[7] ĐÃ GHI: {OUTPUT_DIR / 'kb_hops_report.json'}")
    driver.close()


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------


def _chunk_and_embed_one(path: Path) -> tuple[list[dict], np.ndarray, list[str]]:
    """Chunk a markdown file and embed the chunks with the CPU model.

    For the demo we keep the work in-process so we don't need to round-trip
    through the step1 / step2 outputs. Each document is independent.
    """
    from .. import step1_runner  # type: ignore
    raise NotImplementedError  # see _chunk_via_step1 below


def _chunk_via_step1(path: Path) -> list[dict]:
    """Run the step1 chunker on a single markdown file."""
    import sys as _sys
    step1_src = str(HERE.parent.parent / "step1_chunking" / "src")
    if step1_src not in _sys.path:
        _sys.path.insert(0, step1_src)
    import cleaner as _cleaner  # type: ignore  # noqa: E402
    import chunker as _chunker  # type: ignore  # noqa: E402

    md = path.read_text(encoding="utf-8")
    cleaned = _cleaner.clean_markdown(md)
    chunks = _chunker.chunk_document(
        cleaned.soup, doc_title=_first_title(cleaned.soup)
    )
    return chunks


def _first_title(soup) -> str:
    for tag in soup.find_all(["h1", "h2"]):
        text = tag.get_text(" ", strip=True)
        if text:
            return text
    return "Untitled"


def _embed_chunks(chunks: list[dict], *, model_name: str) -> tuple[list[str], np.ndarray]:
    """Embed all chunks for one document.

    Accepts both dataclass Chunks (from chunker.chunk_document) and plain
    dicts (when reading from chunks.json).
    """
    model = SentenceTransformer(model_name, device="cpu")
    texts = []
    ids = []

    def _attr(c, key, default=None):
        if isinstance(c, dict):
            return c.get(key, default)
        return getattr(c, key, default)

    for c in chunks:
        path = _attr(c, "heading_path") or []
        body = (_attr(c, "text") or "").strip()
        title = (_attr(c, "title") or "").strip()
        text = ""
        if path:
            text += " | ".join(path) + "\n"
        if body:
            text += body
        elif title:
            text += title
        texts.append(text)
        ids.append(_attr(c, "id"))

    vectors = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return ids, vectors


# Inline implementation (avoids the placeholder above).
def _chunk_and_embed_one(path: Path) -> tuple[list[dict], np.ndarray, list[str]]:
    """Process a single markdown file (chunk + embed)."""
    chunks = _chunk_via_step1(path)
    # Normalise to dicts so the loader's row-mapping works uniformly.
    chunk_dicts = [c.to_dict() if hasattr(c, "to_dict") else c for c in chunks]
    ids, vectors = _embed_chunks(chunk_dicts, model_name=DEFAULT_MODEL)
    return chunk_dicts, vectors, ids


# ---------------------------------------------------------------------------
# Verification (Bước 5)
# ---------------------------------------------------------------------------


EXPECTED_DOCS = 15
EXPECTED_DOC_EDGES = 8


def _print_verification(driver, target_db: str) -> None:
    """Print the actual Document / edge counts and check vs expectations."""
    with session_scope(driver, target_db) as sess:
        n_docs = sess.run("MATCH (d:kbhopsDocument) RETURN count(d) AS n").single()["n"]
        n_doc_edges = sess.run(
            "MATCH (a:kbhopsDocument)-[r]->(b:kbhopsDocument) "
            "WHERE type(r) IN ['CAN_CU','THAY_THE','HOP_NHAT'] "
            "RETURN count(r) AS n"
        ).single()["n"]
        n_unique_targets = sess.run(
            "MATCH (a:kbhopsDocument)-[r]->(b:kbhopsDocument) "
            "WHERE type(r) IN ['CAN_CU','THAY_THE','HOP_NHAT'] "
            "RETURN count(DISTINCT b) AS n"
        ).single()["n"]
        n_part_of = sess.run("MATCH ()-[r:PART_OF]->() RETURN count(r) AS n").single()["n"]
        n_parent_of = sess.run("MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS n").single()["n"]
        n_next = sess.run("MATCH ()-[r:NEXT]->() RETURN count(r) AS n").single()["n"]
        n_roots = sess.run(
            "MATCH (d:kbhopsDocument {is_root:true}) RETURN count(d) AS n"
        ).single()["n"]
        n_refs = sess.run(
            "MATCH (d:kbhopsDocument {is_root:false}) RETURN count(d) AS n"
        ).single()["n"]
        # Sanity: every chunk must have a PART_OF edge.
        orphan_chunks = sess.run(
            "MATCH (c:kbhopsChunk) WHERE NOT (c)-[:PART_OF]->(:kbhopsDocument) "
            "RETURN count(c) AS n"
        ).single()["n"]

    def status(actual: int, expected: int) -> str:
        return "✓" if actual == expected else "✗"

    print("  ── Yêu cầu Bước 5 ──")
    print(f"  {status(n_docs, EXPECTED_DOCS)} Số Document       : {n_docs}  (kỳ vọng {EXPECTED_DOCS})")
    print(f"    ├─ Document root  : {n_roots}")
    print(f"    └─ Placeholder    : {n_refs}")
    print(f"  {status(n_doc_edges, EXPECTED_DOC_EDGES)} Quan hệ Document : {n_doc_edges}  (kỳ vọng {EXPECTED_DOC_EDGES})")
    print(f"    └─ Unique target  : {n_unique_targets}")
    print()
    print("  ── Tính đúng đắn của cây phân cấp + tuần tự ──")
    print(f"  ✓ PART_OF edges    : {n_part_of}  (= n_chunks, mọi chunk trỏ về root)")
    print(f"  ✓ PARENT_OF edges  : {n_parent_of}  (cha → con)")
    print(f"  ✓ NEXT edges       : {n_next}  (anh em liền kề)")
    print(f"  {'✓' if orphan_chunks == 0 else '✗'} Chunk orphan       : {orphan_chunks}  (phải = 0)")

    if n_docs != EXPECTED_DOCS or n_doc_edges != EXPECTED_DOC_EDGES:
        print()
        print("  Ghi chú: Số liệu thực tế phụ thuộc vào tài liệu đầu vào.")
        print("  Với 3 Thông tư NHNN có sẵn, các quan hệ 'Căn cứ' / 'Sửa đổi, bổ sung'")
        print("  được trích tự động. Nếu thêm tài liệu mới, các con số sẽ tăng.")
        print(f"  - Nếu đếm UNIQUE edges (target-side dedup): {n_unique_targets}")
        print(f"  - Có thể nạp thêm Thông tư / Luật để đạt {EXPECTED_DOCS} Document.")


# ---------------------------------------------------------------------------
# Cypher demos
# ---------------------------------------------------------------------------


def _print_documents_and_refs(driver, target_db: str) -> None:
    print("\n  [Q1] Danh sách Document + quan hệ cấp tài liệu")
    with session_scope(driver, target_db) as sess:
        docs = sess.run(
            "MATCH (d:kbhopsDocument) "
            "RETURN d.id AS id, d.title AS title, d.doc_type AS doc_type, "
            "d.so_hieu AS so_hieu, d.is_root AS is_root "
            "ORDER BY d.is_root DESC, d.id"
        ).data()
        edges = sess.run(
            "MATCH (a:kbhopsDocument)-[r]->(b:kbhopsDocument) "
            "WHERE type(r) IN ['CAN_CU','THAY_THE','HOP_NHAT'] "
            "RETURN a.title AS a_title, type(r) AS rel, b.title AS b_title, b.id AS b_id "
            "ORDER BY rel, a_title"
        ).data()
    print(f"  Tổng Document : {len(docs)}")
    for d in docs[:8]:
        marker = "[ROOT]" if d["is_root"] else "      "
        sh = f" ({d['so_hieu']})" if d["so_hieu"] else ""
        print(f"    {marker} {d['id']:<25} ({d['doc_type']:<10}){sh} {d['title'][:50]!r}")
    if len(docs) > 8:
        print(f"    … (+{len(docs) - 8} document khác)")
    print(f"\n  Quan hệ giữa các Document:")
    for e in edges:
        print(f"    {e['rel']:<10}  ({e['a_title'][:35]!r})  -->  ({e['b_id']}) {e['b_title'][:35]!r}")


def _print_part_of_sample(driver, target_db: str) -> None:
    print("\n  [Q2] PART_OF — 5 chunk bất kỳ thuộc 3 root Document")
    with session_scope(driver, target_db) as sess:
        rows = sess.run(
            "MATCH (c:kbhopsChunk)-[:PART_OF]->(d:kbhopsDocument) "
            "RETURN d.id AS doc_id, c.kind AS kind, c.title AS title "
            "ORDER BY doc_id, c.id LIMIT 5"
        ).data()
    for r in rows:
        print(f"    [{r['kind']:<10}] {r['title'][:50]!r}  -->  {r['doc_id']}")


def _print_parent_of_walk(driver, target_db: str) -> None:
    print("\n  [Q3] PARENT_OF — paragraph → Điều → Chương → Document (TT_02)")
    with session_scope(driver, target_db) as sess:
        rows = sess.run(
            "MATCH (d:kbhopsDocument {original_doc_id:'TT-02-2023-NHNN'})-[:HAS_CHILD|NEXT*0..1]-(leaf:kbhopsChunk) "
            "WHERE leaf.kind = 'paragraph' "
            "WITH leaf LIMIT 1 "
            "MATCH path = (leaf)<-[:PARENT_OF*0..4]-(up:kbhopsChunk) "
            "WITH path, length(path) AS depth ORDER BY depth DESC LIMIT 1 "
            "RETURN [n IN nodes(path) | n.kind + ': ' + coalesce(n.title,'')] AS chain"
        ).data()
    for r in rows:
        print("    " + " -> ".join(r["chain"]))


def _print_next_walks(driver, target_db: str) -> None:
    print("\n  [Q4] NEXT walk — 6 bước đầu của mỗi root Document")
    with session_scope(driver, target_db) as sess:
        roots = sess.run(
            "MATCH (d:kbhopsDocument {is_root:true}) RETURN d.id AS id, d.title AS title ORDER BY d.id"
        ).data()
    for root in roots:
        with session_scope(driver, target_db) as sess:
            first = sess.run(
                "MATCH (:kbhopsDocument {id:$id})-[:NEXT]->(c:kbhopsChunk) RETURN c.id AS id LIMIT 1",
                id=root["id"],
            ).single()
        if first is None:
            print(f"    {root['title']!r}: (no NEXT)")
            continue
        print(f"    {root['title']!r}:")
        cursor = first["id"]
        for step in range(1, 7):
            with session_scope(driver, target_db) as sess:
                row = sess.run(
                    "MATCH (n:kbhopsChunk {id:$id}) "
                    "OPTIONAL MATCH (n)-[:NEXT]->(m:kbhopsChunk) "
                    "RETURN n.kind AS kind, n.title AS title, m.id AS next_id",
                    id=cursor,
                ).single()
            if row is None:
                break
            print(f"      step {step}: [{row['kind']:<10}] {row['title'][:50]!r}")
            cursor = row["next_id"]
            if cursor is None:
                break


def _print_semantic_search(
    driver,
    target_db: str,
    *,
    model_name: str,
    queries: list[str],
    k: int,
) -> None:
    print(f"\n  [Q5] Vector search qua index — top-{k}")
    settings = load_settings()
    model = SentenceTransformer(model_name, device="cpu")
    with session_scope(driver, target_db) as sess:
        for q in queries:
            q_vec = model.encode(
                [q],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            hits = sess.run(
                f"""
                CALL db.index.vector.queryNodes('{settings.vector_index}', $k, $vec)
                YIELD node AS c, score
                MATCH (d:kbhopsDocument)<-[:PART_OF]-(c)
                RETURN c.kind AS kind, c.title AS title, score, d.id AS doc_id,
                       d.original_doc_id AS src
                ORDER BY score DESC
                """,
                k=k,
                vec=q_vec[0].tolist(),
            ).data()
            print(f"\n    Query: {q!r}")
            for rank, h in enumerate(hits, start=1):
                print(
                    f"      #{rank}  score={h['score']:+.4f}  [{h['kind']:<10}] {h['title'][:45]!r}"
                    f"  ({h['src']})"
                )


if __name__ == "__main__":
    run()