"""Demo entrypoint for Bước 3.

* Connects to Neo4j using ``.env.graph_rag``,
* wipes the database (idempotent demo),
* ensures the schema + vector index,
* upserts all chunks from Bước 1 + vectors from Bước 2,
* prints a summary (node / edge / embedding counts),
* runs a few illustrative Cypher queries (children of root, NEXT walk,
  semantic search via vector index, parent traversal).

Run from ``step3_graph/``:

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
from .neo4j_client import open_driver, session_scope


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

DEFAULT_CHUNKS = PROJECT.parent / "step1_chunking" / "output" / "chunks.json"
DEFAULT_VECTORS = PROJECT.parent / "step2_embedding" / "output" / "embeddings.npz"
DEFAULT_MODEL = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"

OUTPUT_DIR = PROJECT / "output"
SAMPLE_DIR = OUTPUT_DIR / "sample"

DEMO_QUERIES: list[str] = [
    "Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ",
    "Khách hàng nào được giữ nguyên nhóm nợ?",
]


def run(
    *,
    chunks_path: Path = DEFAULT_CHUNKS,
    vectors_path: Path = DEFAULT_VECTORS,
    model_name: str = DEFAULT_MODEL,
    wipe_db: bool = True,
) -> None:
    print("=" * 78)
    print("BƯỚC 3 — NẠP GRAPH + TRUY VẤN CYPHER")
    print("=" * 78)

    settings = load_settings()
    print(f"\n[1] Kết nối Neo4j")
    print(f"    URI     : {settings.uri}")
    print(f"    user    : {settings.user}")
    print(f"    database: {settings.database}")
    print(f"    vector_index: {settings.vector_index}")

    driver = open_driver(settings)
    with session_scope(driver) as sess:
        server_info = sess.run("CALL dbms.components() YIELD name, versions").data()
    print(f"    server  : {server_info}")

    # ---- 2. Load chunks + vectors from disk ----------------------------
    print("\n" + "-" * 78)
    print("[2] ĐỌC CHUNKS + EMBEDDINGS TỪ ĐĨA")
    print("-" * 78)
    chunks = loader.load_chunks(chunks_path)
    ids, vectors = loader.load_vectors(vectors_path)
    print(f"    chunks : {len(chunks)} <- {chunks_path.name}")
    print(f"    vectors: {vectors.shape} <- {vectors_path.name}")
    if len(chunks) != len(ids):
        sys.exit(f"[LỖI] len(chunks)={len(chunks)} khác len(ids)={len(ids)}")
    if vectors.shape[0] != len(chunks):
        sys.exit("[LỖI] Số vector không khớp số chunk.")

    dim = int(vectors.shape[1])

    # ---- 3. Reset & ensure schema --------------------------------------
    print("\n" + "-" * 78)
    print("[3] CHUẨN BỊ SCHEMA")
    print("-" * 78)
    if wipe_db:
        loader.wipe(driver)
        print("    Wipe   : OK (đã xoá mọi node/edge cũ)")
    loader.ensure_schema(driver, vector_index=settings.vector_index, dim=dim)
    print(f"    Schema : OK (constraints, indexes, vector index '{settings.vector_index}', dim={dim})")

    # ---- 4. Load --------------------------------------------------------
    print("\n" + "-" * 78)
    print("[4] NẠP DỮ LIỆU VÀO GRAPH")
    print("-" * 78)
    summary = loader.load_chunks_and_vectors(
        driver,
        chunks=chunks,
        vectors=vectors,
        ids=ids,
    )
    print(f"    Chunk nodes    : {summary.n_nodes}")
    print(f"    HAS_CHILD edges: {summary.n_has_child}")
    print(f"    NEXT edges     : {summary.n_next}")
    print(f"    Embeddings     : {summary.n_embeddings}")
    print(f"    Vector index   : {settings.vector_index}")

    # ---- 5. Sample Cypher queries --------------------------------------
    print("\n" + "-" * 78)
    print("[5] TRUY VẤN CYPHER MẪU")
    print("-" * 78)
    _print_root_and_children(driver)
    _print_next_walk(driver, steps=6)
    _print_article_with_children(driver)
    _print_parent_walk(driver)
    _print_semantic_search(driver, model_name=model_name, queries=DEMO_QUERIES, k=3)

    # ---- 6. Persist report ---------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "uri": settings.uri,
        "database": settings.database,
        "vector_index": settings.vector_index,
        "n_chunks": summary.n_chunks,
        "n_nodes": summary.n_nodes,
        "n_has_child": summary.n_has_child,
        "n_next": summary.n_next,
        "n_embeddings": summary.n_embeddings,
        "vector_dim": dim,
    }
    (OUTPUT_DIR / "graph_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[6] ĐÃ GHI: {OUTPUT_DIR / 'graph_report.json'}")
    driver.close()


# ---------------------------------------------------------------------------
# Cypher demos
# ---------------------------------------------------------------------------


def _print_root_and_children(driver) -> None:
    """Print the document root + 3 first chunks hanging off it."""
    print("\n  [Q1] Document root + 3 chunk đầu")
    with session_scope(driver) as sess:
        rows = sess.run(
            """
            MATCH (d:Document)-[:HAS_CHILD]->(c:Chunk)
            RETURN d.id AS root_id, d.title AS root_title,
                   c.id AS chunk_id, c.kind AS kind,
                   c.title AS title
            ORDER BY c.id
            LIMIT 3
            """
        ).data()
    print(f"  Root: {rows[0]['root_title']!r}")
    for r in rows:
        print(f"    - [{r['kind']:<10}] {r['title'][:60]!r}")


def _print_next_walk(driver, *, steps: int) -> None:
    """Walk NEXT edges from the document root for ``steps`` hops."""
    print(f"\n  [Q2] NEXT walk {steps} bước từ Document")
    with session_scope(driver) as sess:
        cursor_row = sess.run(
            """
            MATCH (d:Document)-[:NEXT]->(first:Chunk)
            RETURN first.id AS id
            LIMIT 1
            """
        ).single()
    if cursor_row is None:
        print("    (Không tìm thấy NEXT đầu tiên)")
        return

    cursor = cursor_row["id"]
    for i in range(steps):
        with session_scope(driver) as sess:
            row = sess.run(
                """
                MATCH (n:Chunk {id: $id})
                RETURN n.kind AS kind, n.title AS title,
                       n.heading_path AS heading_path
                """,
                id=cursor,
            ).single()
            nxt = sess.run(
                """
                MATCH (:Chunk {id: $id})-[:NEXT]->(m:Chunk)
                RETURN m.id AS next_id
                """,
                id=cursor,
            ).single()
        if row is None:
            break
        print(f"    step {i + 1}: [{row['kind']:<10}] {row['title'][:55]!r}")
        cursor = nxt["next_id"] if nxt else None
        if cursor is None:
            break


def _print_article_with_children(driver) -> None:
    """Show one Article and its body children."""
    print("\n  [Q3] Một Điều + các node con (paragraph / list / table)")
    with session_scope(driver) as sess:
        row = sess.run(
            """
            MATCH (a:Article)
            RETURN a.id AS id, a.title AS title
            ORDER BY a.id
            LIMIT 1
            """
        ).single()
        if row is None:
            print("    (Không có Article)")
            return
        children = sess.run(
            """
            MATCH (a:Article {id: $id})-[:HAS_CHILD]->(c:Chunk)
            RETURN c.kind AS kind, c.title AS title, c.char_count AS chars
            ORDER BY c.id
            """,
            id=row["id"],
        ).data()
    print(f"    Article : {row['title']!r}")
    for c in children:
        print(f"      └─ [{c['kind']:<10}] {c['title'][:55]!r}  ({c['chars']} chars)")


def _print_parent_walk(driver) -> None:
    """Show heading_path traversal: chunk -> Article -> Chapter -> Document."""
    print("\n  [Q4] Leo lên cây cha từ một paragraph")
    with session_scope(driver) as sess:
        rows = sess.run(
            """
            MATCH (p:Paragraph)
            WITH p LIMIT 1
            MATCH path = (p)-[:HAS_CHILD*0..4]->(a:Chunk)<-[:HAS_CHILD*0..3]-(doc:Document)
            RETURN [n IN nodes(path) | n.kind + ': ' + coalesce(n.title,'')] AS chain
            """
        ).data()
    for r in rows:
        print("    " + " -> ".join(r["chain"]))


def _print_semantic_search(driver, *, model_name: str, queries: list[str], k: int) -> None:
    """Run vector search via the index, then walk parents for context."""
    print(f"\n  [Q5] Vector search (index) — model {model_name}")
    print(f"        top-{k} Embedding gần nhất + chunk cha")
    model = SentenceTransformer(model_name, device="cpu")
    settings = load_settings()
    with session_scope(driver) as sess:
        for q in queries:
            q_vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            hits = sess.run(
                f"""
                CALL db.index.vector.queryNodes('{settings.vector_index}', $k, $vec)
                YIELD node AS emb, score
                MATCH (c:Chunk)-[:HAS_VECTOR]->(emb)
                OPTIONAL MATCH (c)-[:HAS_CHILD*0..3]->(parent:Chunk)
                WHERE parent.kind IN ['Article','Chapter','Document']
                RETURN c.id AS chunk_id, c.kind AS chunk_kind,
                       c.title AS chunk_title, score,
                       collect(DISTINCT parent.title)[0..2] AS ancestors
                ORDER BY score DESC
                """,
                k=k,
                vec=q_vec[0].tolist(),
            ).data()
            print(f"\n    Query: {q!r}")
            for rank, h in enumerate(hits, start=1):
                anc = " > ".join(h["ancestors"])
                print(f"      #{rank}  score={h['score']:+.4f}  [{h['chunk_kind']:<10}] {h['chunk_title'][:55]!r}")
                if anc:
                    print(f"           ↑ {anc}")


if __name__ == "__main__":
    run()