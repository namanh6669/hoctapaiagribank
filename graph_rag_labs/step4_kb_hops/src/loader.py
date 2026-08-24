"""Load kb-hops knowledge graph into Neo4j.

Schema (matches the Bước 4 spec):

* ``(:kbhops:Document {id, title, source_doc, doc_type, so_hieu, ngay_ban_hanh})``
  — the root document we ingested, plus *placeholder* documents for every
  external reference we extract from the text.
* ``(:kbhops:Chunk {id, kind, title, text, heading_path, depth, char_count,
  source_doc, embedding})``
  — text segments; the dense vector lives on the node directly (small enough,
  no need for a separate ``Embedding`` node).

Edges (all carry a ``kbhops`` label so we can isolate the subgraph):

* ``(:Chunk)-[:kbhops:PART_OF]->(:Document)``
* ``(:Chunk)-[:kbhops:PARENT_OF]->(:Chunk)``        (down the tree)
* ``(:Chunk)-[:kbhops:NEXT]->(:Chunk)``             (reading flow)
* ``(:Document)-[:kbhops:CAN_CU]->(:Document)``
* ``(:Document)-[:kbhops:THAY_THE]->(:Document)``
* ``(:Document)-[:kbhops:HOP_NHAT]->(:Document)``

The ``:kbhops`` extra label is what we add when we cannot create a real
``kb-hops`` database (Community Edition). Enterprise users get the same data
without that label — ``get_kb_label()`` returns the empty string in that
case.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from neo4j import Driver

from .db_manager import session_scope
from .relationship_extractor import (
    DocumentRef,
    DocumentRelationships,
    extract_relationships,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GraphSummary:
    n_documents: int
    n_chunks: int
    n_part_of: int
    n_parent_of: int
    n_next: int
    n_can_cu: int
    n_thay_the: int
    n_hop_nhat: int


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


CONSTRAINTS_TEMPLATE = """
CREATE CONSTRAINT kbhops_document_id IF NOT EXISTS
FOR (n:kbhopsDocument) REQUIRE n.id IS UNIQUE
"""

# Note: the constraint is templated because we don't know whether we're
# using the ``:kbhops`` extra label or not. We replace ``kbhopsDocument``
# with ``Document`` (without the extra label) on Enterprise.

INDEXES: list[str] = []


def _constraints(target_label: str) -> list[str]:
    return [
        f"CREATE CONSTRAINT {target_label.lower()}_document_id IF NOT EXISTS "
        f"FOR (n:{target_label}) REQUIRE n.id IS UNIQUE",
        f"CREATE CONSTRAINT {target_label.lower()}_chunk_id IF NOT EXISTS "
        f"FOR (n:{target_label}Chunk) REQUIRE n.id IS UNIQUE",
    ]


def ensure_schema(driver: Driver, *, target_db: str, vector_index: str, dim: int) -> None:
    """Create constraints, the BTREE indexes and the vector index."""
    label = "kbhops"
    chunk_label = "kbhopsChunk"
    with session_scope(driver, target_db) as sess:
        for stmt in _constraints(label):
            sess.run(stmt)
        sess.run(
            f"CREATE INDEX {label}_chunk_kind IF NOT EXISTS "
            f"FOR (n:{chunk_label}) ON (n.kind)"
        )
        # Vector index on the dense embedding property.
        sess.run(
            f"""
            CREATE VECTOR INDEX {vector_index} IF NOT EXISTS
            FOR (n:{chunk_label})
            ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )


def wipe(driver: Driver, *, target_db: str) -> None:
    """Delete every kb-hops node & edge in the target database."""
    with session_scope(driver, target_db) as sess:
        # Wipe nodes that have the kbhops extra label, plus their edges.
        sess.run("MATCH (n:kbhops) DETACH DELETE n")
        sess.run("MATCH (n:kbhopsChunk) DETACH DELETE n")
        sess.run("MATCH (n:kbhopsDocument) DETACH DELETE n")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_kb_hops(
    driver: Driver,
    *,
    target_db: str,
    chunks: list[dict],
    vectors: np.ndarray,
    ids: list[str],
    source_doc_title: str,
    source_doc_id: str,
    vector_index: str,
    batch_size: int = 200,
) -> GraphSummary:
    """Upsert kb-hops data into the target database."""
    # Find the actual document chunk in the input chunks (kind="document").
    # Its id is what chunks use as parent_id, so we MUST use it as the
    # graph Document's id to keep PART_OF consistent.
    doc_chunk = next((c for c in chunks if c["kind"] == "document"), None)
    if doc_chunk is None:
        raise RuntimeError("chunks list does not contain a 'document' node")
    graph_doc_id = doc_chunk["id"]

    # ---- 1. Extract document-level references -----------------------
    root_text = "\n".join(c["text"] for c in chunks if c["kind"] == "paragraph")
    rels = extract_relationships(document_text=root_text, document_heading=source_doc_title)

    # ---- 2. Upsert documents ----------------------------------------
    with session_scope(driver, target_db) as sess:
        sess.run(
            """
            MERGE (d:kbhopsDocument {id: $id})
            ON CREATE SET
                d.title = $title,
                d.doc_type = $doc_type,
                d.so_hieu = $so_hieu,
                d.ngay_ban_hanh = $ngay_ban_hanh,
                d.is_root = $is_root,
                d.source_doc = $source_doc,
                d.original_doc_id = $original_doc_id
            """,
            id=graph_doc_id,
            title=source_doc_title,
            doc_type="thong_tu",
            so_hieu="02/2023/TT-NHNN",
            ngay_ban_hanh="2023-04-23",
            is_root=True,
            source_doc=source_doc_id,
            original_doc_id=source_doc_id,
        )

        for ref in rels.can_cu:
            _upsert_document(sess, ref, kind="CAN_CU")
        for ref in rels.thay_the:
            _upsert_document(sess, ref, kind="THAY_THE")
        for ref in rels.hop_nhat:
            _upsert_document(sess, ref, kind="HOP_NHAT")

        # ---- 3. Edges between documents --------------------------------
        for ref in rels.can_cu:
            _create_doc_edge(sess, graph_doc_id, ref.canonical_id, "CAN_CU")
        for ref in rels.thay_the:
            _create_doc_edge(sess, graph_doc_id, ref.canonical_id, "THAY_THE")
        for ref in rels.hop_nhat:
            _create_doc_edge(sess, graph_doc_id, ref.canonical_id, "HOP_NHAT")

    # ---- 4. Chunks + embedding --------------------------------------
    vec_by_id: dict[str, np.ndarray] = {cid: vectors[i] for i, cid in enumerate(ids)}

    with session_scope(driver, target_db) as sess:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            sess.run(
                _CHUNK_CYPHER,
                rows=[_chunk_row(c, vec_by_id) for c in batch],
                doc_id=graph_doc_id,
            )

        # ---- 5. PARENT_OF edges (chunk hierarchy) ---------------------
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            sess.run(
                _PARENT_OF_CYPHER,
                rows=[
                    {"pid": c["parent_id"], "cid": c["id"]}
                    for c in batch
                    if c.get("parent_id")
                ],
            )

        # ---- 6. NEXT edges -------------------------------------------
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            sess.run(
                _NEXT_CYPHER,
                rows=[{"a": c["id"], "b": c["next_id"]} for c in batch if c.get("next_id")],
            )

    # ---- 7. NEXT edges from the root Document to its first child ----
    with session_scope(driver, target_db) as sess:
        first_child = next(
            (c["id"] for c in chunks if c.get("parent_id") == graph_doc_id), None
        )
        if first_child is not None:
            sess.run(
                """
                MATCH (d:kbhopsDocument {id: $doc_id})
                MATCH (c:kbhopsChunk {id: $child_id})
                MERGE (d)-[:NEXT]->(c)
                """,
                doc_id=graph_doc_id,
                child_id=first_child,
            )

    return _summarise(driver, target_db=target_db)


# ---------------------------------------------------------------------------
# Multi-document loader
# ---------------------------------------------------------------------------


@dataclass
class DocumentInput:
    """One document to ingest: its chunks, vectors, and metadata."""

    title: str
    doc_id: str         # User-supplied id (e.g. "TT-02-2023-NHNN"). The
                        # graph document id will be the *Document chunk's*
                        # id from step1 (e.g. "c0001-535e30") so PART_OF
                        # works without remapping parent_id.
    chunks: list[dict]
    vectors: np.ndarray
    ids: list[str]
    raw_text: str = ""  # Full markdown source. Used by the relationship
                          # extractor so it can see the title / subtitle
                          # text that doesn't end up in any chunk.


def load_multiple_kb_hops(
    driver: Driver,
    *,
    target_db: str,
    documents: list[DocumentInput],
    batch_size: int = 200,
) -> GraphSummary:
    """Ingest multiple documents in one pass.

    Each document contributes:

    * its own root ``(:kbhopsDocument)`` node,
    * a ``kbhopsChunk`` subtree under it,
    * one PART_OF edge per chunk back to its document root,
    * document-level edges (CAN_CU / THAY_THE / HOP_NHAT).

    Cross-references that happen to point at another *root* (e.g. TT_06
    says "Sửa đổi, bổ sung một số điều của Thông tư số 39/2016/TT-NHNN")
    are linked to the actual root node, not a duplicate placeholder.
    """
    if not documents:
        raise ValueError("documents list is empty")

    # Pre-compute every root's id + (số hiệu, original_doc_id) so we can
    # resolve "39/2016/TT-NHNN" → the actual root chunk id.
    root_index: dict[str, str] = {}  # canonical / original -> graph id
    root_so_hieus: dict[str, str] = {}  # graph id -> so_hieu guess
    for doc in documents:
        graph_doc_id = _graph_doc_id(doc.chunks)
        root_index[doc.doc_id] = graph_doc_id
        # Try to derive a số hiệu from the doc_id (e.g. "TT-39-2016-NHNN" → "39/2016/TT-NHNN").
        root_so_hieus[graph_doc_id] = _derive_so_hieu(doc.doc_id)

    # ---- 1. Upsert every root + reference Document --------------
    seen_doc_refs: dict[tuple[str, str], DocumentRef] = {}

    # Extract relationships first (no DB writes).
    doc_rels: dict[str, "DocumentRelationships"] = {}
    for doc in documents:
        rels_text = doc.raw_text or "\n".join(c["text"] for c in doc.chunks if c["text"])
        doc_rels[doc.doc_id] = extract_relationships(
            document_text=rels_text,
            document_heading=doc.title,
            scan_body=True,
            preamble_chars=5000,
        )
        for ref in doc_rels[doc.doc_id].can_cu:
            seen_doc_refs[("CAN_CU", ref.canonical_id)] = ref
        for ref in doc_rels[doc.doc_id].thay_the:
            seen_doc_refs[("THAY_THE", ref.canonical_id)] = ref
        for ref in doc_rels[doc.doc_id].hop_nhat:
            seen_doc_refs[("HOP_NHAT", ref.canonical_id)] = ref

    with session_scope(driver, target_db) as sess:
        # First pass: create every root Document so cross-refs can resolve.
        for doc in documents:
            graph_doc_id = _graph_doc_id(doc.chunks)
            so_hieu = root_so_hieus.get(graph_doc_id)
            _upsert_root_document(
                sess,
                graph_id=graph_doc_id,
                title=doc.title,
                source_doc_id=doc.doc_id,
                so_hieu=so_hieu,
            )

        # Second pass: create reference Documents for non-root targets.
        for (kind, cid), ref in seen_doc_refs.items():
            if _resolve_target(root_index, root_so_hieus, cid) is None:
                _upsert_reference_document(sess, ref, kind=kind)

        # Third pass: create the doc-doc edges. Now ALL target nodes exist.
        for doc in documents:
            graph_doc_id = _graph_doc_id(doc.chunks)
            rels = doc_rels[doc.doc_id]
            for ref in rels.can_cu:
                target_id = _resolve_target(root_index, root_so_hieus, ref.canonical_id) or ref.canonical_id
                _create_doc_edge(sess, graph_doc_id, target_id, "CAN_CU")
            for ref in rels.thay_the:
                target_id = _resolve_target(root_index, root_so_hieus, ref.canonical_id) or ref.canonical_id
                _create_doc_edge(sess, graph_doc_id, target_id, "THAY_THE")
            for ref in rels.hop_nhat:
                target_id = _resolve_target(root_index, root_so_hieus, ref.canonical_id) or ref.canonical_id
                _create_doc_edge(sess, graph_doc_id, target_id, "HOP_NHAT")

    # ---- 2. Chunks + edges per document --------------------------
    for doc in documents:
        graph_doc_id = _graph_doc_id(doc.chunks)
        vec_by_id: dict[str, np.ndarray] = {
            cid: doc.vectors[i] for i, cid in enumerate(doc.ids)
        }

        with session_scope(driver, target_db) as sess:
            for start in range(0, len(doc.chunks), batch_size):
                batch = doc.chunks[start : start + batch_size]
                sess.run(
                    _CHUNK_CYPHER,
                    rows=[_chunk_row(c, vec_by_id) for c in batch],
                    doc_id=graph_doc_id,
                )
            for start in range(0, len(doc.chunks), batch_size):
                batch = doc.chunks[start : start + batch_size]
                sess.run(
                    _PARENT_OF_CYPHER,
                    rows=[
                        {"pid": c["parent_id"], "cid": c["id"]}
                        for c in batch
                        if c.get("parent_id")
                    ],
                )
            for start in range(0, len(doc.chunks), batch_size):
                batch = doc.chunks[start : start + batch_size]
                sess.run(
                    _NEXT_CYPHER,
                    rows=[{"a": c["id"], "b": c["next_id"]} for c in batch if c.get("next_id")],
                )
            first_child = next(
                (c["id"] for c in doc.chunks if c.get("parent_id") == graph_doc_id), None
            )
            if first_child is not None:
                sess.run(
                    """
                    MATCH (d:kbhopsDocument {id: $doc_id})
                    MATCH (c:kbhopsChunk {id: $child_id})
                    MERGE (d)-[:NEXT]->(c)
                    """,
                    doc_id=graph_doc_id,
                    child_id=first_child,
                )

    return _summarise(driver, target_db=target_db)


def _resolve_target(
    root_index: dict[str, str],
    root_so_hieus: dict[str, str],
    canonical_id: str,
) -> str | None:
    """If ``canonical_id`` matches a known root (by id or by số hiệu),
    return that root's graph id; otherwise return None so the loader
    creates a placeholder Document.
    """
    if canonical_id in root_index.values():
        return canonical_id
    for graph_id, so_hieu in root_so_hieus.items():
        if so_hieu and so_hieu.upper() == canonical_id.upper():
            return graph_id
    return None


def _derive_so_hieu(doc_id: str) -> str | None:
    """Extract the số hiệu from a canonical doc id.

    E.g. ``TT-39-2016-NHNN`` → ``39/2016/TT-NHNN``,
         ``TT-02-2023-NHNN`` → ``02/2023/TT-NHNN``.
    Returns None if the pattern doesn't match.
    """
    parts = doc_id.split("-")
    if len(parts) >= 4 and parts[0] in {"TT", "ND", "NQ", "QD"}:
        # TT-39-2016-NHNN → ["TT", "39", "2016", "NHNN"]
        kind = parts[0]
        number = parts[1]
        year = parts[2]
        agency = "-".join(parts[3:])
        suffix = {
            "TT": "TT",
            "ND": "NĐ-CP",
            "NQ": "NQ-CP",
            "QD": "QĐ",
        }.get(kind, kind)
        return f"{number}/{year}/{suffix}-{agency}"
    return None


def _graph_doc_id(chunks: list[dict]) -> str:
    doc_chunk = next((c for c in chunks if c["kind"] == "document"), None)
    if doc_chunk is None:
        raise RuntimeError("chunks list does not contain a 'document' node")
    return doc_chunk["id"]


def _upsert_root_document(
    sess, *, graph_id: str, title: str, source_doc_id: str, so_hieu: str | None = None
) -> None:
    sess.run(
        """
        MERGE (d:kbhopsDocument {id: $id})
        ON CREATE SET
            d.title = $title,
            d.doc_type = 'thong_tu',
            d.is_root = true,
            d.source_doc = $source_doc,
            d.original_doc_id = $original_doc_id,
            d.so_hieu = $so_hieu
        """,
        id=graph_id,
        title=title,
        source_doc=source_doc_id,
        original_doc_id=source_doc_id,
        so_hieu=so_hieu,
    )


def _upsert_reference_document(sess, ref: DocumentRef, *, kind: str) -> None:
    sess.run(
        """
        MERGE (d:kbhopsDocument {id: $id})
        ON CREATE SET
            d.title = $title,
            d.doc_type = $doc_type,
            d.so_hieu = $so_hieu,
            d.ngay_ban_hanh = $ngay_ban_hanh,
            d.is_root = false,
            d.discovered_via = $kind
        """,
        id=ref.canonical_id,
        title=ref.title,
        doc_type=ref.doc_type,
        so_hieu=ref.so_hieu,
        ngay_ban_hanh=ref.ngay_ban_hanh,
        kind=kind,
    )


# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------


_CHUNK_CYPHER = """
UNWIND $rows AS row
MERGE (c:kbhopsChunk {id: row.id})
ON CREATE SET
    c.kind = row.kind,
    c.title = row.title,
    c.text = row.text,
    c.heading_path = row.heading_path,
    c.depth = row.depth,
    c.char_count = row.char_count,
    c.source_doc = row.source_doc,
    c.embedding = row.embedding
WITH c, row
MATCH (d:kbhopsDocument {id: $doc_id})
MERGE (c)-[:PART_OF]->(d)
"""


_PARENT_OF_CYPHER = """
UNWIND $rows AS row
MATCH (p:kbhopsChunk {id: row.pid})
MATCH (c:kbhopsChunk {id: row.cid})
MERGE (p)-[:PARENT_OF]->(c)
"""


_NEXT_CYPHER = """
UNWIND $rows AS row
MATCH (a:kbhopsChunk {id: row.a})
MATCH (b:kbhopsChunk {id: row.b})
MERGE (a)-[:NEXT]->(b)
"""


_DOC_EDGE_CYPHER_TMPL = """
MATCH (a:kbhopsDocument {id: $from_id})
MATCH (b:kbhopsDocument {id: $to_id})
MERGE (a)-[r:`__KIND__`]->(b)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upsert_document(sess, ref: DocumentRef, *, kind: str) -> None:
    sess.run(
        """
        MERGE (d:kbhopsDocument {id: $id})
        ON CREATE SET
            d.title = $title,
            d.doc_type = $doc_type,
            d.so_hieu = $so_hieu,
            d.ngay_ban_hanh = $ngay_ban_hanh,
            d.is_root = false,
            d.discovered_via = $kind
        """,
        id=ref.canonical_id,
        title=ref.title,
        doc_type=ref.doc_type,
        so_hieu=ref.so_hieu,
        ngay_ban_hanh=ref.ngay_ban_hanh,
        kind=kind,
    )


def _create_doc_edge(sess, from_id: str, to_id: str, kind: str) -> None:
    stmt = _DOC_EDGE_CYPHER_TMPL.replace("__KIND__", kind)
    sess.run(stmt, from_id=from_id, to_id=to_id)


def _chunk_row(c: dict, vec_by_id: dict[str, np.ndarray]) -> dict:
    cid = c["id"]
    vec = vec_by_id.get(cid)
    return {
        "id": cid,
        "kind": c["kind"],
        "title": c.get("title", "") or "",
        "text": c.get("text", "") or "",
        "heading_path": c.get("heading_path") or [],
        "depth": int(c.get("depth", 0)),
        "char_count": int(c.get("char_count", 0)),
        "source_doc": c.get("source_doc"),
        "embedding": vec.tolist() if vec is not None else None,
    }


def _summarise(driver: Driver, *, target_db: str) -> GraphSummary:
    """Count every node / edge separately. Splitting queries avoids the
    "0 rows" problem when a relationship type has no edges."""
    with session_scope(driver, target_db) as sess:
        n_docs = sess.run("MATCH (d:kbhopsDocument) RETURN count(d) AS n").single()["n"]
        n_chunks = sess.run("MATCH (c:kbhopsChunk) RETURN count(c) AS n").single()["n"]
        n_part_of = sess.run("MATCH ()-[r:PART_OF]->() RETURN count(r) AS n").single()["n"]
        n_parent_of = sess.run("MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS n").single()["n"]
        n_next = sess.run("MATCH ()-[r:NEXT]->() RETURN count(r) AS n").single()["n"]
        n_can_cu = _count_or_zero(sess, "MATCH ()-[r:CAN_CU]->() RETURN count(r) AS n")
        n_thay_the = _count_or_zero(sess, "MATCH ()-[r:THAY_THE]->() RETURN count(r) AS n")
        n_hop_nhat = _count_or_zero(sess, "MATCH ()-[r:HOP_NHAT]->() RETURN count(r) AS n")

    return GraphSummary(
        n_documents=int(n_docs),
        n_chunks=int(n_chunks),
        n_part_of=int(n_part_of),
        n_parent_of=int(n_parent_of),
        n_next=int(n_next),
        n_can_cu=int(n_can_cu),
        n_thay_the=int(n_thay_the),
        n_hop_nhat=int(n_hop_nhat),
    )


def _count_or_zero(sess, cypher: str) -> int:
    """Run a count query; return 0 when the relationship type doesn't exist."""
    try:
        rec = sess.run(cypher).single()
        return int(rec["n"]) if rec else 0
    except Exception:  # noqa: BLE001
        # Neo4j emits a warning when the type doesn't exist, but still returns 0 rows.
        return 0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return list(data["ids"]), data["vectors"]