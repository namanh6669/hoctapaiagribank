"""Build the Graph-RAG knowledge graph in Neo4j from chunks + embeddings.

Schema
------

All nodes carry the abstract ``Chunk`` label *and* a concrete label based on
their kind (e.g. ``:Chunk:Chapter``, ``:Chunk:Article``, ``:Chunk:Paragraph``).
The shared label makes the loader simple; the concrete label keeps traversal
expressive.

Concrete labels
---------------

* ``Document``     — the root
* ``Chapter``      — ``Chương``
* ``Section``      — ``Mục``     (optional)
* ``Article``      — ``Điều``
* ``Subarticle``   — sub-clause heading
* ``Paragraph``    — body paragraph
* ``List``         — ``<ul>`` / ``<ol>``
* ``Table``        — ``<table>``

Common properties
-----------------

* ``id``          (string, unique)
* ``kind``        (string)
* ``title``       (string)
* ``text``        (string, body text)
* ``heading_path`` (list[string], breadcrumb)
* ``depth``       (int, 0..5)
* ``char_count``  (int)
* ``source_doc``  (string)

Edges
-----

* ``(:Chunk)-[:HAS_CHILD]->(:Chunk)`` — tree containment
* ``(:Chunk)-[:NEXT]->(:Chunk)``     — linear reading flow
* ``(:Chunk)-[:HAS_VECTOR]->(:Embedding)`` — links a leaf chunk to its
                                              standalone ``Embedding`` node.
                                              The embedding node holds the
                                              dense vector under ``vec`` so
                                              it can be indexed.

The loader also creates a Neo4j **vector index** (``chunk_embedding`` by
default) over the leaf embeddings so the demo can run
``db.index.vector.queryNodes`` for semantic search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from neo4j import Driver

from .neo4j_client import session_scope


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Concrete label per ``chunk.kind``. This is the only place the hierarchy
# is enumerated, so changing the schema means editing this dict.
_KIND_LABEL: dict[str, str] = {
    "document": "Document",
    "chapter": "Chapter",
    "section": "Section",
    "article": "Article",
    "subarticle": "Subarticle",
    "paragraph": "Paragraph",
    "list": "List",
    "table": "Table",
}

# Kinds we expose a vector embedding for. Anything outside this set is
# stored as a structural node without ``HAS_VECTOR`` — chapters and
# sections are containers, not searchable leaves.
_LEAF_KINDS: frozenset[str] = frozenset(
    {"paragraph", "list", "table", "article", "subarticle"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class GraphSummary:
    n_chunks: int
    n_nodes: int
    n_has_child: int
    n_next: int
    n_embeddings: int
    vector_index: str


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------


CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT embedding_id IF NOT EXISTS FOR (n:Embedding) "
    "REQUIRE n.id IS UNIQUE",
]


INDEXES: list[str] = [
    "CREATE INDEX chunk_kind IF NOT EXISTS FOR (n:Chunk) ON (n.kind)",
]


def ensure_schema(driver: Driver, *, vector_index: str, dim: int) -> None:
    """Create constraints, indexes and the vector index."""
    with session_scope(driver) as sess:
        for stmt in CONSTRAINTS + INDEXES:
            sess.run(stmt)
        # Vector index — Neo4j 5.11+ syntax.
        sess.run(
            f"""
            CREATE VECTOR INDEX {vector_index} IF NOT EXISTS
            FOR (n:Embedding)
            ON (n.vec)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )


def wipe(driver: Driver) -> None:
    """Delete every node & edge in the target database.

    The loader is idempotent and re-runnable; the wipe keeps repeated demos
    honest. Production code should avoid this and use MERGE only.
    """
    with session_scope(driver) as sess:
        sess.run("MATCH (n) DETACH DELETE n")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_chunks_and_vectors(
    driver: Driver,
    *,
    chunks: list[dict],
    vectors: np.ndarray,
    ids: list[str],
    batch_size: int = 200,
) -> GraphSummary:
    """Upsert nodes, edges and embeddings into Neo4j.

    Parameters
    ----------
    chunks:
        Output of Bước 1 (``chunk_document``) — list of dicts.
    vectors, ids:
        Output of Bước 2 — vectors aligned to ``chunks`` by index.
    """
    vec_by_id: dict[str, np.ndarray] = {cid: vectors[i] for i, cid in enumerate(ids)}
    n_chunks = len(chunks)

    # Split chunks by kind so each MERGE can hard-code its concrete label
    # (Cypher does not allow parameterised labels).
    by_kind: dict[str, list[dict]] = {}
    for c in chunks:
        by_kind.setdefault(c["kind"], []).append(c)

    with session_scope(driver) as sess:
        # ---- 1. Chunk nodes (one kind per pass) -----------------------
        # Each pass uses a Cypher template that hard-codes its concrete
        # label (e.g. ``:Chapter``). This avoids any APOC dependency.
        for kind, items in by_kind.items():
            label = _KIND_LABEL.get(kind, "Chunk")
            template = _NODE_CYPHER_TEMPLATE.format(label=label)
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                sess.run(
                    template,
                    rows=[_chunk_row(c) for c in batch],
                )

        # ---- 2. Embedding nodes + HAS_VECTOR edges --------------------
        leaf_chunks = [c for c in chunks if c["kind"] in _LEAF_KINDS]
        for start in range(0, len(leaf_chunks), batch_size):
            batch = leaf_chunks[start : start + batch_size]
            sess.run(
                _EMBEDDING_CYPHER,
                rows=[
                    {
                        "id": f"emb-{c['id']}",
                        "source_chunk": c["id"],
                        "vec": vec_by_id[c["id"]].tolist(),
                    }
                    for c in batch
                ],
            )

        # ---- 3. HAS_CHILD edges --------------------------------------
        for start in range(0, n_chunks, batch_size):
            batch = chunks[start : start + batch_size]
            sess.run(
                _HAS_CHILD_CYPHER,
                rows=[{"pid": c["parent_id"], "cid": c["id"]} for c in batch if c.get("parent_id")],
            )

        # ---- 4. NEXT edges -------------------------------------------
        for start in range(0, n_chunks, batch_size):
            batch = chunks[start : start + batch_size]
            sess.run(
                _NEXT_CYPHER,
                rows=[{"a": c["id"], "b": c["next_id"]} for c in batch if c.get("next_id")],
            )

    return _summarise(driver)


# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------


# One template per concrete label. The placeholder ``{label}`` is filled
# in at runtime by ``load_chunks_and_vectors``.
_NODE_CYPHER_TEMPLATE = """
UNWIND $rows AS row
MERGE (n:Chunk {{id: row.id}})
ON CREATE SET
    n:{label},
    n.kind = row.kind,
    n.title = row.title,
    n.text = row.text,
    n.heading_path = row.heading_path,
    n.parent_id = row.parent_id,
    n.next_id = row.next_id,
    n.depth = row.depth,
    n.char_count = row.char_count,
    n.source_doc = row.source_doc
RETURN count(n)
"""


_EMBEDDING_CYPHER = """
UNWIND $rows AS row
MERGE (e:Embedding {id: row.id})
ON CREATE SET
    e.vec = row.vec,
    e.source_chunk = row.source_chunk
WITH e, row
MATCH (c:Chunk {id: row.source_chunk})
MERGE (c)-[:HAS_VECTOR]->(e)
"""


_HAS_CHILD_CYPHER = """
UNWIND $rows AS row
MATCH (p:Chunk {id: row.pid})
MATCH (c:Chunk {id: row.cid})
MERGE (p)-[:HAS_CHILD]->(c)
"""


_NEXT_CYPHER = """
UNWIND $rows AS row
MATCH (a:Chunk {id: row.a})
MATCH (b:Chunk {id: row.b})
MERGE (a)-[:NEXT]->(b)
"""


def _chunk_row(c: dict) -> dict:
    """Build a row dict for the UNWIND above."""
    return {
        "id": c["id"],
        "kind": c["kind"],
        "title": c.get("title", "") or "",
        "text": c.get("text", "") or "",
        "heading_path": c.get("heading_path") or [],
        "parent_id": c.get("parent_id"),
        "next_id": c.get("next_id"),
        "depth": int(c.get("depth", 0)),
        "char_count": int(c.get("char_count", 0)),
        "source_doc": c.get("source_doc"),
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


_SUMMARY_CYPHER = """
MATCH (c:Chunk)
WITH count(c) AS n_chunks
MATCH ()-[r:HAS_CHILD]->()
WITH n_chunks, count(r) AS n_has_child
MATCH ()-[nx:NEXT]->()
WITH n_chunks, n_has_child, count(nx) AS n_next
MATCH (:Chunk)-[:HAS_VECTOR]->(e:Embedding)
RETURN n_chunks, n_has_child, n_next, count(e) AS n_emb
"""


def _summarise(driver: Driver) -> GraphSummary:
    with session_scope(driver) as sess:
        rec = sess.run(_SUMMARY_CYPHER).single()
    return GraphSummary(
        n_chunks=int(rec["n_chunks"]),
        n_nodes=int(rec["n_chunks"]),
        n_has_child=int(rec["n_has_child"]),
        n_next=int(rec["n_next"]),
        n_embeddings=int(rec["n_emb"]),
        vector_index="(configured separately)",
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return list(data["ids"]), data["vectors"]