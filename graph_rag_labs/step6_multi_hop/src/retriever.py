"""Multi-hop vector + graph retriever over the kb-hops knowledge graph.

Pipeline
--------

1. **Embed** the user query with the Vietnamese MSMARCO encoder.
2. **Vector search** the Neo4j vector index for the top-``k`` similar
   :Chunk nodes.
3. **Group** the chunks by their parent :Document (already linked via
   ``PART_OF``).
4. **Multi-hop expansion** — for each parent document, walk the
   cross-document relationships (``CAN_CU``, ``THAY_THE``, ``HOP_NHAT``)
   up to ``num_hops`` steps in either direction. The traversal returns
   additional documents and the chunks that hang off them.
5. **Score aggregation** — the original chunks keep their vector
   similarity score; appended multi-hop chunks inherit the parent's
   score (decayed by hops) so the caller can rank them.
6. **Return** a list of :class:`RetrievedChunk` records plus a list of
   :class:`RetrievedDocument` records describing the trajectory.

The implementation is fully self-contained (no APOC, no extra plugins).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from neo4j import Driver

from .embedder import QueryEmbedder
from .neo4j_client import open_driver, session_scope


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    chunk_id: str
    kind: str
    title: str
    text: str
    heading_path: list[str]
    parent_doc_id: str
    score: float
    source: str          # "vector" | "hop:1" | "hop:2" | …
    via_doc: str | None = None  # which document triggered the hop


@dataclass
class RetrievedDocument:
    """A document the retriever touched, with its path from the originals."""

    doc_id: str
    title: str
    doc_type: str
    is_root: bool
    score: float
    hops: int             # 0 = matched directly, 1+ = reached via graph
    via_relationship: str | None
    path: list[str]       # chain of doc ids from the matched root


@dataclass
class QueryResult:
    """Full result of one multi-hop search."""

    query: str
    top_k: int
    num_hops: int
    chunks: list[RetrievedChunk] = field(default_factory=list)
    documents: list[RetrievedDocument] = field(default_factory=list)

    def chunks_by_source(self) -> dict[str, list[RetrievedChunk]]:
        out: dict[str, list[RetrievedChunk]] = {}
        for c in self.chunks:
            out.setdefault(c.source, []).append(c)
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MultiHopRetriever:
    """Encapsulates the connection + encoder; exposes :meth:`search`."""

    def __init__(
        self,
        *,
        driver: Driver | None = None,
        embedder: QueryEmbedder | None = None,
        database: str | None = None,
        vector_index: str = "kbhops_chunk_embedding",
    ) -> None:
        from .config import load_settings

        self.settings = load_settings()
        self.driver = driver or open_driver(self.settings)
        self.database = database or self.settings.database
        self.vector_index = vector_index or self.settings.vector_index
        self.embedder = embedder or QueryEmbedder()

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "MultiHopRetriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Cypher queries
    # ------------------------------------------------------------------

    def _vector_search(self, query_vec: list[float], top_k: int) -> list[dict]:
        """Step 1: dense-vector search over the chunk embeddings.

        For container chunks (chapter / article / section) whose body
        text is empty, also fetch the concatenated text of their direct
        children (paragraph / list / table) so the LLM has the actual
        legal text to read.
        """
        with session_scope(self.driver, self.database) as sess:
            rows = sess.run(
                f"""
                CALL db.index.vector.queryNodes(
                    '{self.vector_index}', $k, $vec
                )
                YIELD node AS c, score
                MATCH (d:kbhopsDocument)<-[:PART_OF]-(c)
                OPTIONAL MATCH (c)-[:PARENT_OF]->(child:kbhopsChunk)
                WHERE child.kind IN ['paragraph', 'list', 'table']
                WITH c, score, d,
                     collect(DISTINCT child.text)[0..5] AS child_texts
                RETURN c.id AS chunk_id, c.kind AS kind,
                       c.title AS title, c.text AS text,
                       c.heading_path AS heading_path,
                       c.parent_id AS parent_chunk_id,
                       d.id AS doc_id, d.title AS doc_title,
                       d.original_doc_id AS doc_original_id,
                       d.is_root AS doc_is_root,
                       score,
                       child_texts
                ORDER BY score DESC
                LIMIT $k
                """,
                k=top_k,
                vec=query_vec,
            ).data()

        # If a container chunk has empty text, fall back to its
        # children's text. This makes article hits much more useful.
        for row in rows:
            if not row["text"] and row["child_texts"]:
                joined = "\n\n".join(t for t in row["child_texts"] if t)
                row["text"] = joined[:4000]  # cap to avoid runaway prompts
        return rows

    def _expand_hops(
        self,
        start_doc_ids: list[str],
        *,
        num_hops: int,
        hop_decay: float = 0.85,
    ) -> list[RetrievedDocument]:
        """Step 2: walk CAN_CU/THAY_THE/HOP_NHAT from each start doc.

        Traverses BOTH directions (incoming + outgoing) of the
        relationship so we can grab documents that reference the matched
        doc. The decay multiplier weights how far a hit is from the seed.
        """
        if num_hops <= 0 or not start_doc_ids:
            return []

        with session_scope(self.driver, self.database) as sess:
            rows = sess.run(
                """
                MATCH (start:kbhopsDocument)
                WHERE start.id IN $start_ids
                MATCH path = (start)-[rels:CAN_CU|THAY_THE|HOP_NHAT*1..""" + str(num_hops) + """]-(related:kbhopsDocument)
                WHERE related.id <> start.id
                WITH related,
                     rels   AS rels,
                     nodes(path) AS path_nodes,
                     length(path) AS hops
                RETURN DISTINCT related.id AS doc_id,
                       related.title AS title,
                       related.doc_type AS doc_type,
                       related.is_root AS is_root,
                       [n IN path_nodes | n.id] AS path_ids,
                       [rel IN rels | type(rel)] AS rel_types,
                       hops
                ORDER BY hops ASC
                """,
                start_ids=start_doc_ids,
            ).data()

        out: list[RetrievedDocument] = []
        seen: set[tuple[str, int]] = set()
        for row in rows:
            hops = int(row["hops"])
            rel_types = row.get("rel_types") or []
            via_rel = rel_types[0] if rel_types else None
            key = (row["doc_id"], hops)
            if key in seen:
                continue
            seen.add(key)
            score = 0.85 ** hops
            out.append(
                RetrievedDocument(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    doc_type=row["doc_type"] or "unknown",
                    is_root=bool(row["is_root"]),
                    score=score,
                    hops=hops,
                    via_relationship=via_rel,
                    path=list(row["path_ids"]),
                )
            )
        return out

    def _chunks_for_documents(
        self,
        doc_ids: list[str],
        *,
        max_per_doc: int = 3,
    ) -> list[RetrievedChunk]:
        """Fetch the top-``max_per_doc`` chunks for each document id."""
        if not doc_ids:
            return []
        with session_scope(self.driver, self.database) as sess:
            rows = sess.run(
                """
                MATCH (d:kbhopsDocument)<-[:PART_OF]-(c:kbhopsChunk)
                WHERE d.id IN $ids AND c.text IS NOT NULL AND c.text <> ''
                WITH d, c, c.char_count AS size
                ORDER BY size DESC
                WITH d, collect(c)[0..$max] AS picked
                UNWIND picked AS c
                RETURN d.id AS doc_id,
                       c.id AS chunk_id,
                       c.kind AS kind,
                       c.title AS title,
                       c.text AS text,
                       c.heading_path AS heading_path
                """,
                ids=doc_ids,
                max=max_per_doc,
            ).data()
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                kind=r["kind"],
                title=r["title"] or "",
                text=r["text"] or "",
                heading_path=r["heading_path"] or [],
                parent_doc_id=r["doc_id"],
                score=0.0,
                source="hop",
                via_doc=r["doc_id"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        num_hops: int = 1,
        max_extra_chunks_per_doc: int = 3,
    ) -> QueryResult:
        """Run a multi-hop retrieval.

        Parameters
        ----------
        query:
            The user query in Vietnamese.
        top_k:
            Number of vector-search hits to keep for the seed.
        num_hops:
            Number of graph hops to expand. ``0`` disables expansion and
            returns only the vector matches.
        max_extra_chunks_per_doc:
            How many chunks per expansion doc to append to the result.
        """
        result = QueryResult(query=query, top_k=top_k, num_hops=num_hops)

        # 1. Embed query
        query_vec = self.embedder.encode(query)

        # 2. Vector search
        seed_rows = self._vector_search(query_vec, top_k)
        if not seed_rows:
            return result

        # 3. Seed chunks
        seed_chunks: list[RetrievedChunk] = []
        seed_doc_ids: list[str] = []
        for row in seed_rows:
            seed_chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    kind=row["kind"],
                    title=row["title"] or "",
                    text=row["text"] or "",
                    heading_path=row["heading_path"] or [],
                    parent_doc_id=row["doc_id"],
                    score=float(row["score"]),
                    source="vector",
                )
            )
            if row["doc_id"] not in seed_doc_ids:
                seed_doc_ids.append(row["doc_id"])

        # Seed documents (the matched docs themselves).
        seed_documents = [
            RetrievedDocument(
                doc_id=row["doc_id"],
                title=row["doc_title"],
                doc_type="thong_tu",
                is_root=bool(row["doc_is_root"]),
                score=float(row["score"]),
                hops=0,
                via_relationship=None,
                path=[row["doc_id"]],
            )
            for row in seed_rows
        ]

        # Deduplicate docs by id keeping the highest score.
        by_id: dict[str, RetrievedDocument] = {}
        for d in seed_documents:
            prev = by_id.get(d.doc_id)
            if prev is None or d.score > prev.score:
                by_id[d.doc_id] = d
        result.documents = list(by_id.values())
        result.chunks = seed_chunks

        # 4. Multi-hop expansion
        if num_hops > 0:
            expanded_docs = self._expand_hops(
                seed_doc_ids, num_hops=num_hops
            )
            for d in expanded_docs:
                # Don't overwrite a seed doc with a lower-scored hop doc.
                if d.doc_id in by_id:
                    continue
                by_id[d.doc_id] = d
                result.documents.append(d)

            # 5. Pull representative chunks from each expanded doc.
            extra_doc_ids = [d.doc_id for d in expanded_docs if d.doc_id not in seed_doc_ids]
            extra_chunks = self._chunks_for_documents(
                extra_doc_ids, max_per_doc=max_extra_chunks_per_doc
            )
            for c in extra_chunks:
                # Score = parent doc score (decayed by hop count).
                parent = by_id.get(c.parent_doc_id)
                if parent is None:
                    c.score = 0.0
                    c.source = "hop"
                else:
                    c.score = parent.score
                    c.source = f"hop:{parent.hops}"
            result.chunks.extend(extra_chunks)

        # Sort: vector hits first by score, then hop chunks by score.
        result.chunks.sort(key=lambda x: (-x.score, x.source != "vector"))
        return result