"""Load Neo4j connection settings from ``.env.graph_rag``.

Tries to honour a ``NEO4J_DATABASE=kb-hops`` value but falls back to the
default ``neo4j`` database when running against a Community Edition that
does not support multiple databases.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_HERE = Path(__file__).resolve().parent
_ENV_CANDIDATES: list[Path] = [
    _HERE.parent.parent / ".env.graph_rag",          # graph_rag_labs/.env.graph_rag
    _HERE.parent / ".env.graph_rag",
    Path.cwd() / ".env.graph_rag",
]


@dataclass
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str       # The DB we will WRITE to (defaults to "neo4j")
    admin_database: str # Always "system" — used for CREATE DATABASE
    vector_index: str
    kb_label: str       # Extra label applied to every kb-hops node
    kb_db_requested: str  # Original requested DB name (e.g. "kb-hops")


def load_settings() -> Neo4jSettings:
    for candidate in _ENV_CANDIDATES:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break

    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password or password == "change_me":
        raise RuntimeError(
            "NEO4J_PASSWORD is unset or still 'change_me'. "
            "Copy `.env.graph_rag.template` to `.env.graph_rag` and "
            "fill in your real Neo4j password."
        )

    requested = os.environ.get("NEO4J_DATABASE", "kb-hops")
    return Neo4jSettings(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=password,
        database=os.environ.get("NEO4J_FALLBACK_DATABASE", "neo4j"),
        admin_database="system",
        vector_index=os.environ.get("NEO4J_VECTOR_INDEX_KBHOPS", "kbhops_chunk_embedding"),
        # The label always uses an underscore (Neo4j disallows hyphens).
        kb_label=os.environ.get("NEO4J_KB_LABEL", "kbhops"),
        kb_db_requested=requested,
    )