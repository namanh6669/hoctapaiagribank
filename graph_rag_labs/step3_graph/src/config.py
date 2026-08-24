"""Load Neo4j connection settings from ``.env.graph_rag``.

The loader walks up from the ``step3_graph`` folder to find the env file so
the user can keep credentials outside of version control.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Search paths, in priority order. The first match wins.
_HERE = Path(__file__).resolve().parent
_ENV_CANDIDATES: list[Path] = [
    _HERE.parent.parent / ".env.graph_rag",          # graph_rag_labs/.env.graph_rag
    _HERE.parent / ".env.graph_rag",                # step3_graph/.env.graph_rag
    Path.cwd() / ".env.graph_rag",                  # cwd
]


@dataclass
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str
    vector_index: str


def load_settings() -> Neo4jSettings:
    """Read settings from the first env file that exists."""
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

    return Neo4jSettings(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=password,
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        vector_index=os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding"),
    )