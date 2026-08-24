"""Load Neo4j connection settings from ``.env.graph_rag``.

Same loader as Bước 4 — kept here so step6 can be run independently.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_HERE = Path(__file__).resolve().parent
_ENV_CANDIDATES: list[Path] = [
    _HERE.parent.parent / ".env.graph_rag",
    _HERE.parent / ".env.graph_rag",
    Path.cwd() / ".env.graph_rag",
]


@dataclass
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str
    vector_index: str
    kb_label: str


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

    return Neo4jSettings(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=password,
        database=os.environ.get("NEO4J_FALLBACK_DATABASE", "neo4j"),
        vector_index=os.environ.get(
            "NEO4J_VECTOR_INDEX_KBHOPS", "kbhops_chunk_embedding"
        ),
        kb_label=os.environ.get("NEO4J_KB_LABEL", "kbhops"),
    )