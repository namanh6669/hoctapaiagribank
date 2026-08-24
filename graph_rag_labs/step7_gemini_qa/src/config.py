"""Load settings for step7_gemini_qa.

Reuses the same ``.env.graph_rag`` machinery as the previous steps and
adds the Gemini API key + model name.
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
class Settings:
    # Neo4j (still needed to drive the multi-hop retriever).
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    vector_index: str

    # Gemini.
    gemini_api_key: str
    gemini_model: str


def load_settings() -> Settings:
    for candidate in _ENV_CANDIDATES:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break

    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
    if not neo4j_password or neo4j_password == "change_me":
        raise RuntimeError(
            "NEO4J_PASSWORD is unset or still 'change_me'. "
            "Copy `.env.graph_rag.template` to `.env.graph_rag` and "
            "fill in your real values."
        )

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError(
            "GEMINI_API_KEY is unset. Edit `.env.graph_rag` and add "
            "your key (https://aistudio.google.com/apikey)."
        )

    return Settings(
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=neo4j_password,
        neo4j_database=os.environ.get("NEO4J_FALLBACK_DATABASE", "neo4j"),
        vector_index=os.environ.get(
            "NEO4J_VECTOR_INDEX_KBHOPS", "kbhops_chunk_embedding"
        ),
        gemini_api_key=gemini_key,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
    )