"""Thin wrapper around the official Neo4j Python driver.

The wrapper:

* reads connection settings via :mod:`config`,
* exposes a small context-manager API,
* runs the same Cypher repeatedly with the official driver's session
  ``execute_write`` / ``execute_read`` helpers (so transactions are
  committed atomically and parameters are bound safely).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from neo4j import Driver, GraphDatabase, Session, unit_of_work

from .config import Neo4jSettings, load_settings


def open_driver(settings: Neo4jSettings | None = None) -> Driver:
    """Create a Bolt driver from env-loaded settings."""
    s = settings or load_settings()
    return GraphDatabase.driver(s.uri, auth=(s.user, s.password))


@contextmanager
def session_scope(driver: Driver, database: str | None = None) -> Iterator[Session]:
    """Yield a session and close it deterministically."""
    sess = driver.session(database=database) if database else driver.session()
    try:
        yield sess
    finally:
        sess.close()


# Re-export the decorator so callers can do ``from .neo4j_client import unit_of_work``.
__all__ = ["open_driver", "session_scope", "load_settings", "Neo4jSettings", "unit_of_work"]