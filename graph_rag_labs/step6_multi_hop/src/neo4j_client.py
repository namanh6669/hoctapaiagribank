"""Thin Neo4j driver wrapper for step6_multi_hop."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from neo4j import Driver, GraphDatabase, Session

from .config import Neo4jSettings


def open_driver(settings: Neo4jSettings) -> Driver:
    return GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))


@contextmanager
def session_scope(driver: Driver, database: str | None = None) -> Iterator[Session]:
    sess = driver.session(database=database) if database else driver.session()
    try:
        yield sess
    finally:
        sess.close()