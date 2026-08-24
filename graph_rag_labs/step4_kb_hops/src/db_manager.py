"""Manage the ``kb-hops`` database and connection.

Community Edition only supports a single user database (``neo4j``); Enterprise
allows ``CREATE DATABASE kb-hops``. This helper:

1. Tries to create the requested DB. If it already exists → fine.
2. If the call fails with ``UnsupportedAdministrationCommand`` or
   ``DatabaseNotFound``, it falls back to writing into ``neo4j`` and
   namespaces the data via an extra label (e.g. ``:kbhops``).

It exposes :func:`get_target_database` so the loader / demo can ask
"where do I write?" without hard-coding either path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase, Session

from .config import Neo4jSettings


@dataclass
class DbLocation:
    """Where the loader should write."""

    name: str                # "kb-hops" if Enterprise, else "neo4j"
    is_default: bool         # True when we fell back to default DB
    note: str                # Human-readable explanation for the demo log


def open_driver(settings: Neo4jSettings) -> Driver:
    """Create a Bolt driver — connects to the default DB so we can run
    admin commands in the ``system`` database if needed."""
    return GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))


@contextmanager
def session_scope(driver: Driver, database: str | None = None) -> Iterator[Session]:
    sess = driver.session(database=database) if database else driver.session()
    try:
        yield sess
    finally:
        sess.close()


def ensure_database(driver: Driver, settings: Neo4jSettings) -> DbLocation:
    """Try to create ``kb-hops``; fall back to ``neo4j`` for Community."""
    requested = settings.kb_db_requested

    # Step 1: list existing databases via the admin (system) database.
    try:
        with session_scope(driver, settings.admin_database) as sess:
            existing = sess.run("SHOW DATABASES").data()
            names = {row["name"] for row in existing}
    except Exception as exc:  # noqa: BLE001
        # Older versions / locked-down systems → fall back immediately.
        return DbLocation(
            name=settings.database,
            is_default=True,
            note=f"Không truy vấn được SHOW DATABASES ({exc}); dùng database mặc định.",
        )

    if requested in names:
        return DbLocation(
            name=requested,
            is_default=False,
            note=f"Database {requested!r} đã có sẵn, dùng luôn.",
        )

    # Step 2: try CREATE DATABASE via the admin DB.
    try:
        with session_scope(driver, settings.admin_database) as sess:
            sess.run(f"CREATE DATABASE `{requested}`")
        return DbLocation(
            name=requested,
            is_default=False,
            note=f"Đã tạo database mới {requested!r}.",
        )
    except Exception as exc:  # noqa: BLE001
        # Community Edition does not allow CREATE DATABASE. Fall back.
        return DbLocation(
            name=settings.database,
            is_default=True,
            note=(
                f"Không thể CREATE DATABASE {requested!r} "
                f"({type(exc).__name__}: {exc}). "
                f"Community Edition chỉ có 1 database — sử dụng "
                f"{settings.database!r} với label phụ :{settings.kb_label} "
                f"để cô lập dữ liệu kb-hops."
            ),
        )