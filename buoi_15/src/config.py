"""Centralised configuration for Buoi 15 — RBAC.

Importing ``VALID_ROLES`` from this module is the single source of truth
so we cannot fat-finger a role name in any script, Streamlit page, or
audit test.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Working directory & .env
# ---------------------------------------------------------------------------

# buoi_15/ — the directory this file lives in (.. from src/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env from buoi_15/ (override=False so real env still wins).
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Roles (RBAC)
# ---------------------------------------------------------------------------

VALID_ROLES: frozenset[str] = frozenset(
    {
        "Admin",
        "HR_Manager",
        "Risk_Officer",
        "Employee",
        "Guest",
    }
)

# Convenience alias kept as a tuple so iteration order is stable / displayable.
ROLE_LIST: tuple[str, ...] = (
    "Admin",
    "HR_Manager",
    "Risk_Officer",
    "Employee",
    "Guest",
)

# Default security label each role is allowed to see at minimum.
# Higher roles are still allowed to see lower labels — see
# ROLE_ACCESS_MATRIX below.
ROLE_DEFAULT_LABEL: dict[str, str] = {
    "Admin": "Restricted",
    "HR_Manager": "Confidential",
    "Risk_Officer": "Confidential",
    "Employee": "Internal",
    "Guest": "Public",
}

# Security labels in ascending order of sensitivity.
SECURITY_LABELS: tuple[str, ...] = ("Public", "Internal", "Confidential", "Restricted")

# Maximum label each role is allowed to read. A role can read anything
# at-or-below its ceiling.
ROLE_MAX_LABEL: dict[str, str] = {
    "Admin": "Restricted",
    "HR_Manager": "Confidential",
    "Risk_Officer": "Confidential",
    "Employee": "Internal",
    "Guest": "Public",
}


# ---------------------------------------------------------------------------
# Database (Neo4j) — read from .env
# ---------------------------------------------------------------------------

def get_neo4j_config() -> dict[str, str]:
    """Return Neo4j connection settings, sourced from `.env`.

    Missing variables fall back to safe defaults so importing this module
    never raises — but attempting to *connect* without a password will.
    """
    return {
        "uri": os.getenv("NEO4J_URI", ""),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", ""),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }


def assert_valid_role(role: str) -> str:
    """Normalise + validate a role string. Raises ``ValueError`` on typo."""
    cleaned = role.strip()
    if cleaned not in VALID_ROLES:
        raise ValueError(
            f"Unknown role '{role}'. Valid roles are: {', '.join(ROLE_LIST)}"
        )
    return cleaned


__all__ = [
    "PROJECT_ROOT",
    "VALID_ROLES",
    "ROLE_LIST",
    "ROLE_DEFAULT_LABEL",
    "ROLE_MAX_LABEL",
    "SECURITY_LABELS",
    "get_neo4j_config",
    "assert_valid_role",
]
