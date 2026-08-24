"""BƯỚC 7: kiểm tra kết nối Neo4j.

- Đọc cấu hình từ .env
- KHÔNG in password
- Driver official neo4j
- Verify connectivity + chạy query đọc đơn giản
- Đóng driver

KHÔNG import dữ liệu.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 7 — NEO4J CONNECTIVITY CHECK")

    # 1) Load .env
    env_path = BASE / ".env"
    if not env_path.exists():
        print(f"FAIL: .env not found at {env_path}")
        return 1
    load_dotenv(env_path, verbose=False)

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    # 2) Sanity: không in password, chỉ in bool
    print(f"  Config loaded from .env:")
    print(f"    NEO4J_URI      : {uri}")
    print(f"    NEO4J_USER     : {user}")
    print(f"    NEO4J_PASSWORD : {'<set, ' + str(len(password)) + ' chars>' if password else '<NOT SET>'}")
    print(f"    NEO4J_DATABASE : {database}")

    if not password:
        print("FAIL: NEO4J_PASSWORD is not set in .env")
        return 1

    # 3) Import official driver
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        print(f"FAIL: cannot import neo4j driver: {e}")
        return 1
    print(f"\n  Driver: neo4j Python driver (version={GraphDatabase.__module__})")

    # 4) Open driver
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"  Driver opened: {uri}")

        # 5) Verify connectivity
        try:
            driver.verify_connectivity()
            print(f"  verify_connectivity(): OK")
        except Exception as e:
            print(f"  verify_connectivity(): FAIL ({type(e).__name__}: {e})")
            return 1

        # 6) Simple read query
        try:
            with driver.session(database=database) as session:
                # Database info
                db_info = session.run("CALL dbms.components() YIELD name, versions, edition")
                comp = db_info.single()
                if comp:
                    print(f"  Neo4j {comp['name']} version: {comp['versions'][0] if comp['versions'] else '?'}, edition: {comp['edition']}")

                # DB name (use configured; `CALL db.name()` không có ở 2026.07.x)
                print(f"  Database in use      : {database}")

                # Simple read query (avoid installed-procedure probe)
                rows = list(session.run("RETURN 1 AS n, 'hello' AS msg"))
                print(f"  Read query 'RETURN 1 AS n, ...' -> rows={len(rows)}, n={rows[0]['n']}, msg={rows[0]['msg']!r}")

                # List databases via SHOW DATABASES (standard 5.x+)
                try:
                    db_rows = list(session.run("SHOW DATABASES"))
                    names = [r.get('name') for r in db_rows]
                    print(f"  Databases available : {names}")
                except Exception as e:
                    print(f"  SHOW DATABASES not available: {type(e).__name__}")

                # Graph counts
                try:
                    node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
                    rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
                    print(f"  Current graph        : {node_count} nodes, {rel_count} relationships")
                except Exception as e:
                    print(f"  Count query failed: {type(e).__name__}: {e}")

            print(f"\n  OVERALL: PASS")
            return 0
        except Exception as e:
            print(f"  Query execution: FAIL ({type(e).__name__}: {e})")
            return 1
    except Exception as e:
        print(f"  Driver open: FAIL ({type(e).__name__}: {e})")
        return 1
    finally:
        # 7) Close driver đúng cách
        if driver is not None:
            try:
                driver.close()
                print(f"  Driver closed cleanly")
            except Exception as e:
                print(f"  WARN: driver close issue: {e}")


if __name__ == "__main__":
    sys.exit(main())
