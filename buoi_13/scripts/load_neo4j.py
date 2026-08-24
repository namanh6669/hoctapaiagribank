#!/usr/bin/env python3
"""Load outputs/entities.csv + relations.csv vào Neo4j (chỉ phạm vi MVP).

Yêu cầu:
    pip install -r requirements.txt        # cài 'neo4j' + 'python-dotenv'
    Tạo file `.env` ở thư mục gốc dự án (xem `.env.example`).
    Neo4j đang chạy và database đã tồn tại.

Đặc điểm thiết kế:
    - Đọc `.env` (NEO4J_URI / USER / PASSWORD / DATABASE) — KHÔNG hard-code.
    - Dùng MERGE thay cho CREATE để idempotent (chạy lại không tạo duplicate).
    - Truyền dữ liệu qua tham số (parametrized Cypher), không string-interpolate.
    - Áp dụng `cypher/schema.cypher` trước khi load dữ liệu.
    - Nếu thiếu biến môi trường, driver chưa cài, hoặc Neo4j không chạy:
      báo lý do rõ ràng và thoát với exit code != 0. KHÔNG đụng tới
      `wiki/` hay `outputs/` đã có từ các bước trước.

Cách chạy:
    python scripts/load_neo4j.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENTITIES_PATH = REPO / "outputs" / "entities.csv"
RELATIONS_PATH = REPO / "outputs" / "relations.csv"
SCHEMA_PATH = REPO / "cypher" / "schema.cypher"
ENV_PATH = REPO / ".env"

REQUIRED = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"]

# Ánh xạ type (trong entities.csv) -> nhãn Neo4j
LABEL_BY_TYPE = {
    "RuiRo": "RuiRo",
    "KiemSoat": "KiemSoat",
    "SuKienRuiRo": "SuKienRuiRo",
}

# Quan hệ MVP: relationship_type -> (source_label, target_label)
REL_TEMPLATES = {
    "MITIGATES": ("KiemSoat", "RuiRo"),
    "OBSERVED_AS": ("RuiRo", "SuKienRuiRo"),
}

HEADER = "=" * 70


def load_env_file() -> None:
    """Đọc `.env` (nếu có) — không ghi đè biến môi trường đã được set sẵn."""
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def none_for_empty(row: dict[str, str]) -> dict:
    """Chuỗi rỗng -> None (để Cypher lưu là NULL, không phải chuỗi rỗng)."""
    return {k: (v if v else None) for k, v in row.items()}


def split_statements(cypher_text: str) -> list[str]:
    """Tách file schema thành các statement, bỏ qua comment `//`."""
    out: list[str] = []
    cur: list[str] = []
    for line in cypher_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        cur.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(cur).rstrip().rstrip(";").strip()
            if stmt:
                out.append(stmt)
            cur = []
    tail = "\n".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def print_env_instructions() -> None:
    print("""
  Tạo file `.env` ở thư mục gốc dự án (xem `.env.example`):

      NEO4J_URI=bolt://localhost:7687
      NEO4J_USER=neo4j
      NEO4J_PASSWORD=your_password_here
      NEO4J_DATABASE=neo4j

  Rồi chạy lại:  python scripts/load_neo4j.py
""")


def print_install_instructions() -> None:
    print("\n  Cài đặt Python driver:")
    print("      pip install -r requirements.txt\n")


def main() -> int:
    print(HEADER)
    print(" Wiki Risk Graph — Load outputs/ → Neo4j (MVP) ")
    print(HEADER)

    load_env_file()

    cfg = {k: os.environ.get(k, "") for k in REQUIRED}
    missing = [k for k in REQUIRED if not cfg[k]]
    if missing:
        print(f"❌ Thiếu biến môi trường: {', '.join(missing)}")
        print_env_instructions()
        print("ℹ  Wiki scripts vẫn chạy bình thường — phần Neo4j là tuỳ chọn.")
        return 1

    print(f"  URI     : {cfg['NEO4J_URI']}")
    print(f"  USER    : {cfg['NEO4J_USER']}")
    print(f"  DATABASE: {cfg['NEO4J_DATABASE']}")
    print(f"  PASSWORD: {'*' * len(cfg['NEO4J_PASSWORD'])}  (không log)")
    print()

    # Check neo4j driver
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        print("❌ Chưa cài Python driver 'neo4j'.")
        print_install_instructions()
        print("ℹ  Wiki scripts không cần dep này — vẫn dùng được.")
        return 1

    # Try connect
    print(f"Kết nối tới {cfg['NEO4J_URI']} ...")
    try:
        driver = GraphDatabase.driver(
            cfg["NEO4J_URI"],
            auth=(cfg["NEO4J_USER"], cfg["NEO4J_PASSWORD"]),
        )
        driver.verify_connectivity()
        print("✓ Kết nối OK\n")
    except Exception as exc:
        print(f"❌ Không kết nối được:\n   {type(exc).__name__}: {exc}")
        print("\nHướng dẫn:")
        print("  1. Cài Neo4j: https://neo4j.com/download/")
        print("  2. Tạo project, start database (đặt tên trùng NEO4J_DATABASE).")
        print("  3. Kiểm tra URI + USER + PASSWORD trong .env.")
        print("  4. Chạy lại: python scripts/load_neo4j.py")
        print("\nℹ  Wiki scripts không phụ thuộc Neo4j — vẫn chạy bình thường.")
        return 1

    database = cfg["NEO4J_DATABASE"]

    # Apply schema
    if SCHEMA_PATH.exists():
        stmts = split_statements(SCHEMA_PATH.read_text(encoding="utf-8"))
        print(f"Áp dụng {SCHEMA_PATH.relative_to(REPO).as_posix()} ({len(stmts)} statement) ...")
        with driver.session(database=database) as sess:
            for stmt in stmts:
                sess.run(stmt)
        print("✓ Schema OK\n")
    else:
        print(f"⚠ Không tìm thấy {SCHEMA_PATH}, bỏ qua schema.\n")

    # Read CSVs
    entities = read_csv(ENTITIES_PATH)
    relations = read_csv(RELATIONS_PATH)
    print(f"Đọc: {len(entities)} entity, {len(relations)} relation")

    # Load nodes via MERGE, parametrized
    print("\nLoad node (MERGE theo `id`, parameterized) ...")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        etype = e["type"]
        if etype not in LABEL_BY_TYPE:
            print(f"  ⚠ Bỏ qua type không thuộc MVP: {etype}")
            continue
        by_type[etype].append(none_for_empty(e))

    with driver.session(database=database) as sess:
        for label, rows in by_type.items():
            cypher = (
                "UNWIND $rows AS row "
                f"MERGE (n:{label} {{id: row.id}}) "
                "SET n += row"
            )
            sess.run(cypher, rows=rows)
            print(f"  ✓ {label}: {len(rows)} node")

    # Load edges via MERGE, parametrized
    print("\nLoad edge (MERGE theo cặp source/target, parameterized) ...")
    edges_by_rt: dict[str, list[dict]] = defaultdict(list)
    skipped_rt: list[str] = []
    for r in relations:
        rt = r["relationship_type"]
        if rt not in REL_TEMPLATES:
            if rt not in skipped_rt:
                print(f"  ⚠ Bỏ qua loại quan hệ ngoài MVP: {rt}")
                skipped_rt.append(rt)
            continue
        # Không truyền source_id / target_id / relationship_type vào properties của edge
        edge_props = {
            k: (v if v else None)
            for k, v in r.items()
            if k not in ("source_id", "target_id", "relationship_type")
        }
        edges_by_rt[rt].append({
            "source_id": r["source_id"],
            "target_id": r["target_id"],
            "props": edge_props,
        })

    with driver.session(database=database) as sess:
        for rt, rows in edges_by_rt.items():
            src_lbl, tgt_lbl = REL_TEMPLATES[rt]
            cypher = (
                "UNWIND $rows AS row "
                f"MATCH (s:{src_lbl} {{id: row.source_id}}) "
                f"MATCH (t:{tgt_lbl} {{id: row.target_id}}) "
                f"MERGE (s)-[r:{rt}]->(t) "
                "SET r += row.props"
            )
            sess.run(cypher, rows=rows)
            print(f"  ✓ {rt}: {len(rows)} edge")

    # Sanity check
    print("\nKiểm tra nhanh trong database:")
    with driver.session(database=database) as sess:
        for label in ("RuiRo", "KiemSoat", "SuKienRuiRo"):
            n = sess.run(
                f"MATCH (n:{label}) RETURN count(n) AS c"
            ).single()["c"]
            print(f"  {label}: {n}")
        for rt in ("MITIGATES", "OBSERVED_AS"):
            n = sess.run(
                f"MATCH ()-[r:{rt}]->() RETURN count(r) AS c"
            ).single()["c"]
            print(f"  {rt}: {n}")

    driver.close()
    print(f"\n{HEADER}")
    print("✓ Hoàn tất load.")
    print(HEADER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
