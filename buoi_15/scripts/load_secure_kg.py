#!/usr/bin/env python3
"""load_secure_kg.py — Buổi 15.

Cập nhật thuộc tính ``allowed_roles`` (List of Strings) lên các node
``VanBan`` và ``DieuKhoan`` trong Neo4j, dựa trên
``buoi_15/data/processed/chunks_secure.csv``.

Quy tắc an toàn:
- **MERGE theo ``id``** — KHÔNG DETACH DELETE, KHÔNG string-interpolate.
- Mọi query dùng ``$params``.
- ``lab_session = 'buoi_15'`` đánh dấu version đã gán tag RBAC.
- Nếu .env thiếu ``NEO4J_PASSWORD`` → SKIP push, vẫn emit Cypher ra file.

Aggregation rule cho VanBan (lựa chọn thiết kế, ghi rõ trong output):
- ``VanBan.allowed_roles`` = **INTERSECTION** của tất cả chunks trong VanBan
  (chỉ những role thấy ĐƯỢC MỌI chunk mới được mở VanBan — đây là
  default an toàn nhất cho RBAC). Có thể đổi sang UNION nếu muốn.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Make src.config importable
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_neo4j_config  # noqa: E402

LAB = "buoi_15"
INPUT = PROJECT_ROOT / "data" / "processed" / "chunks_secure.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CYPHER_OUT = OUTPUT_DIR / "load_secure_kg.cypher"
PARAMS_OUT = OUTPUT_DIR / "load_secure_kg_params.json"


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _parse_roles(s: str) -> list[str]:
    return [r.strip() for r in str(s).split(",") if r.strip()]


def build_payloads(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Return ``(vanban_payloads, dieukhoan_payloads)``.

    DieuKhoan: 1 row per chunk — giữ nguyên ``allowed_roles`` từ CSV.

    VanBan: aggregate theo INTERSECTION (an toàn nhất) + mode label.
    """
    dk_payloads: list[dict] = []
    for _, row in df.iterrows():
        dk_payloads.append(
            {
                "id": str(row["chunk_id"]).strip(),
                "document_id": str(row["document_id"]).strip(),
                "allowed_roles": _parse_roles(row["allowed_roles"]),
                "security_label": str(row["security_label"]).strip(),
            }
        )

    # Group DieuKhoan by document_id → INTERSECTION of roles
    roles_by_doc: dict[str, list[set[str]]] = defaultdict(list)
    labels_by_doc: dict[str, list[str]] = defaultdict(list)
    title_by_doc: dict[str, str] = {}
    for dk in dk_payloads:
        roles_by_doc[dk["document_id"]].append(set(dk["allowed_roles"]))
        labels_by_doc[dk["document_id"]].append(dk["security_label"])

    # Title lookup from CSV (take first non-empty per doc)
    for _, row in df.iterrows():
        did = str(row["document_id"]).strip()
        t = str(row.get("title", "")).strip()
        if did and t and did not in title_by_doc:
            title_by_doc[did] = t

    vb_payloads: list[dict] = []
    for did, role_sets in roles_by_doc.items():
        common = sorted(set.intersection(*role_sets)) if role_sets else []
        if not common:
            # Edge case: no role can see every chunk → fall back to Admin only.
            common = ["Admin"]
        mode_label = Counter(labels_by_doc[did]).most_common(1)[0][0]
        vb_payloads.append(
            {
                "id": did,
                "title": title_by_doc.get(did, ""),
                "allowed_roles": common,
                "security_label": mode_label,
            }
        )

    return vb_payloads, dk_payloads


# ---------------------------------------------------------------------------
# Cypher emit (for browser/CLI use, no Neo4j needed)
# ---------------------------------------------------------------------------


def emit_cypher(vb_payloads: list[dict], dk_payloads: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cypher = f"""// Buổi 15 — Secure KG update (auto-generated)
// Source: buoi_15/data/processed/chunks_secure.csv ({len(dk_payloads)} chunks)
// MERGE by id; SET allowed_roles (List<String>) + lab_session = 'buoi_15'
// Counts: VanBan={len(vb_payloads)} · DieuKhoan={len(dk_payloads)}
//
// ⚠️  KHÔNG DETACH DELETE. KHÔNG string-interpolate.

// ───────────────────────────────────────────
// (:VanBan) — document-level access (INTERSECTION of chunk roles)
// ───────────────────────────────────────────
UNWIND $vanbans AS v
MERGE (vb:VanBan {{id: v.id}})
SET vb.allowed_roles = v.allowed_roles,
    vb.security_label = v.security_label,
    vb.title          = coalesce(v.title, vb.title, ''),
    vb.lab_session    = '{LAB}';

// ───────────────────────────────────────────
// (:DieuKhoan) — chunk-level access (per-chunk)
// ───────────────────────────────────────────
UNWIND $dieukhoans AS d
MERGE (dk:DieuKhoan {{id: d.id}})
SET dk.allowed_roles = d.allowed_roles,
    dk.security_label = d.security_label,
    dk.document_id    = d.document_id,
    dk.lab_session    = '{LAB}';

// ───────────────────────────────────────────
// Verification queries (run after the two MERGE blocks)
// ───────────────────────────────────────────
// 1) Count nodes carrying allowed_roles:
//    MATCH (n) WHERE n.allowed_roles IS NOT NULL
//    RETURN labels(n) AS l, count(n) AS c ORDER BY c DESC;
//
// 2) Sample 1 VanBan + linked DieuKhoan:
//    MATCH (vb:VanBan {{lab_session:'{LAB}'}})
//    WITH vb, rand() AS r ORDER BY r LIMIT 1
//    OPTIONAL MATCH (vb)-[:CONTAINS]->(dk:DieuKhoan)
//    RETURN vb {{.*, allowed_roles: vb.allowed_roles}} AS vanban,
//           collect(dk {{.*, allowed_roles: dk.allowed_roles}})[..3] AS dks;
"""
    CYPHER_OUT.write_text(cypher, encoding="utf-8")

    PARAMS_OUT.write_text(
        json.dumps(
            {"vanbans": vb_payloads, "dieukhoans": dk_payloads},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Neo4j push
# ---------------------------------------------------------------------------


def push_to_neo4j(driver, database: str, vb: list[dict], dk: list[dict]) -> None:
    with driver.session(database=database) as session:
        # VanBan — also set title (read from CSV) so the node is self-describing.
        session.run(
            "UNWIND $xs AS v "
            "MERGE (vb:VanBan {id: v.id}) "
            "SET vb.allowed_roles = v.allowed_roles, "
            "    vb.security_label = v.security_label, "
            "    vb.title          = coalesce(v.title, vb.title, ''), "
            "    vb.lab_session    = $lab",
            xs=vb,
            lab=LAB,
        )
        # DieuKhoan — also persist document_id so we can navigate VanBan→DieuKhoan
        # even when the buoi_14 CONTAINS relationship isn't in the DB.
        session.run(
            "UNWIND $xs AS d "
            "MERGE (dk:DieuKhoan {id: d.id}) "
            "SET dk.allowed_roles = d.allowed_roles, "
            "    dk.security_label = d.security_label, "
            "    dk.document_id    = d.document_id, "
            "    dk.lab_session    = $lab",
            xs=dk,
            lab=LAB,
        )


# ---------------------------------------------------------------------------
# Verification queries
# ---------------------------------------------------------------------------


def verify(driver, database: str) -> dict:
    out: dict = {}
    with driver.session(database=database) as session:
        # 1) Count nodes with allowed_roles, grouped by label
        rows = session.run(
            "MATCH (n) "
            "WHERE n.allowed_roles IS NOT NULL AND n.lab_session = $lab "
            "RETURN labels(n)[0] AS lbl, count(n) AS c "
            "ORDER BY c DESC",
            lab=LAB,
        )
        out["counts_by_label"] = {r["lbl"]: r["c"] for r in rows}

        rows2 = session.run(
            "MATCH (n) WHERE n.allowed_roles IS NOT NULL "
            "RETURN labels(n)[0] AS lbl, count(n) AS c "
            "ORDER BY c DESC"
        )
        out["counts_any_lab"] = {r["lbl"]: r["c"] for r in rows2}

        # 2) Sample VanBan + linked DieuKhoan
        #    Traverse via document_id (works whether or not buoi_14 CONTAINS
        #    edges were loaded into the DB).
        sample = session.run(
            "MATCH (vb:VanBan {lab_session:$lab}) "
            "OPTIONAL MATCH (dk:DieuKhoan {lab_session:$lab}) "
            "WHERE dk.document_id = vb.id "
            "WITH vb, count(dk) AS n_dk "
            "ORDER BY n_dk DESC, rand() LIMIT 1 "
            "MATCH (dk:DieuKhoan {document_id: vb.id, lab_session:$lab}) "
            "RETURN vb.id AS vb_id, vb.title AS vb_title, "
            "       vb.allowed_roles AS vb_roles, "
            "       vb.security_label AS vb_label, "
            "       collect({"
            "           id: dk.id, "
            "           roles: dk.allowed_roles, "
            "           label: dk.security_label, "
            "           preview: left(dk.text, 80)"
            "       })[..5] AS dks",
            lab=LAB,
        ).single()

        if sample:
            out["sample"] = {
                "vb_id": sample["vb_id"],
                "vb_title": sample["vb_title"],
                "vb_roles": sample["vb_roles"],
                "vb_label": sample["vb_label"],
                "dks": list(sample["dks"]),
            }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not INPUT.exists():
        raise SystemExit(f"❌ Missing input: {INPUT}")

    print(f"📂 Reading {INPUT.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(INPUT)
    print(f"   {len(df):,} chunks ({len(df.columns)} columns)")

    vb_payloads, dk_payloads = build_payloads(df)
    print(f"   VanBan payloads    : {len(vb_payloads)} (INTERSECTION aggregation)")
    print(f"   DieuKhoan payloads : {len(dk_payloads)}")

    # Always emit Cypher + JSON (for traceability / browser fallback)
    emit_cypher(vb_payloads, dk_payloads)
    print(f"📄 Wrote {CYPHER_OUT.relative_to(PROJECT_ROOT)}")
    print(f"📄 Wrote {PARAMS_OUT.relative_to(PROJECT_ROOT)}")

    cfg = get_neo4j_config()
    if not (cfg["uri"] and cfg["password"]):
        print("⏭ SKIP Neo4j push — missing NEO4J_URI or NEO4J_PASSWORD in .env")
        print("   → Edit buoi_15/.env, then re-run this script.")
        return 0

    from neo4j import GraphDatabase

    print(f"🔌 Connecting to {cfg['uri']} (db=`{cfg['database']}`) ...")
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    try:
        driver.verify_connectivity()
        push_to_neo4j(driver, cfg["database"], vb_payloads, dk_payloads)
        print("   ✓ Push OK")

        stats = verify(driver, cfg["database"])
        print()
        print("🔍 Verification — nodes with allowed_roles (lab='buoi_15'):")
        for lbl, c in sorted(stats["counts_by_label"].items(), key=lambda x: -x[1]):
            print(f"   {lbl:<14} {c:>6,}")
        print(f"   {'TOTAL':<14} {sum(stats['counts_by_label'].values()):>6,}")

        print("\n🔍 Same count grouped by ANY lab_session:")
        for lbl, c in sorted(stats["counts_any_lab"].items(), key=lambda x: -x[1]):
            print(f"   {lbl:<14} {c:>6,}")

        if "sample" in stats:
            s = stats["sample"]
            print()
            print(f"🔍 Sample VanBan (random pick):")
            print(f"   id    : {s['vb_id']}")
            print(f"   title : {s['vb_title']}")
            print(f"   roles : {s['vb_roles']}")
            print(f"   label : {s['vb_label']}")
            print(f"   linked DieuKhoan: {len(s['dks'])}")
            for dk in s["dks"][:3]:
                if dk["id"] is None:
                    continue
                print(
                    f"     - id={dk['id'][:8]}…  "
                    f"label={dk['label']:<8} "
                    f"roles={dk['roles']}"
                )
                print(f"       preview: {dk['preview']!r}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
