"""BƯỚC 9: kiểm tra Knowledge Graph sau import.

- Query counts từ Neo4j
- Đối chiếu với CSV trước import
- Sample relations
- KHÔNG sửa dữ liệu
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
DOC_CLEAN = BASE / "cleaned_documents.csv"
ENT = BASE / "entities.csv"
REL = BASE / "relationships.csv"

LABELS = ["Document", "CoQuan", "NguoiKy", "DoiTuongApDung", "LinhVuc"]
REL_TYPES = [
    "THAM_CHIEU", "SUA_DOI_BO_SUNG", "THAY_THE_BOI",
    "BAN_HANH_BOI", "KY_BOI", "AP_DUNG_CHO", "THUOC_LINH_VUC",
]


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 9 — VERIFY KNOWLEDGE GRAPH POST-IMPORT")

    # 1) Load .env
    env_path = BASE / ".env"
    load_dotenv(env_path, verbose=False)
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    print(f"  Connected: {uri} (db={database})")

    # 2) Load CSV for cross-check
    docs = pd.read_csv(DOC_CLEAN, dtype=str, keep_default_na=False)
    ents = pd.read_csv(ENT, dtype=str, keep_default_na=False)
    rels = pd.read_csv(REL, dtype=str, keep_default_na=False)
    print(f"\n  CSV inputs:")
    print(f"    cleaned_documents.csv : {len(docs)} rows")
    print(f"    entities.csv          : {len(ents)} rows")
    print(f"    relationships.csv     : {len(rels)} rows")

    # 3) Expected counts from CSV
    expected_doc = len(docs)
    expected_ent_by_type = ents["entity_type"].value_counts().to_dict()
    expected_rel_by_type = rels["relationship_type"].value_counts().to_dict()

    # 4) Open driver
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database=database) as session:
            # 1) Node counts
            banner("[1] NODE COUNT BY LABEL")
            actual_nodes: dict[str, int] = {}
            for label in LABELS:
                cnt = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                actual_nodes[label] = cnt
                expected = expected_doc if label == "Document" else expected_ent_by_type.get(label, 0)
                delta = cnt - expected
                marker = "✓" if delta == 0 or (label == "Document" and delta <= 5) else "?"
                print(f"  {label:20s} actual={cnt:>4d}   expected={expected:>4d}   Δ={delta:+d}  {marker}")

            # 2) Relationship counts
            banner("[2] RELATIONSHIP COUNT BY TYPE")
            actual_rels: dict[str, int] = {}
            for rt in REL_TYPES:
                cnt = session.run(f"MATCH ()-[r:{rt}]->() RETURN count(r) AS c").single()["c"]
                actual_rels[rt] = cnt
                expected = expected_rel_by_type.get(rt, 0)
                delta = cnt - expected
                marker = "✓" if delta == 0 else "?"
                print(f"  {rt:20s} actual={cnt:>4d}   expected={expected:>4d}   Δ={delta:+d}  {marker}")

            # 3) Document -> NguoiKy samples
            banner("[3] SAMPLE: Document -> NguoiKy (KY_BOI)")
            rows = session.run(
                """
                MATCH (d:Document)-[r:KY_BOI]->(p:NguoiKy)
                RETURN d.so_ky_hieu AS doc, p.canonical_name AS nguoi_ky,
                       r.method AS method, r.confidence AS conf
                ORDER BY d.so_ky_hieu
                LIMIT 10
                """
            )
            samples_dn = list(rows)
            print(f"  Total KY_BOI: {actual_rels.get('KY_BOI', 0)}  (CSV expects: {expected_rel_by_type.get('KY_BOI', 0)})")
            for i, rec in enumerate(samples_dn, 1):
                print(f"    [{i}] {rec['doc']:20s} --[KY_BOI]--> {rec['nguoi_ky']}  "
                      f"(method={rec['method']}, conf={rec['conf']})")

            # 4) Document -> DoiTuongApDung samples
            banner("[4] SAMPLE: Document -> DoiTuongApDung (AP_DUNG_CHO)")
            rows = session.run(
                """
                MATCH (d:Document)-[r:AP_DUNG_CHO]->(t:DoiTuongApDung)
                RETURN d.so_ky_hieu AS doc, t.canonical_name AS doi_tuong,
                       r.method AS method, r.confidence AS conf
                ORDER BY d.so_ky_hieu, t.canonical_name
                LIMIT 10
                """
            )
            samples_dt = list(rows)
            print(f"  Total AP_DUNG_CHO: {actual_rels.get('AP_DUNG_CHO', 0)}  "
                  f"(CSV expects: {expected_rel_by_type.get('AP_DUNG_CHO', 0)})")
            for i, rec in enumerate(samples_dt, 1):
                print(f"    [{i}] {rec['doc']:20s} --[AP_DUNG_CHO]--> {rec['doi_tuong']}  "
                      f"(method={rec['method']}, conf={rec['conf']})")

            # 5) Document -> Document samples
            banner("[5] SAMPLE: Document -> Document (THAM_CHIEU / SUA_DOI_BO_SUNG / THAY_THE_BOI)")
            for rt in ["THAM_CHIEU", "SUA_DOI_BO_SUNG", "THAY_THE_BOI"]:
                rows = session.run(
                    f"""
                    MATCH (a:Document)-[r:{rt}]->(b:Document)
                    RETURN a.so_ky_hieu AS src, b.so_ky_hieu AS tgt,
                           r.method AS method, r.confidence AS conf
                    ORDER BY a.so_ky_hieu, b.so_ky_hieu
                    LIMIT 5
                    """
                )
                samples = list(rows)
                print(f"\n  {rt} (actual={actual_rels.get(rt, 0)}, "
                      f"expected={expected_rel_by_type.get(rt, 0)}):")
                if not samples:
                    print(f"    (no rows)")
                for i, rec in enumerate(samples, 1):
                    arrow = "-->"
                    print(f"    [{i}] {rec['src']:20s} {arrow} {rec['tgt']:20s}  "
                          f"(method={rec['method']}, conf={rec['conf']})")

            # 6) Cross-check totals
            banner("[6] CROSS-CHECK SUMMARY")
            issues = []
            for label in LABELS:
                cnt = actual_nodes[label]
                expected = expected_doc if label == "Document" else expected_ent_by_type.get(label, 0)
                if label == "Document":
                    if cnt < expected:
                        issues.append(f"Document: actual={cnt} < expected={expected}")
                else:
                    if cnt != expected:
                        issues.append(f"{label}: actual={cnt} != expected={expected}")
            for rt in REL_TYPES:
                cnt = actual_rels.get(rt, 0)
                expected = expected_rel_by_type.get(rt, 0)
                if cnt != expected:
                    issues.append(f"{rt}: actual={cnt} != expected={expected}")

            if not issues:
                print("  No discrepancies found.")
            else:
                print("  Discrepancies:")
                for s in issues:
                    print(f"    - {s}")

            # Giải thích Document count chênh
            if actual_nodes["Document"] > expected_doc:
                extra = actual_nodes["Document"] - expected_doc
                print(f"\n  Note: Document count = {actual_nodes['Document']} (Expected {expected_doc}).")
                print(f"        Extra {extra} node(s) từ session trước (so_ky_hieu không thuộc 30 docs hiện tại).")
                # In các so_ky_hieu nằm ngoài current batch
                current_sos = set(docs["so_ky_hieu"])
                extra_rows = session.run(
                    "MATCH (d:Document) WHERE NOT d.so_ky_hieu IN $sos "
                    "RETURN d.so_ky_hieu AS so, d.title AS title",
                    sos=list(current_sos),
                )
                extras = list(extra_rows)
                for r in extras:
                    print(f"        - old: {r['so']} (title={r['title'][:60] if r['title'] else ''!r})")

        driver.close()
        print("\n  Driver closed cleanly")
        print("  OVERALL: PASS" if not issues else "  OVERALL: HAS_DISCREPANCIES (review above)")
        return 0
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        try:
            driver.close()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
