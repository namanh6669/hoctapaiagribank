"""BƯỚC 8: import Knowledge Graph vào Neo4j.

Input:
  - ner_kb/cleaned_documents.csv   (Document nodes)
  - ner_kb/entities.csv            (Entity nodes)
  - ner_kb/relationships.csv       (relations)

Idempotent: re-run không tăng node/edge count.
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
OUT_ERR = BASE / "import_errors.csv"

# Labels & relationship types
DOC_LABEL = "Document"
ENTITY_LABELS = ["CoQuan", "NguoiKy", "DoiTuongApDung", "LinhVuc"]
REL_TYPES = [
    "THAM_CHIEU", "SUA_DOI_BO_SUNG", "THAY_THE_BOI",
    "BAN_HANH_BOI", "KY_BOI", "AP_DUNG_CHO", "THUOC_LINH_VUC",
]

# Các cột Document giữ làm properties
DOC_PROPS = [
    "id", "title", "so_ky_hieu", "ngay_ban_hanh", "loai_van_ban",
    "ngay_co_hieu_luc", "ngay_het_hieu_luc", "nguon_thu_thap",
    "ngay_dang_cong_bao", "nganh", "linh_vuc", "co_quan_ban_hanh",
    "chuc_danh", "nguoi_ky", "pham_vi", "thong_tin_ap_dung",
    "tinh_trang_hieu_luc",
]


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 8 — IMPORT KNOWLEDGE GRAPH TO NEO4J")

    # 1) Load .env
    env_path = BASE / ".env"
    if not env_path.exists():
        print(f"FAIL: .env not found at {env_path}")
        return 1
    load_dotenv(env_path, verbose=False)

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    print(f"  Config loaded: {uri} (db={database})")
    print(f"  NEO4J_PASSWORD: <set, {len(password)} chars>")

    # 2) Open driver
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    print(f"  Driver opened: {uri}")

    try:
        with driver.session(database=database) as session:
            # 3) Create uniqueness constraints
            print("\n  Creating uniqueness constraints...")
            constraints = [
                f"CREATE CONSTRAINT doc_so_unique IF NOT EXISTS "
                f"FOR (d:{DOC_LABEL}) REQUIRE d.so_ky_hieu IS UNIQUE",
                f"CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
                f"FOR (d:{DOC_LABEL}) REQUIRE d.id IS UNIQUE",
            ]
            for etype in ENTITY_LABELS:
                constraints.append(
                    f"CREATE CONSTRAINT {etype.lower()}_eid_unique IF NOT EXISTS "
                    f"FOR (n:{etype}) REQUIRE n.entity_id IS UNIQUE"
                )
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    print(f"    constraint note: {type(e).__name__}: {e}")
            print(f"    {len(constraints)} constraints ensured")

            # 4) Load CSVs
            docs = pd.read_csv(DOC_CLEAN, dtype=str, keep_default_na=False)
            ents = pd.read_csv(ENT, dtype=str, keep_default_na=False)
            rels = pd.read_csv(REL, dtype=str, keep_default_na=False)
            print(f"\n  Loaded: {len(docs)} docs, {len(ents)} entities, {len(rels)} relations")

            # Build doc id -> so_ky_hieu lookup
            id_to_so = dict(zip(docs["id"], docs["so_ky_hieu"]))
            so_to_id = dict(zip(docs["so_ky_hieu"], docs["id"]))

            # 5) MERGE Document nodes
            print("\n  Importing Document nodes...")
            doc_count = 0
            for _, r in docs.iterrows():
                props = {k: (r[k] if k in r else "") for k in DOC_PROPS}
                # Build Cypher params (escape None)
                params = {k: ("" if pd.isna(v) else str(v)) for k, v in props.items()}
                cypher = (
                    f"MERGE (d:{DOC_LABEL} {{so_ky_hieu: $so_ky_hieu}}) "
                    f"ON CREATE SET d += $props "
                    f"ON MATCH SET d += $props"
                )
                session.run(cypher, {
                    "so_ky_hieu": params["so_ky_hieu"],
                    "props": params,
                })
                doc_count += 1
            print(f"    {doc_count} Document nodes merged")

            # 6) MERGE Entity nodes (mỗi entity dùng đúng label của nó)
            print("\n  Importing Entity nodes...")
            ent_count = 0
            for _, r in ents.iterrows():
                etype = r["entity_type"]
                if etype not in ENTITY_LABELS:
                    continue
                cypher = (
                    f"MERGE (n:{etype} {{entity_id: $entity_id}}) "
                    f"ON CREATE SET n += $props "
                    f"ON MATCH SET n += $props"
                )
                props = {
                    "entity_id": r["entity_id"],
                    "canonical_name": r["canonical_name"],
                    "entity_type": etype,
                    "occurrences": int(r["occurrences"]) if str(r["occurrences"]).isdigit() else 0,
                    "source_methods": r["source_methods"],
                    "first_seen_doc": r["first_seen_doc"],
                    "original_names": r["original_names"],
                }
                session.run(cypher, {"entity_id": r["entity_id"], "props": props})
                ent_count += 1
            print(f"    {ent_count} Entity nodes merged")

            # 7) Build relationships
            print("\n  Importing relationships...")
            errs: list[dict] = []
            rel_count = 0
            rel_count_by_type: dict[str, int] = {}

            for _, r in rels.iterrows():
                rt = r["relationship_type"]
                if rt not in REL_TYPES:
                    errs.append({
                        "relationship_type": rt,
                        "source_id": r["source_id"],
                        "target_id": r["target_id"],
                        "source_name": r["source_name"],
                        "target_name": r["target_name"],
                        "reason": f"unknown_relationship_type:{rt}",
                    })
                    continue

                # Resolve source
                src_kind = r["source_kind"]
                src_id = r["source_id"]
                src_label = None
                if src_kind == "Document":
                    if src_id in id_to_so:
                        src_so = id_to_so[src_id]
                        src_label = DOC_LABEL
                    elif src_id in so_to_id:
                        src_so = src_id
                        src_label = DOC_LABEL
                    else:
                        errs.append({
                            "relationship_type": rt,
                            "source_id": src_id,
                            "target_id": r["target_id"],
                            "source_name": r["source_name"],
                            "target_name": r["target_name"],
                            "reason": f"source_document_not_found:{src_id}",
                        })
                        continue

                # Resolve target
                tgt_kind = r["target_kind"]
                tgt_id = r["target_id"]
                tgt_label = None
                if tgt_kind == "Document":
                    if tgt_id in id_to_so:
                        tgt_so = id_to_so[tgt_id]
                        tgt_label = DOC_LABEL
                    elif tgt_id in so_to_id:
                        tgt_so = tgt_id
                        tgt_label = DOC_LABEL
                    else:
                        errs.append({
                            "relationship_type": rt,
                            "source_id": src_id,
                            "target_id": tgt_id,
                            "source_name": r["source_name"],
                            "target_name": r["target_name"],
                            "reason": f"target_document_not_found:{tgt_id}",
                        })
                        continue
                elif tgt_kind == "Entity":
                    tgt_label = rt_to_entity_label(rt)
                    if tgt_label is None:
                        errs.append({
                            "relationship_type": rt,
                            "source_id": src_id,
                            "target_id": tgt_id,
                            "source_name": r["source_name"],
                            "target_name": r["target_name"],
                            "reason": f"cannot_infer_entity_label_for_type:{rt}",
                        })
                        continue
                    # Verify entity exists
                    res = session.run(
                        f"MATCH (n:{tgt_label} {{entity_id: $eid}}) RETURN n LIMIT 1",
                        eid=tgt_id,
                    )
                    if res.single() is None:
                        errs.append({
                            "relationship_type": rt,
                            "source_id": src_id,
                            "target_id": tgt_id,
                            "source_name": r["source_name"],
                            "target_name": r["target_name"],
                            "reason": f"target_entity_not_found:{tgt_id}",
                        })
                        continue

                # MERGE relationship
                try:
                    conf_val = float(r["confidence"]) if r["confidence"] else 0.0
                except ValueError:
                    conf_val = 0.0
                rel_props = {
                    "method": r["method"],
                    "confidence": conf_val,
                    "evidence": r["evidence"],
                }

                if src_label == DOC_LABEL and tgt_label == DOC_LABEL:
                    cypher = (
                        f"MATCH (a:{DOC_LABEL} {{so_ky_hieu: $src_so}}) "
                        f"MATCH (b:{DOC_LABEL} {{so_ky_hieu: $tgt_so}}) "
                        f"MERGE (a)-[rel:{rt}]->(b) "
                        f"ON CREATE SET rel += $props "
                        f"ON MATCH SET rel += $props"
                    )
                    session.run(cypher, {
                        "src_so": src_so,
                        "tgt_so": tgt_so,
                        "props": rel_props,
                    })
                elif src_label == DOC_LABEL and tgt_label in ENTITY_LABELS:
                    cypher = (
                        f"MATCH (a:{DOC_LABEL} {{so_ky_hieu: $src_so}}) "
                        f"MATCH (b:{tgt_label} {{entity_id: $tgt_eid}}) "
                        f"MERGE (a)-[rel:{rt}]->(b) "
                        f"ON CREATE SET rel += $props "
                        f"ON MATCH SET rel += $props"
                    )
                    session.run(cypher, {
                        "src_so": src_so,
                        "tgt_eid": tgt_id,
                        "props": rel_props,
                    })
                else:
                    errs.append({
                        "relationship_type": rt,
                        "source_id": src_id,
                        "target_id": tgt_id,
                        "source_name": r["source_name"],
                        "target_name": r["target_name"],
                        "reason": f"unsupported_endpoint_combo:{src_label}->{tgt_label}",
                    })
                    continue

                rel_count += 1
                rel_count_by_type[rt] = rel_count_by_type.get(rt, 0) + 1

            print(f"    {rel_count} relationships merged")
            print(f"    {len(errs)} import errors")

            # 8) Save errors
            if errs:
                err_df = pd.DataFrame(errs)
                err_df.to_csv(OUT_ERR, index=False, encoding="utf-8")
                print(f"    Saved errors to {OUT_ERR.name}")

            # 9) Verify counts
            print("\n  Database counts after import:")
            for label in [DOC_LABEL] + ENTITY_LABELS:
                cnt = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                print(f"    {label:20s} : {cnt} nodes")
            for rt in REL_TYPES:
                cnt = session.run(f"MATCH ()-[r:{rt}]->() RETURN count(r) AS c").single()["c"]
                print(f"    {rt:25s} : {cnt} relationships")

        driver.close()
        print("\n  Driver closed cleanly")
        print("  OVERALL: PASS")
        return 0
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        try:
            driver.close()
        except Exception:
            pass
        return 1


def rt_to_entity_label(rt: str) -> str | None:
    """Map relationship type (Doc->Entity) sang entity label."""
    return {
        "BAN_HANH_BOI": "CoQuan",
        "KY_BOI": "NguoiKy",
        "THUOC_LINH_VUC": "LinhVuc",
        "AP_DUNG_CHO": "DoiTuongApDung",
    }.get(rt)


if __name__ == "__main__":
    sys.exit(main())
