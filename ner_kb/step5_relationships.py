"""BƯỚC 5: Relationship Extraction.

Input:
  - ner_kb/cleaned_documents.csv
  - ner_kb/relation_candidates.csv
  - ner_kb/entities.csv
  - ner_kb/enriched_metadata.csv
Output:
  - ner_kb/relationships_raw.csv

KHÔNG import Neo4j.
KHÔNG dùng ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DOC_CLEAN = BASE / "cleaned_documents.csv"
CAND = BASE / "relation_candidates.csv"
ENT = BASE / "entities.csv"
META = BASE / "enriched_metadata.csv"
OUTPUT = BASE / "relationships_raw.csv"

# Mapping trigger phrase -> (relationship_type, direction_flip)
# direction_flip=True nghĩa là target (in candidate) -> source (in candidate) trong relation
# (để có chiều "Document cũ -> Document mới" cho THAY_THE_BOI).
TRIGGER_MAP: dict[str, tuple[str, bool]] = {
    "Căn cứ":              ("THAM_CHIEU",       False),
    "Sửa đổi, bổ sung":    ("SUA_DOI_BO_SUNG",  False),
    "bãi bỏ":              ("THAY_THE_BOI",      True),
    "thay thế":            ("THAY_THE_BOI",      True),
}


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 5 — RELATIONSHIP EXTRACTION")

    # --- Load ---
    docs = pd.read_csv(DOC_CLEAN, dtype=str, keep_default_na=False)
    cands = pd.read_csv(CAND, dtype=str, keep_default_na=False)
    ents = pd.read_csv(ENT, dtype=str, keep_default_na=False)
    meta = pd.read_csv(META, dtype=str, keep_default_na=False)

    print(f"cleaned_documents.csv : {len(docs)} rows")
    print(f"relation_candidates.csv: {len(cands)} rows")
    print(f"entities.csv          : {len(ents)} rows")
    print(f"enriched_metadata.csv : {len(meta)} rows")

    # Build lookup: id -> so_ky_hieu, so_ky_hieu -> id
    id_to_so = dict(zip(docs["id"], docs["so_ky_hieu"]))
    so_to_id = dict(zip(docs["so_ky_hieu"], docs["id"]))

    # Build entity lookup: (entity_type, canonical_name lower) -> entity_id
    ent_lookup: dict[tuple[str, str], str] = {}
    for _, r in ents.iterrows():
        ent_lookup[(r["entity_type"], r["canonical_name"].lower())] = r["entity_id"]

    # -------- 1. Document -> Document relations --------
    banner("[A] Document -> Document relations")

    doc_doc_rels: list[dict] = []
    skipped_trigger: list[dict] = []

    for _, c in cands.iterrows():
        source_id = c["source_id"]
        source_so = c["source_so_ky_hieu"]
        target_so = c["target_so_ky_hieu"]
        trigger = c["trigger"]
        evidence = c["evidence"]

        if not evidence or not evidence.strip():
            continue  # spec: no evidence -> no relation

        if trigger not in TRIGGER_MAP:
            skipped_trigger.append({"trigger": trigger, "row": c.to_dict()})
            continue

        rel_type, flip = TRIGGER_MAP[trigger]

        if flip:
            # THAY_THE_BOI: old doc (target) -> new doc (source)
            src_id, src_so = target_so, target_so
            tgt_id, tgt_so = source_id, source_so
        else:
            src_id, src_so = source_id, source_so
            tgt_id, tgt_so = target_so, target_so

        # Confidence: rule-based 0.9
        doc_doc_rels.append({
            "source_kind": "Document",
            "source_id": src_id,
            "source_name": src_so,
            "target_kind": "Document",
            "target_id": tgt_id,
            "target_name": tgt_so,
            "relationship_type": rel_type,
            "method": "rule",
            "confidence": 0.9,
            "evidence": evidence.strip(),
        })

    print(f"  Produced {len(doc_doc_rels)} doc->doc relations (rule-based, no Gemini needed)")
    print(f"  Skipped {len(skipped_trigger)} candidates with unknown trigger")

    # -------- 2. Document -> Entity relations --------
    banner("[B] Document -> Entity relations")

    # Field mapping in enriched_metadata.csv (enriched_X is the resolved value)
    META_TO_ENTITY_TYPE = {
        "co_quan_ban_hanh": "CoQuan",
        "nguoi_ky": "NguoiKy",
        "linh_vuc": "LinhVuc",
        "doi_tuong_ap_dung": "DoiTuongApDung",
    }
    META_TO_REL_TYPE = {
        "co_quan_ban_hanh": "BAN_HANH_BOI",
        "nguoi_ky": "KY_BOI",
        "linh_vuc": "THUOC_LINH_VUC",
        "doi_tuong_ap_dung": "AP_DUNG_CHO",
    }

    doc_ent_rels: list[dict] = []
    unresolved: list[dict] = []

    for _, row in meta.iterrows():
        doc_id = row["id"]
        doc_so = row["so_ky_hieu"]
        for meta_field, etype in META_TO_ENTITY_TYPE.items():
            enriched_col = f"enriched_{meta_field}"
            method_col = f"method_{meta_field}"
            if enriched_col not in row:
                continue
            value = str(row[enriched_col]).strip()
            if not value:
                continue  # spec: no evidence -> no relation
            method = str(row.get(method_col, "metadata"))
            confidence = 1.0 if method == "metadata" else 0.75

            # Split comma-separated lists (Gemini sometimes returns multiple subjects)
            values = [v.strip() for v in value.split(",") if v.strip()]

            for v in values:
                ent_id = ent_lookup.get((etype, v.lower()))
                if ent_id is None:
                    unresolved.append({
                        "doc_id": doc_id, "doc_so": doc_so,
                        "entity_type": etype, "value": v,
                    })
                    continue
                doc_ent_rels.append({
                    "source_kind": "Document",
                    "source_id": doc_id,
                    "source_name": doc_so,
                    "target_kind": "Entity",
                    "target_id": ent_id,
                    "target_name": v,
                    "relationship_type": META_TO_REL_TYPE[meta_field],
                    "method": method,
                    "confidence": confidence,
                    "evidence": f"enriched_metadata.{meta_field} = {v!r} (id={doc_id})",
                })

    print(f"  Produced {len(doc_ent_rels)} doc->entity relations")
    print(f"  Unresolved entity values (not in entities.csv): {len(unresolved)}")

    # -------- 3. Combine + dedupe --------
    all_rels = doc_doc_rels + doc_ent_rels
    rels_df = pd.DataFrame(
        all_rels,
        columns=[
            "source_kind", "source_id", "source_name",
            "target_kind", "target_id", "target_name",
            "relationship_type", "method", "confidence", "evidence",
        ],
    )
    # Dedupe by (source_id, target_id, relationship_type)
    before = len(rels_df)
    rels_df = rels_df.drop_duplicates(
        subset=["source_id", "target_id", "relationship_type"]
    ).reset_index(drop=True)
    after = len(rels_df)
    print(f"\nDedupe: {before} -> {after} (removed {before - after})")

    # -------- 4. Save --------
    rels_df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Saved {OUTPUT}: {len(rels_df)} rows x {rels_df.shape[1]} cols")

    # -------- 5. Report --------
    banner("STEP 5 REPORT")
    print("\n  Relations by type:")
    for t, n in rels_df["relationship_type"].value_counts().items():
        print(f"    {t:25s} : {n}")
    print()
    print("  Relations by method:")
    for m, n in rels_df["method"].value_counts().items():
        print(f"    {m:25s} : {n}")
    print()
    print("  Relations by source_kind -> target_kind:")
    for (s, t), n in rels_df.groupby(["source_kind", "target_kind"]).size().items():
        print(f"    {s} -> {t}: {n}")

    # Direction check for THAY_THE_BOI
    banner("DIRECTION CHECK (THAY_THE_BOI)")
    thay = rels_df[rels_df["relationship_type"] == "THAY_THE_BOI"]
    if len(thay) > 0:
        print(f"  THAY_THE_BOI count: {len(thay)}")
        # Each row: source_id is the OLD doc (target_so_ky_hieu from candidates),
        # target_id is the NEW doc (source_id from candidates)
        for _, r in thay.head(5).iterrows():
            print(f"    OLD {r['source_id']} ({r['source_name']}) -> NEW {r['target_id']} ({r['target_name']})")
    else:
        print("  (no THAY_THE_BOI rows)")

    # Sample 10 of each
    banner("10 SAMPLE RELATIONS")
    sample = rels_df.head(10)
    for i, r in sample.iterrows():
        print(f"\n  [{i+1}] {r['source_kind']} {r['source_name']} --[{r['relationship_type']}]--> {r['target_kind']} {r['target_name']}")
        print(f"      method={r['method']}, confidence={r['confidence']:.2f}")
        print(f"      evidence: {r['evidence'][:160]}")

    if unresolved:
        print(f"\n  UNRESOLVED entity values (first 10):")
        for u in unresolved[:10]:
            print(f"    - {u['doc_so']} ({u['doc_id']}): {u['entity_type']} = {u['value']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
