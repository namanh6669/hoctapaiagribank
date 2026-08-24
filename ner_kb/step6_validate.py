"""BƯỚC 6: validate relationship.

Input:
  - ner_kb/relationships_raw.csv
  - ner_kb/cleaned_documents.csv
  - ner_kb/entities.csv
Output:
  - ner_kb/relationships.csv         (đạt)
  - ner_kb/validation_report.csv     (toàn bộ kèm trạng thái + lý do)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
REL_RAW = BASE / "relationships_raw.csv"
DOC_CLEAN = BASE / "cleaned_documents.csv"
ENT = BASE / "entities.csv"
OUT_VALID = BASE / "relationships.csv"
OUT_REPORT = BASE / "validation_report.csv"

# Allowed relationship types
DOC_DOC_TYPES = {"THAM_CHIEU", "SUA_DOI_BO_SUNG", "THAY_THE_BOI"}
DOC_ENT_TYPES = {"BAN_HANH_BOI", "KY_BOI", "AP_DUNG_CHO", "THUOC_LINH_VUC"}
ALLOWED_TYPES = DOC_DOC_TYPES | DOC_ENT_TYPES


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 6 — VALIDATE RELATIONSHIPS")

    rels = pd.read_csv(REL_RAW, dtype=str, keep_default_na=False)
    docs = pd.read_csv(DOC_CLEAN, dtype=str, keep_default_na=False)
    ents = pd.read_csv(ENT, dtype=str, keep_default_na=False)

    print(f"relationships_raw.csv : {len(rels)} rows")
    print(f"cleaned_documents.csv : {len(docs)} rows")
    print(f"entities.csv          : {len(ents)} rows")

    # ---- Build lookups ----
    # Document có 2 dạng id hợp lệ: numeric id HOẶC so_ky_hieu.
    doc_id_set = set(docs["id"].tolist())
    doc_so_set = set(docs["so_ky_hieu"].tolist())
    doc_valid_ids = doc_id_set | doc_so_set

    # Entity chỉ chấp nhận entity_id (E0001, ...)
    entity_id_set = set(ents["entity_id"].tolist())

    def is_valid_doc(value: str) -> bool:
        return value in doc_valid_ids

    def is_valid_entity(value: str) -> bool:
        return value in entity_id_set

    # ---- Validate từng relation ----
    statuses: list[str] = []
    reasons: list[str] = []

    for _, r in rels.iterrows():
        fails: list[str] = []

        # 1) source_id / target_id không rỗng
        src = str(r["source_id"]).strip()
        tgt = str(r["target_id"]).strip()
        if not src:
            fails.append("missing_source_id")
        if not tgt:
            fails.append("missing_target_id")

        # 2) relationship_type hợp lệ
        rt = str(r["relationship_type"]).strip()
        if rt not in ALLOWED_TYPES:
            fails.append(f"invalid_relationship_type:{rt}")

        # 3) source_kind / target_kind hợp lệ theo schema
        sk = str(r["source_kind"]).strip()
        tk = str(r["target_kind"]).strip()
        expected_sk = "Document"
        if sk != expected_sk:
            fails.append(f"invalid_source_kind:{sk}")
        if tk not in {"Document", "Entity"}:
            fails.append(f"invalid_target_kind:{tk}")

        # 4) relationship_type ↔ (source_kind,target_kind) phù hợp
        if not fails:  # chỉ check khi type đã hợp lệ
            if tk == "Document" and rt not in DOC_DOC_TYPES:
                fails.append(f"type_mismatch:{rt}_not_doc_doc")
            if tk == "Entity" and rt not in DOC_ENT_TYPES:
                fails.append(f"type_mismatch:{rt}_not_doc_entity")

        # 5) source/target tồn tại trong dataset
        if sk == "Document":
            if not is_valid_doc(src):
                fails.append(f"unknown_document_source:{src}")
        if tk == "Document":
            if not is_valid_doc(tgt):
                fails.append(f"unknown_document_target:{tgt}")
        if tk == "Entity":
            if not is_valid_entity(tgt):
                fails.append(f"unknown_entity_target:{tgt}")

        # 6) self-loop
        if src and tgt and src == tgt:
            fails.append("self_loop")

        # 7) missing evidence
        ev = str(r["evidence"]).strip()
        if not ev:
            fails.append("missing_evidence")

        if not fails:
            statuses.append("PASS")
            reasons.append("")
        else:
            statuses.append("FAIL")
            reasons.append("; ".join(fails))

    rels["validation_status"] = statuses
    rels["validation_reason"] = reasons

    # ---- Dedupe (sau khi validate) ----
    # Kiểm tra duplicate toàn cục: cùng (source_id, target_id, relationship_type)
    dup_mask = rels.duplicated(subset=["source_id", "target_id", "relationship_type"], keep="first")
    # Gắn cờ duplicate cho tất cả bản sao (kể cả bản giữ)
    dup_all = rels.duplicated(subset=["source_id", "target_id", "relationship_type"], keep=False)
    rels.loc[dup_all & ~dup_mask, "validation_status"] = "FAIL"
    rels.loc[dup_all & ~dup_mask, "validation_reason"] = (
        rels.loc[dup_all & ~dup_mask, "validation_reason"].astype(str).map(
            lambda s: (s + "; " if s else "") + "duplicate"
        )
    )
    # Sau đó loại trùng, chỉ giữ bản đầu
    rels = rels.drop_duplicates(subset=["source_id", "target_id", "relationship_type"], keep="first")
    # Reset index để số thứ tự an toàn
    rels = rels.reset_index(drop=True)

    # ---- Save outputs ----
    # relationships.csv: chỉ PASS
    pass_df = rels[rels["validation_status"] == "PASS"].drop(
        columns=["validation_status", "validation_reason"]
    ).reset_index(drop=True)
    pass_df.to_csv(OUT_VALID, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_VALID.name}: {len(pass_df)} PASS rows x {pass_df.shape[1]} cols")

    # validation_report.csv: ALL rows + status + reason
    rels.to_csv(OUT_REPORT, index=False, encoding="utf-8")
    print(f"Saved {OUT_REPORT.name}: {len(rels)} rows (PASS + FAIL) x {rels.shape[1]} cols")

    # ---- Report ----
    banner("STEP 6 REPORT")
    total = len(rels)
    n_pass = (rels["validation_status"] == "PASS").sum()
    n_fail = (rels["validation_status"] == "FAIL").sum()
    print(f"  Total raw relations : {total}")
    print(f"  PASS                : {n_pass}")
    print(f"  FAIL                : {n_fail}")

    print("\n  PASS by relationship_type:")
    for t, n in pass_df["relationship_type"].value_counts().items():
        print(f"    {t:25s} : {n}")

    # Nguyên nhân fail phổ biến
    fail_reasons = rels[rels["validation_status"] == "FAIL"]["validation_reason"]
    primary = Counter()
    for r in fail_reasons:
        for tag in str(r).split("; "):
            primary[tag] += 1
    print("\n  FAIL reasons (primary tag count):")
    for tag, n in primary.most_common():
        print(f"    {tag:50s} : {n}")

    # Toàn bộ FAIL rows
    fail_df = rels[rels["validation_status"] == "FAIL"]
    if len(fail_df) > 0:
        print(f"\n  All failing rows ({len(fail_df)}):")
        for _, r in fail_df.iterrows():
            print(f"    [{r['relationship_type']}] {r['source_kind']} {r['source_id']} -> "
                  f"{r['target_kind']} {r['target_id']}  :: {r['validation_reason']}")

    # 10 PASS mẫu
    banner("10 SAMPLE PASS RELATIONS")
    sample = pass_df.head(10)
    for i, r in sample.iterrows():
        print(f"\n  [{i+1}] {r['source_kind']} {r['source_id']} --[{r['relationship_type']}]--> "
              f"{r['target_kind']} {r['target_id']}")
        print(f"      method={r['method']}, confidence={r['confidence']}")
        print(f"      evidence: {str(r['evidence'])[:160]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
