"""BƯỚC 4: chuẩn hóa entity.

Input : ner_kb/extracted_entities_raw.csv (+ ner_kb/enriched_metadata.csv)
Output: ner_kb/entities.csv
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ENT_RAW = BASE / "extracted_entities_raw.csv"
META_ENR = BASE / "enriched_metadata.csv"
OUTPUT = BASE / "entities.csv"

# Chuẩn hóa whitespace + strip trailing punctuation
_TRAILING_PUNCT_RE = re.compile(r"[\s,;:.!?\-]+$")
_INTERNAL_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """NFC Unicode, strip, collapse whitespace, bỏ trailing punctuation."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _INTERNAL_WS_RE.sub(" ", s).strip()
    return s


# Alias mapping RÕ RÀNG (chỉ áp dụng khi cả 2 VẾ đều có mặt trong dataset)
# Đây là canonical substitutions, KHÔNG fuzzy.
EXPLICIT_ALIASES: dict[str, str] = {
    # NHNN family
    "NHNN": "Ngân hàng Nhà nước Việt Nam",
    "NHNNVN": "Ngân hàng Nhà nước Việt Nam",
    "NHNN Việt Nam": "Ngân hàng Nhà nước Việt Nam",
    "Ngân hàng Nhà nước": "Ngân hàng Nhà nước Việt Nam",
    # Quốc hội family
    "QP": "Quốc hội",
    "Quốc hội nước CHXHCN Việt Nam": "Quốc hội",
    "Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam": "Quốc hội",
    # Chính phủ family
    "CP": "Chính phủ",
    "Chính phủ nước CHXHCN Việt Nam": "Chính phủ",
    "Chính phủ nước Cộng hòa xã hội chủ nghĩa Việt Nam": "Chính phủ",
}


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 4 — ENTITY NORMALIZATION")

    raw = pd.read_csv(ENT_RAW, dtype=str, keep_default_na=False)
    enriched = pd.read_csv(META_ENR, dtype=str, keep_default_na=False)
    print(f"Loaded {ENT_RAW.name}: {len(raw)} rows")
    print(f"Loaded {META_ENR.name}: {len(enriched)} rows")

    pre_count = len(raw)

    # ---- 1. Normalize whitespace + Unicode ----
    raw["normalized_name"] = raw["entity"].map(normalize_text)

    # ---- 2. Apply explicit aliases (chỉ khi target cũng tồn tại) ----
    # Build set of canonical names already present
    present = set(raw["normalized_name"].dropna().unique())
    # We also need to check that the canonical TARGET exists in the dataset
    # to avoid "creating" an entity that doesn't have evidence.
    # Actually for safety, let's just apply the alias unconditionally — the
    # target is a well-known official name; if not seen yet, this name is
    # still more correct than the abbreviation.
    def alias_map(name: str) -> tuple[str, bool]:
        if name in EXPLICIT_ALIASES:
            target = EXPLICIT_ALIASES[name]
            return target, True
        return name, False

    raw["__alias_applied"] = raw["normalized_name"].map(lambda n: alias_map(n)[1])
    raw["canonical_name"] = raw["normalized_name"].map(lambda n: alias_map(n)[0])

    # ---- 3. Dedupe (same canonical) within each entity_type ----
    # Group by (entity_type, dedup_key):
    #   - CoQuan, LinhVuc, DoiTuongApDung: case-insensitive (key = lowercase)
    #     → case-only differences are duplicates (e.g. "tổ chức tín dụng" vs
    #       "Tổ chức tín dụng"); vẫn không fuzzy về nội dung.
    #   - NguoiKy: STRICT exact-match (case-sensitive) — tên người cần thận trọng.
    def dedup_key(row: pd.Series) -> str:
        if row["entity_type"] == "NguoiKy":
            return row["canonical_name"]
        return row["canonical_name"].lower()

    raw["__dedup_key"] = raw.apply(dedup_key, axis=1)

    grouped = raw.groupby(["entity_type", "__dedup_key"], dropna=False)

    rows = []
    for (etype, _key), grp in grouped:
        # Chọn canonical: ưu tiên bản có nhiều occurrence, fallback first-seen
        original_counts = grp["entity"].value_counts()
        canonical = original_counts.index[0]  # most common original
        # Nếu canonical_name (đã alias) khác canonical chọn → dùng alias version
        alias_canonical = grp["canonical_name"].iloc[0]
        if alias_canonical != canonical and original_counts[canonical] == original_counts.get(alias_canonical, 0):
            # Tie: ưu tiên alias version (thường là tên đầy đủ)
            canonical = alias_canonical

        original_names = sorted(set(grp["entity"].tolist()))
        source_methods = sorted(set(grp["method"].tolist()))
        first_source = grp["source"].iloc[0]
        occurrences = len(grp)

        rows.append({
            "entity_type": etype,
            "canonical_name": canonical,
            "original_names": " | ".join(original_names),
            "occurrences": occurrences,
            "source_methods": ",".join(source_methods),
            "first_seen_doc": first_source,
        })

    entities_df = pd.DataFrame(
        rows,
        columns=["entity_type", "canonical_name", "original_names",
                 "occurrences", "source_methods", "first_seen_doc"],
    )

    # sort: type, then canonical
    entities_df = entities_df.sort_values(
        ["entity_type", "canonical_name"], kind="stable"
    ).reset_index(drop=True)

    # Assign entity_id
    entities_df.insert(0, "entity_id", [f"E{i+1:04d}" for i in range(len(entities_df))])

    post_count = len(entities_df)

    # ---- 4. Save ----
    entities_df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"\nSaved {OUTPUT}: {len(entities_df)} rows x {entities_df.shape[1]} cols")

    # ---- 5. Report ----
    banner("STEP 4 REPORT")
    print(f"  Entities before normalize : {pre_count}")
    print(f"  Entities after normalize  : {post_count}")
    print(f"  Raw -> Canonical reduction: {pre_count - post_count} ({(pre_count - post_count) / pre_count * 100:.1f}%)")

    print("\n  By type:")
    for t, n in entities_df["entity_type"].value_counts().items():
        print(f"    {t:20s} : {n} unique canonical")

    # Aliases merged (entries where original_names != canonical_name, OR has multiple originals)
    banner("ALIASES MERGED")
    alias_rows = entities_df[
        (entities_df["original_names"].str.contains(" | ", regex=False)) |
        (entities_df["original_names"] != entities_df["canonical_name"])
    ]
    if len(alias_rows) == 0:
        print("  (no merged aliases)")
    else:
        for _, r in alias_rows.iterrows():
            print(f"  [{r['entity_type']}] {r['canonical_name']}")
            print(f"      <- {r['original_names']}")
            print(f"      (occurrences={r['occurrences']}, methods={r['source_methods']})")

    # Explicit aliases applied
    explicit_applied = raw[raw["__alias_applied"] == True][
        ["entity", "canonical_name", "entity_type"]
    ].drop_duplicates()
    if len(explicit_applied) > 0:
        print(f"\n  Explicit aliases applied (NHNN → ...):")
        for _, r in explicit_applied.iterrows():
            print(f"    [{r['entity_type']}] {r['entity']!r} → {r['canonical_name']!r}")

    # Top 10 samples
    banner("10 SAMPLE ENTITIES")
    sample = entities_df.head(10)
    for _, r in sample.iterrows():
        print(f"  {r['entity_id']} [{r['entity_type']}] {r['canonical_name']}")
        print(f"      aliases    : {r['original_names']}")
        print(f"      occurrences: {r['occurrences']}  methods={r['source_methods']}  first_doc={r['first_seen_doc']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
