#!/usr/bin/env python3
"""Kiểm tra 4 file CSV hạt giống cho Wiki Risk Graph MVP.

Cách chạy:
    python scripts/inspect_data.py

Đầu vào (tương đối với thư mục gốc dự án):
    data/risk_profiles_seed.csv
    data/controls_seed.csv
    data/risk_events_seed.csv
    data/relationships_seed.csv

Báo cáo:
    - số dòng & tên cột của từng file
    - khóa chính (cột `id`) và tính duy nhất
    - ứng viên khóa ngoại và tính toàn vẹn tham chiếu
    - phân bố loại quan hệ
    - số giá trị null/empty theo cột
    - trùng lặp khóa & trùng lặp dòng
    - tham chiếu bị thiếu (nếu có)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 3 bảng có cột `id` làm khóa chính
NODE_TABLES = {
    "risk_profiles_seed": "id",
    "controls_seed": "id",
    "risk_events_seed": "id",
}

# Khóa ngoại được khai báo trong mô tả dữ liệu (ngoài quan hệ trong relationships_seed)
FK_CANDIDATES = {
    "risk_events_seed": [("risk_id", "risk_profiles_seed")],
}

HEADER = "=" * 72
SUBHEADER = "-" * 72


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Đọc CSV và trả về (tên-cột, danh-sách-dòng)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    return cols, rows


def inspect_node_table(name: str, cols: list[str], rows: list[dict[str, str]]) -> dict:
    pk = NODE_TABLES[name]
    ids = [r.get(pk, "").strip() for r in rows]
    id_counter = Counter(ids)
    pk_dups = sorted(i for i, c in id_counter.items() if c > 1 and i)
    empty_ids = sum(1 for i in ids if not i)

    nulls = {c: sum(1 for r in rows if not (r.get(c) or "").strip()) for c in cols}
    row_keys = Counter(tuple(r.get(c, "") for c in cols) for r in rows)
    row_dups = [k for k, c in row_keys.items() if c > 1]

    return {
        "name": name,
        "rows": len(rows),
        "cols": cols,
        "pk": pk,
        "id_count": sum(1 for i in ids if i),
        "unique_ids": len({i for i in ids if i}),
        "pk_dups": pk_dups,
        "empty_ids": empty_ids,
        "nulls": nulls,
        "row_dups_count": len(row_dups),
        "ids": {i for i in ids if i},
    }


def inspect_relationship_table(cols: list[str], rows: list[dict[str, str]]) -> dict:
    rt_counter = Counter(r.get("relationship_type", "").strip() for r in rows)
    nulls = {c: sum(1 for r in rows if not (r.get(c) or "").strip()) for c in cols}

    edge_keys = Counter(
        (
            r.get("source_id", "").strip(),
            r.get("relationship_type", "").strip(),
            r.get("target_id", "").strip(),
        )
        for r in rows
    )
    edge_dups = [k for k, c in edge_keys.items() if c > 1 and all(k)]

    return {
        "rows": len(rows),
        "cols": cols,
        "rels": dict(rt_counter),
        "nulls": nulls,
        "edge_dups": edge_dups,
        "raw": rows,
    }


def main() -> int:
    print(HEADER)
    print(" Wiki Risk Graph — Inspect Seed Data (MVP) ")
    print(HEADER)
    print(f"Data directory: {DATA_DIR}\n")

    tables: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for name in NODE_TABLES:
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            print(f"⚠ Missing file: {path}")
            continue
        tables[name] = read_csv(path)

    rel_path = DATA_DIR / "relationships_seed.csv"
    if rel_path.exists():
        tables["relationships_seed"] = read_csv(rel_path)
    else:
        print(f"⚠ Missing file: {rel_path}")

    # === Báo cáo từng bảng node ===
    node_reports: dict[str, dict] = {}
    for name in NODE_TABLES:
        if name not in tables:
            continue
        cols, rows = tables[name]
        rep = inspect_node_table(name, cols, rows)
        node_reports[name] = rep

        print(f"\n{SUBHEADER}")
        print(f"[{name}]")
        print(SUBHEADER)
        print(f"Rows : {rep['rows']}")
        print(f"Cols : {len(rep['cols'])}  ->  {', '.join(rep['cols'])}")
        print(f"PK   : {rep['pk']}")
        print(
            f"      non-empty={rep['id_count']}, "
            f"unique={rep['unique_ids']}, empty={rep['empty_ids']}"
        )
        if rep["pk_dups"]:
            print(f"      ⚠ Duplicate PK values: {rep['pk_dups']}")
        if rep["row_dups_count"]:
            print(f"      ⚠ Duplicate row content: {rep['row_dups_count']} tuple(s)")
        print("Null/empty per column:")
        for c, n in rep["nulls"].items():
            mark = "  ⚠" if n else ""
            print(f"  {c}: {n}{mark}")

    # === Khóa ngoại được khai báo ===
    print(f"\n{SUBHEADER}")
    print("[Foreign key reference checks (declared)]")
    print(SUBHEADER)
    for tbl, pairs in FK_CANDIDATES.items():
        for col, ref in pairs:
            print(f"  {tbl}.{col} -> {ref}.id")
    for tbl, pairs in FK_CANDIDATES.items():
        if tbl not in tables:
            continue
        cols, rows = tables[tbl]
        ref_name = pairs[0][1]
        ref_rep = node_reports.get(ref_name)
        if ref_rep is None:
            continue
        ref_ids = ref_rep["ids"]
        for col, ref in pairs:
            counter = Counter(r.get(col, "").strip() for r in rows)
            missing = {k: c for k, c in counter.items() if k and k not in ref_ids}
            print(f"\n  {tbl}.{col} -> {ref}.id")
            print(f"    Distinct referenced values: {len(counter)}")
            if missing:
                print(f"    ⚠ Missing references:")
                for k, c in sorted(missing.items()):
                    print(f"        {k}: {c} lần")
            else:
                print(f"    ✓ Tất cả tham chiếu đều trỏ về {ref}")

    # === Bảng quan hệ ===
    if "relationships_seed" in tables:
        cols, rows = tables["relationships_seed"]
        rep = inspect_relationship_table(cols, rows)

        print(f"\n{SUBHEADER}")
        print("[relationships_seed]")
        print(SUBHEADER)
        print(f"Rows : {rep['rows']}")
        print(f"Cols : {len(rep['cols'])}  ->  {', '.join(rep['cols'])}")
        print("Null/empty per column:")
        for c, n in rep["nulls"].items():
            mark = "  ⚠" if n else ""
            print(f"  {c}: {n}{mark}")

        print("\nRelationship types:")
        for rt, c in sorted(rep["rels"].items(), key=lambda x: -x[1]):
            print(f"  {rt or '(empty)'}: {c}")
        if rep["edge_dups"]:
            print("\n⚠ Duplicate (source_id, type, target_id) triples:")
            for k in rep["edge_dups"]:
                print(f"  {k}")

        print("\nPer-type source/target prefix breakdown:")
        by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in rows:
            by_type[r.get("relationship_type", "").strip()].append(r)
        for rt in sorted(by_type):
            rt_rows = by_type[rt]
            src_prefix = Counter(
                (r.get("source_id", "").split("-", 1)[0]
                 if "-" in r.get("source_id", "")
                 else r.get("source_id", "") or "(empty)")
                for r in rt_rows
            )
            tgt_prefix = Counter(
                (r.get("target_id", "").split("-", 1)[0]
                 if "-" in r.get("target_id", "")
                 else r.get("target_id", "") or "(empty)")
                for r in rt_rows
            )
            print(f"  [{rt}] rows={len(rt_rows)}")
            print(f"    source_id prefixes: {dict(src_prefix)}")
            print(f"    target_id prefixes: {dict(tgt_prefix)}")

        # Tính toàn vẹn tham chiếu so với hợp của node IDs
        all_node_ids: set[str] = set()
        for r_ in node_reports.values():
            all_node_ids |= r_["ids"]
        missing_src: Counter = Counter()
        missing_tgt: Counter = Counter()
        src_per_type: dict[str, Counter] = defaultdict(Counter)
        tgt_per_type: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            src = r.get("source_id", "").strip()
            tgt = r.get("target_id", "").strip()
            rt = r.get("relationship_type", "").strip()
            if src and src not in all_node_ids:
                missing_src[src] += 1
                src_per_type[rt][src] += 1
            if tgt and tgt not in all_node_ids:
                missing_tgt[tgt] += 1
                tgt_per_type[rt][tgt] += 1

        print(f"\nReference integrity (vs union of node IDs across 3 node tables):")
        print(f"  Known node IDs: {len(all_node_ids)}")
        if missing_src:
            print("  ⚠ Missing source_id references:")
            for k, c in missing_src.most_common():
                rts = ", ".join(
                    f"{rt_}={cnt}" for rt_, cnt in src_per_type.items()
                    if k in src_per_type[rt_]
                )
                print(f"      {k}: {c} ({rts})")
        else:
            print("  ✓ Mọi source_id đều trỏ về một node đã biết")
        if missing_tgt:
            print("  ⚠ Missing target_id references:")
            for k, c in missing_tgt.most_common():
                rts = ", ".join(
                    f"{rt_}={cnt}" for rt_, cnt in tgt_per_type.items()
                    if k in tgt_per_type[rt_]
                )
                print(f"      {k}: {c} ({rts})")
        else:
            print("  ✓ Mọi target_id đều trỏ về một node đã biết")

    print(f"\n{HEADER}")
    print("Done.")
    print(HEADER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
