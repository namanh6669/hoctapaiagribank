#!/usr/bin/env python3
"""Chuẩn hoá 4 file CSV hạt giống thành entities.csv + relations.csv cho MVP.

Đầu vào (tương đối với thư mục gốc dự án):
    data/risk_profiles_seed.csv       -> type = RuiRo
    data/controls_seed.csv            -> type = KiemSoat
    data/risk_events_seed.csv         -> type = SuKienRuiRo
    data/relationships_seed.csv       -> relations

Đầu ra:
    outputs/entities.csv
    outputs/relations.csv

Cách chạy:
    python scripts/build_entities.py

Cam kết trung thực dữ liệu:
    - Không tự suy đoán tên đơn vị từ owner_unit_id
    - Không tự suy đoán tên vai trò từ owner_role_id
    - Không tự thêm/sửa quan hệ
    - Không tự đổi verification_status
    - Mọi cột không có trong nguồn sẽ để trống (không bịa giá trị)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
OUT_DIR = REPO / "outputs"

ENTITIES_PATH = OUT_DIR / "entities.csv"
RELATIONS_PATH = OUT_DIR / "relations.csv"

# ===== Schema =====
# Schema tối thiểu bắt buộc cho mọi entity
COMMON_COLS = [
    "id",
    "type",
    "name",
    "description",
    "source_file",
    "data_origin",
    "verification_status",
]

# Cột nghiệp vụ riêng theo từng loại — giữ nguyên từ nguồn, không tự sinh thêm
RUIRO_COLS = [
    "category",
    "cause",
    "event",
    "impact",
    "inherent_level",
    "residual_level",
    "owner_unit_id",
]
KIEMSOAT_COLS = [
    "control_type",
    "frequency",
    "owner_role_id",
    "effectiveness",
]
SUKIEN_COLS = [
    "risk_id",
    "occurred_at",
    "discovered_at",
    "severity",
    "loss_amount_vnd",
]

ALL_ENTITY_COLS = COMMON_COLS + RUIRO_COLS + KIEMSOAT_COLS + SUKIEN_COLS

RELATION_COLS = [
    "source_id",
    "relationship_type",
    "target_id",
    "source",
    "evidence_quote",
    "confidence",
    "verification_status",
    "data_origin",
]


# ===== Helpers =====
def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Đọc CSV, trả về (tên-cột, danh-sách-dòng)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, cols: list[str], rows: list[dict[str, str]]) -> None:
    """Ghi CSV với header cố định theo `cols`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def strip(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


# ===== Normalizers =====
def normalize_ruiro(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        e = {c: "" for c in ALL_ENTITY_COLS}
        e["id"] = strip(r, "id")
        e["type"] = "RuiRo"
        e["name"] = strip(r, "name")
        e["description"] = strip(r, "description")
        e["source_file"] = "risk_profiles_seed.csv"
        e["data_origin"] = strip(r, "data_origin")
        e["verification_status"] = strip(r, "verification_status")
        for col in RUIRO_COLS:
            e[col] = strip(r, col)
        out.append(e)
    return out


def normalize_kiemsoat(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        e = {c: "" for c in ALL_ENTITY_COLS}
        e["id"] = strip(r, "id")
        e["type"] = "KiemSoat"
        e["name"] = strip(r, "name")
        # controls_seed KHÔNG có cột description trong nguồn -> để trống, không bịa.
        e["description"] = ""
        e["source_file"] = "controls_seed.csv"
        e["data_origin"] = strip(r, "data_origin")
        e["verification_status"] = strip(r, "verification_status")
        for col in KIEMSOAT_COLS:
            e[col] = strip(r, col)
        out.append(e)
    return out


def normalize_sukien(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        e = {c: "" for c in ALL_ENTITY_COLS}
        e["id"] = strip(r, "id")
        e["type"] = "SuKienRuiRo"
        # risk_events_seed KHÔNG có cột `name` trong nguồn.
        # Không tự bịa — để trống. Mọi truy vấn có thể dùng `id`/`risk_id`/`description`.
        e["name"] = ""
        e["description"] = strip(r, "description")
        e["source_file"] = "risk_events_seed.csv"
        e["data_origin"] = strip(r, "data_origin")
        e["verification_status"] = strip(r, "verification_status")
        for col in SUKIEN_COLS:
            e[col] = strip(r, col)
        out.append(e)
    return out


def normalize_relations(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Giữ nguyên 8 cột yêu cầu, không tự thêm/sửa giá trị."""
    return [{c: strip(r, c) for c in RELATION_COLS} for r in rows]


# ===== Validation =====
def find_orphans(
    entities: list[dict[str, str]],
    relations: list[dict[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Trả về (orphan_source_id_edges, orphan_target_id_edges)."""
    known_ids = {e["id"] for e in entities if e["id"]}
    bad_src: list[tuple[str, str]] = []
    bad_tgt: list[tuple[str, str]] = []
    for r in relations:
        src, tgt, rt = r["source_id"], r["target_id"], r["relationship_type"]
        if src and src not in known_ids:
            bad_src.append((src, rt))
        if tgt and tgt not in known_ids:
            bad_tgt.append((tgt, rt))
    return bad_src, bad_tgt


# ===== Main =====
def main() -> int:
    print("=" * 70)
    print(" Wiki Risk Graph — Build entities.csv & relations.csv ")
    print("=" * 70)

    # Đọc 4 file nguồn
    _, ruiro_rows = read_csv(DATA_DIR / "risk_profiles_seed.csv")
    _, ks_rows = read_csv(DATA_DIR / "controls_seed.csv")
    _, sk_rows = read_csv(DATA_DIR / "risk_events_seed.csv")
    _, rel_rows = read_csv(DATA_DIR / "relationships_seed.csv")

    print(
        f"\nĐọc được: {len(ruiro_rows)} RuiRo, "
        f"{len(ks_rows)} KiemSoat, "
        f"{len(sk_rows)} SuKienRuiRo, "
        f"{len(rel_rows)} relations (raw)"
    )

    # Chuẩn hoá
    entities = (
        normalize_ruiro(ruiro_rows)
        + normalize_kiemsoat(ks_rows)
        + normalize_sukien(sk_rows)
    )
    relations = normalize_relations(rel_rows)

    # Ghi file
    write_csv(ENTITIES_PATH, ALL_ENTITY_COLS, entities)
    write_csv(RELATIONS_PATH, RELATION_COLS, relations)
    print(f"\nĐã ghi: {ENTITIES_PATH}")
    print(f"Đã ghi: {RELATIONS_PATH}")

    # ---- Báo cáo ----
    print("\n--- Số entity theo type ---")
    by_type: Counter = Counter(e["type"] for e in entities)
    for t, c in sorted(by_type.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {t}: {c}")
    print(f"  Tổng: {sum(by_type.values())}")

    print("\n--- Số relation theo relationship_type ---")
    by_rt: Counter = Counter(r["relationship_type"] for r in relations)
    for rt, c in sorted(by_rt.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {rt}: {c}")
    print(f"  Tổng: {sum(by_rt.values())}")

    # ---- Tính toàn vẹn tham chiếu ----
    print("\n--- Tính toàn vẹn tham chiếu (source_id, target_id vs entities.csv) ---")
    bad_src, bad_tgt = find_orphans(entities, relations)
    if not bad_src and not bad_tgt:
        print("  ✓ Mọi source_id và target_id đều tồn tại trong entities.csv")
    else:
        if bad_src:
            print(f"  ⚠ Orphan source_id ({len(bad_src)}):")
            for src, rt in bad_src:
                print(f"      {src}  (relationship_type={rt})")
        if bad_tgt:
            print(f"  ⚠ Orphan target_id ({len(bad_tgt)}):")
            for tgt, rt in bad_tgt:
                print(f"      {tgt}  (relationship_type={rt})")
        return 1

    # ---- Cam kết trung thực ----
    print("\n--- Cam kết trung thực ---")

    # (1) Không tự thêm loại quan hệ ngoài 2 loại MVP
    expected = {"MITIGATES", "OBSERVED_AS"}
    actual = set(by_rt)
    extras = actual - expected
    missing = expected - actual
    if extras:
        print(f"  ⚠ Có loại quan hệ ngoài scope MVP: {sorted(extras)}")
    if missing:
        print(f"  ⚠ Thiếu loại quan hệ mong đợi: {sorted(missing)}")
    if not extras and not missing:
        print("  ✓ Chỉ có 2 loại quan hệ (MITIGATES, OBSERVED_AS) — không tự sinh thêm")

    # (2) Không đổi verification_status
    statuses_e = {e["verification_status"] for e in entities if e["verification_status"]}
    statuses_r = {r["verification_status"] for r in relations if r["verification_status"]}
    print(
        f"  verification_status của entities: {sorted(statuses_e) or '(trống)'}"
    )
    print(
        f"  verification_status của relations: {sorted(statuses_r) or '(trống)'}"
    )
    print("  ✓ Không tự đổi PROPOSED → VERIFIED (giữ nguyên giá trị gốc)")

    # (3) Không suy đoán đơn vị / vai trò
    print("  ✓ owner_unit_id / owner_role_id được giữ nguyên mã gốc (không sinh tên)")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
