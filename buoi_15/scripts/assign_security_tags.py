#!/usr/bin/env python3
"""assign_security_tags.py — Bu�i 15.

Phân loại bảo mật cho ``chunks_normalized.csv`` và ghi ra
``chunks_secure.csv`` với cột ``allowed_roles`` (chuỗi role phân tách
bằng dấu phẩy) + ``security_label`` (HR / Risk / General).

Quy tắc phân loại — ưu tiên từ trên xuống:

  1. document_id prefix: ``HR-…``  → HR category
                            ``RISK-…`` → Risk category
  2. Keyword trong (title + text):
       HR keywords  = nhân sự, lương thưởng, tuyển dụng, bổ nhiệm
       Risk keywords = tín dụng, rủi ro, hạn mức, phê duyệt khoản vay
  3. Mặc định: General (mọi role đều thấy).

Lưu ý về role names trong đề bài: đề dùng ``HR / Risk_Manager / Staff``
làm ví dụ. Roles THỰC TẾ đã chọn ở Buổi 15 là
``Admin, HR_Manager, Risk_Officer, Employee, Guest`` — mapping được
khai báo trong ``_LEGACY_TO_CANONICAL`` bên dưới.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `src.config` importable when this script is run directly.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROLE_LIST, VALID_ROLES, assert_valid_role  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Classification keywords
# ---------------------------------------------------------------------------

HR_KEYWORDS: tuple[str, ...] = (
    "nhân sự",
    "lương thưởng",
    "tuyển dụng",
    "bổ nhiệm",
)

RISK_KEYWORDS: tuple[str, ...] = (
    "tín dụng",
    "rủi ro",
    "hạn mức",
    "phê duyệt khoản vay",
)

# Optional: convention cho mã tài liệu.
DOC_ID_HR_PREFIXES: tuple[str, ...] = ("HR-", "HR_")
DOC_ID_RISK_PREFIXES: tuple[str, ...] = ("RISK-", "RISK_", "CREDIT-")

# ---------------------------------------------------------------------------
# 2. Role mapping (legacy names used in the lesson → canonical VALID_ROLES)
# ---------------------------------------------------------------------------

# Canonical (chỉ chứa role có trong VALID_ROLES).
HR_ROLES: frozenset[str] = frozenset({"Admin", "HR_Manager"})
RISK_ROLES: frozenset[str] = frozenset({"Admin", "Risk_Officer", "Employee"})
GENERAL_ROLES: frozenset[str] = frozenset(VALID_ROLES)  # all roles

# Kiểm tra lúc import: không để typo lọt vào canonical roles.
for _r in HR_ROLES | RISK_ROLES | GENERAL_ROLES:
    assert_valid_role(_r)


# ---------------------------------------------------------------------------
# 3. Classifier
# ---------------------------------------------------------------------------


def classify(
    document_id: str,
    title: str,
    text: str,
) -> tuple[frozenset[str], str]:
    """Return ``(allowed_roles, security_label)`` for one chunk."""
    doc_id = (document_id or "").strip().upper()

    # Priority 1 — document_id prefix.
    if any(doc_id.startswith(p) for p in DOC_ID_HR_PREFIXES):
        return HR_ROLES, "HR"
    if any(doc_id.startswith(p) for p in DOC_ID_RISK_PREFIXES):
        return RISK_ROLES, "Risk"

    # Priority 2 — keyword scan over title + text.
    haystack = f"{title or ''}\n{text or ''}".lower()
    if any(kw.lower() in haystack for kw in HR_KEYWORDS):
        return HR_ROLES, "HR"
    if any(kw.lower() in haystack for kw in RISK_KEYWORDS):
        return RISK_ROLES, "Risk"

    # Default — general policy text, every role can see it.
    return GENERAL_ROLES, "General"


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------


def validate(df: pd.DataFrame) -> None:
    """Ensure every row has at least one role + print stats."""
    # a) No empty / null allowed_roles.
    empty_mask = df["allowed_roles"].fillna("").str.strip() == ""
    empty_rows = df[empty_mask]
    if len(empty_rows) > 0:
        raise SystemExit(
            f"� VALIDATION FAILED: {len(empty_rows)} chunks have empty allowed_roles. "
            f"First offenders:\n{empty_rows[['chunk_id', 'document_id']].head().to_string()}"
        )

    # b) All listed roles must be valid (catch typos in classify()).
    bad: list[str] = []
    for roles_str in df["allowed_roles"]:
        for r in roles_str.split(","):
            r = r.strip()
            if r and r not in VALID_ROLES:
                bad.append(r)
    if bad:
        raise SystemExit(f"❌ VALIDATION FAILED: unknown roles in output: {set(bad)}")

    # c) Stats per category.
    print("\n📊 Security label distribution:")
    counts = df["security_label"].value_counts()
    total = len(df)
    for label in ("HR", "Risk", "General"):
        n = int(counts.get(label, 0))
        pct = 100 * n / total if total else 0
        print(f"   {label:<8} {n:>5} chunks  ({pct:5.1f}%)")

    # d) Per-document VanBan.allowed_roles (INTERSECTION) — printed so the
    #    audit script can verify RBAC at BOTH chunk + document level.
    from collections import defaultdict
    from src.config import assert_valid_role  # noqa: F401
    roles_by_doc: dict[str, list[set[str]]] = defaultdict(list)
    for _, row in df.iterrows():
        roles_by_doc[row["document_id"]].append(
            set(r.strip() for r in row["allowed_roles"].split(",") if r.strip())
        )
    n_admin_only = sum(
        1 for sets in roles_by_doc.values()
        if set.intersection(*sets) == {"Admin"}
    )
    print(
        f"\n📊 VanBan INTERSECTION stats: {n_admin_only}/{len(roles_by_doc)} docs "
        f"chỉ cho Admin (vì có chunk nhạy cảm)."
    )

    # d) One representative sample per category.
    print("\n🔍 Sample rows (1 per category):")
    shown_any = False
    for label in ("HR", "Risk", "General"):
        sub = df[df["security_label"] == label]
        if sub.empty:
            print(f"   [{label}]  (no chunks in this category — corpus has no matching content)")
            continue
        row = sub.iloc[0]
        preview = str(row["text"]).replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:200] + "…"
        print(f"\n   ── [{label}] ─────────────────────────────")
        print(f"   chunk_id      : {row['chunk_id']}")
        print(f"   document_id   : {row['document_id']}")
        print(f"   title         : {row['title']}")
        print(f"   allowed_roles : {row['allowed_roles']}")
        print(f"   text preview  : {preview}")
        shown_any = True

    if not shown_any:
        print("   (no categories had any chunks)")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "processed" / "chunks_normalized.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "chunks_secure.csv"

    if not input_path.exists():
        raise SystemExit(f"❌ Missing input: {input_path}")

    print(f"📂 Reading {input_path.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(input_path)
    print(f"   {len(df):,} chunks loaded ({len(df.columns)} columns)")

    # Classify row by row.
    roles_per_row: list[str] = []
    labels_per_row: list[str] = []
    for _, row in df.iterrows():
        roles, label = classify(
            document_id=row.get("document_id", ""),
            title=row.get("title", ""),
            text=row.get("text", ""),
        )
        # Stable order using ROLE_LIST so output is deterministic.
        ordered = [r for r in ROLE_LIST if r in roles]
        roles_per_row.append(",".join(ordered))
        labels_per_row.append(label)

    df["allowed_roles"] = roles_per_row
    df["security_label"] = labels_per_row

    # Thêm cột ``vanban_allowed_roles`` = INTERSECTION của mọi chunk
    # trong cùng document_id — để BM25/Dense filter ở CẢ 2 cấp (chunk + VanBan)
    # mà không cần query Neo4j.
    from collections import defaultdict
    by_doc: dict[str, list[set[str]]] = defaultdict(list)
    for roles_str in roles_per_row:
        by_doc  # noqa: F401 — just placeholder for typing
    by_doc = defaultdict(list)
    roles_by_doc: dict[str, list[set[str]]] = by_doc
    for roles_str, doc_id in zip(roles_per_row, df["document_id"]):
        roles_by_doc[doc_id].append(set(roles_str.split(",")))
    vanban_roles: list[str] = []
    for doc_id in df["document_id"]:
        sets = roles_by_doc.get(doc_id, [])
        if not sets:
            vanban_roles.append("")
        else:
            inter = sorted(set.intersection(*sets))
            vanban_roles.append(",".join(inter) if inter else "Admin")
    df["vanban_allowed_roles"] = vanban_roles

    # Save.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"� Wrote {output_path.relative_to(PROJECT_ROOT)} ({len(df):,} rows)")

    # Validate.
    validate(df)

    # Re-state role → max-label mapping for traceability.
    print("\n📋 Role → max security label (from src.config):")
    from src.config import ROLE_MAX_LABEL

    for role in ROLE_LIST:
        print(f"   {role:<14} → {ROLE_MAX_LABEL[role]}")


if __name__ == "__main__":
    main()
