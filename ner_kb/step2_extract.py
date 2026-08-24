"""BƯỚC 2: rule-based candidate extraction.

Đọc ner_kb/cleaned_documents.csv, phát hiện số hiệu văn bản được nhắc trong
content_clean cùng trigger (Căn cứ / Sửa đổi, bổ sung / bãi bỏ / thay thế).
Lưu ner_kb/relation_candidates.csv.

KHÔNG dùng Gemini, KHÔNG tạo relationships.csv, KHÔNG import Neo4j.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT_PATH = BASE / "cleaned_documents.csv"
OUTPUT_PATH = BASE / "relation_candidates.csv"

# ----------Patterns----------

# Số hiệu văn bản, ví dụ:
#   32/2024/QH15, 73/2016/NĐ-CP, 37/2014/TT-NHNN, 17/VBHN-BTC, 202/2012/TT-BTC
# Format: <số>/<phần-2>/<phần-3?>
#   phần-2: 4 chữ số (năm) HOẶC dạng VBHN-XXX (chữ + dấu gạch + chữ)
#   phần-3: chữ cái đầu viết hoa, có dấu hoặc không, có thể có - và phần sau
# Yêu cầu TỐI THIỂU 2 phần; phần-3 bắt buộc có ÍT NHẤT một chữ cái
#   để loại bỏ pattern ngày tháng (NN/NN/YYYY toàn số).
VN_UPPER = "A-ZĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẠÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
DOC_NUMBER_RE = re.compile(
    rf"\b\d{{1,4}}/(?:\d{{4}}|{VN_UPPER[1:]}(?:[{VN_UPPER}\-0-9]{{1,12}})?)"
    rf"(?:/[{VN_UPPER}][{VN_UPPER}0-9\-]{{1,12}})?\b"
)

# Các trigger ưu tiên trong spec (thứ tự kiểm tra)
# Tránh match "Căn cứ" đã nằm trong "Căn cứ vào" các biến thể không liên quan —
# ở đây ta lấy cụm đầy đủ "Căn cứ" và "Căn cứ vào" đều là tín hiệu citation.
TRIGGERS = [
    ("Căn cứ", "Căn cứ"),
    ("Sửa đổi, bổ sung", "Sửa đổi, bổ sung"),
    ("bãi bỏ", "bãi bỏ"),
    ("thay thế", "thay thế"),
]

# Số ký tự context lấy quanh trigger để tìm số hiệu
WINDOW_AFTER = 400   # số hiệu thường xuất hiện SAU trigger
WINDOW_BEFORE = 200  # nhưng đôi khi cũng nằm trước


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def is_valid_doc_number(target: str) -> bool:
    """Lọc các match rác: ngày tháng, số fragment.

    Quy tắc:
      - phải có >= 2 phần
      - 2 phần: phần-2 phải chứa chữ cái (loại NN/NN ngày-tháng)
      - 3 phần: KHÔNG được cả 3 phần toàn là số (loại NN/NN/YYYY ngày-tháng)
      - phần-3 (nếu có) phải có chữ cái
    """
    parts = target.split("/")
    if len(parts) < 2:
        return False
    if len(parts) == 2:
        # 2 phần: phần 2 phải chứa chữ cái (VD: VBHN-BTC)
        return any(ch.isalpha() for ch in parts[1])
    # 3 phần: loại NN/NN/YYYY ngày tháng
    if all(p.isdigit() for p in parts):
        return False
    # Phần 3 phải có chữ cái
    return any(ch.isalpha() for ch in parts[2])


def find_candidates_for_doc(
    source_id: str,
    source_so_ky_hieu: str,
    content_clean: str,
) -> list[dict]:
    """Tìm candidates cho một document.

    Với mỗi trigger phrase:
      - lấy context window sau trigger (WINDOW_AFTER chars)
      - scan tất cả số hiệu văn bản xuất hiện trong window
      - tạo 1 candidate cho mỗi (trigger, số hiệu mục tiêu)
    """
    if not isinstance(content_clean, str) or not content_clean.strip():
        return []

    candidates: list[dict] = []
    seen_in_doc: set[tuple[str, str]] = set()  # (trigger, target) chống trùng trong 1 doc

    for trigger_label, trigger_pat in TRIGGERS:
        for m in re.finditer(re.escape(trigger_pat), content_clean):
            start = m.start()
            window_start = max(0, start - WINDOW_BEFORE)
            window_end = min(len(content_clean), start + WINDOW_AFTER)
            window = content_clean[window_start:window_end]

            for num_m in DOC_NUMBER_RE.finditer(window):
                target = num_m.group(0)

                # Lọc rác (ngày tháng, v.v.)
                if not is_valid_doc_number(target):
                    continue

                # Bỏ self-reference: target khớp hoặc là prefix của source_so_ky_hieu
                if target == source_so_ky_hieu:
                    continue
                if source_so_ky_hieu.startswith(target + "/"):
                    continue

                key = (trigger_label, target)
                if key in seen_in_doc:
                    continue
                seen_in_doc.add(key)

                ev_start = max(0, num_m.start() - 80)
                ev_end = min(len(window), num_m.end() + 80)
                evidence = window[ev_start:ev_end].strip()

                candidates.append({
                    "source_id": source_id,
                    "source_so_ky_hieu": source_so_ky_hieu,
                    "target_so_ky_hieu": target,
                    "trigger": trigger_label,
                    "evidence": evidence,
                })
    return candidates


def main() -> int:
    banner("STEP 2 — RULE-BASED CANDIDATE EXTRACTION")

    df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    print(f"Loaded {INPUT_PATH.name}: {len(df)} rows")

    all_candidates: list[dict] = []
    for _, row in df.iterrows():
        all_candidates.extend(
            find_candidates_for_doc(
                source_id=row["id"],
                source_so_ky_hieu=row["so_ky_hieu"],
                content_clean=row["content_clean"],
            )
        )

    cand_df = pd.DataFrame(
        all_candidates,
        columns=["source_id", "source_so_ky_hieu", "target_so_ky_hieu", "trigger", "evidence"],
    )

    # Loại duplicate toàn cục (cùng source_id + target + trigger)
    before = len(cand_df)
    cand_df = cand_df.drop_duplicates(
        subset=["source_id", "target_so_ky_hieu", "trigger"]
    ).reset_index(drop=True)
    after = len(cand_df)
    print(f"\nRemoved {before - after} duplicate candidates (global dedup on source_id + target + trigger)")

    # Lưu
    cand_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUTPUT_PATH} ({len(cand_df)} rows x {cand_df.shape[1]} cols)")

    # Thống kê
    banner("STEP 2 STATS")
    print(f"  total candidates           : {len(cand_df)}")
    print(f"  unique source docs        : {cand_df['source_id'].nunique()}")
    print(f"  unique target numbers     : {cand_df['target_so_ky_hieu'].nunique()}")
    print()
    print("  candidates per trigger:")
    for trig, n in cand_df["trigger"].value_counts().items():
        print(f"    {trig:25s} : {n}")

    # 10 mẫu
    banner("10 SAMPLE CANDIDATES")
    sample = cand_df.head(10)
    for i, row in sample.iterrows():
        print(f"\n  [{i+1}] source_id={row['source_id']} ({row['source_so_ky_hieu']})")
        print(f"      target   = {row['target_so_ky_hieu']}")
        print(f"      trigger  = {row['trigger']}")
        print(f"      evidence = {row['evidence'][:160]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
