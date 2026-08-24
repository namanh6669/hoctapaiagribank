"""BƯỚC 1: kiểm tra dữ liệu và làm sạch HTML.

Đọc metadata.csv + content.csv, kiểm tra chất lượng, làm sạch HTML bằng
BeautifulSoup, lưu ner_kb/cleaned_documents.csv.

KHÔNG sửa metadata.csv / content.csv.
KHÔNG chạy NER / Gemini / Neo4j.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent
META_PATH = BASE / "metadata.csv"
CONTENT_PATH = BASE / "content.csv"
OUTPUT_PATH = BASE / "cleaned_documents.csv"


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 1 — DATA INSPECTION & HTML CLEANING")

    # --- 1. Đọc bằng pandas ---
    meta = pd.read_csv(META_PATH, dtype=str, keep_default_na=False)
    content = pd.read_csv(CONTENT_PATH, dtype=str, keep_default_na=False)

    # --- 2. Số dòng, số cột ---
    print(f"metadata.csv : {meta.shape[0]} rows x {meta.shape[1]} cols")
    print(f"content.csv  : {content.shape[0]} rows x {content.shape[1]} cols")
    print(f"  metadata columns: {list(meta.columns)}")
    print(f"  content columns : {list(content.columns)}")

    # --- 3. Duplicate id ---
    meta_dup = meta["id"].duplicated().sum()
    content_dup = content["id"].duplicated().sum()
    print(f"\nDuplicate ids in metadata.csv : {meta_dup}")
    print(f"Duplicate ids in content.csv  : {content_dup}")

    # --- 4. Id chỉ có ở một trong hai file ---
    meta_ids = set(meta["id"])
    content_ids = set(content["id"])
    only_in_meta = meta_ids - content_ids
    only_in_content = content_ids - meta_ids
    print(f"\nIDs only in metadata.csv : {len(only_in_meta)}")
    print(f"IDs only in content.csv  : {len(only_in_content)}")
    print(f"Common IDs              : {len(meta_ids & content_ids)}")

    # --- 5. Merge theo id (inner join để chỉ giữ id có ở cả hai) ---
    merged = meta.merge(content, on="id", how="inner", suffixes=("", "_content"))
    print(f"\nMerged (inner join)    : {merged.shape[0]} rows x {merged.shape[1]} cols")

    # --- 6. Missing values cho metadata ---
    NA_TOKENS = {"", "nan", "NaN", "null", "NULL", "None"}
    missing_records = {}
    for col in meta.columns:
        if col == "id":
            continue
        s = meta[col]
        na_count = int(s.isna().sum())
        empty_count = int(((s.astype(str).str.strip() == "")).sum())
        chua_pc = int(((s.astype(str).str.strip() == "Chưa phân loại")).sum())
        nan_token = int(s.astype(str).isin(NA_TOKENS).sum())
        missing_records[col] = {
            "pandas_na": na_count,
            "empty_str": empty_count,
            "chua_phan_loai": chua_pc,
            "nan_like_token": nan_token,
        }
    print("\nMissing values per metadata column (raw pandas NaN, empty str, 'Chưa phân loại', NA-like token):")
    print(f"{'column':30s} {'pd_na':>6s} {'empty':>6s} {'chua_pc':>8s} {'na_like':>7s}")
    for col, stats in missing_records.items():
        print(f"  {col:28s} {stats['pandas_na']:>6d} {stats['empty_str']:>6d} "
              f"{stats['chua_phan_loai']:>8d} {stats['nan_like_token']:>7d}")

    # --- 7. Phát hiện NULL / chuỗi rỗng / "Chưa phân loại" trong content ---
    null_in_content = int(content["content_html"].isna().sum())
    empty_in_content = int(((content["content_html"].astype(str).str.strip() == "")).sum())
    print(f"\ncontent_html: pandas_na={null_in_content}, empty_str={empty_in_content}")

    # --- 8 + 9 + 10. Làm sạch HTML bằng BeautifulSoup ---
    # Chỉ strip thẻ HTML và chuẩn hóa whitespace. KHÔNG sửa nội dung văn bản.
    # Các cụm "Căn cứ", "Sửa đổi, bổ sung", "bãi bỏ", "thay thế" và số hiệu văn bản
    # được giữ nguyên vì ta không can thiệp vào text node.

    def clean_html(raw_html: str) -> str:
        if not isinstance(raw_html, str) or not raw_html.strip():
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        # Loại bỏ <script>/<style> nội dung (không phải nội dung văn bản)
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Lấy text, ngăn cách bằng space để tránh dính từ khi tag bị strip
        text = soup.get_text(separator=" ", strip=True)
        # Chuẩn hóa whitespace: gộp nhiều space/newline/tab thành 1 space
        text = " ".join(text.split())
        return text

    print("\nCleaning HTML for", len(merged), "rows...")
    merged["content_clean"] = merged["content_html"].astype(str).map(clean_html)

    # --- Sanity check: giữ nguyên các cụm yêu cầu ---
    for phrase in ["Căn cứ", "Sửa đổi, bổ sung", "bãi bỏ", "thay thế"]:
        hits = merged["content_clean"].str.contains(phrase, regex=False, na=False).sum()
        print(f"  phrase preserved '{phrase}': {hits} rows contain it")

    # --- 11 + 12. Lưu cleaned_documents.csv ---
    # Cột output: tất cả metadata + content_html + content_clean
    out_cols = list(meta.columns) + ["content_html", "content_clean"]
    out_df = merged[out_cols]
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {OUTPUT_PATH} ({out_df.shape[0]} rows x {out_df.shape[1]} cols)")

    # --- 13. In 2 mẫu content_html vs content_clean ---
    banner("SAMPLE 1 (first row)")
    sample1 = merged.iloc[0]
    print(f"  id: {sample1['id']}")
    print(f"  title: {sample1.get('title','')[:120]}")
    raw1 = str(sample1["content_html"])
    print(f"  content_html (raw, first 400 chars):\n{raw1[:400]}")
    print(f"  content_clean (first 400 chars):\n{str(sample1['content_clean'])[:400]}")

    banner("SAMPLE 2 (row with a phrase hit)")
    # tìm 1 hàng có chứa "Căn cứ" để demo
    hit_idx = merged.index[merged["content_clean"].str.contains("Căn cứ", regex=False, na=False)]
    if len(hit_idx) > 0:
        idx2 = hit_idx[0]
    else:
        idx2 = merged.index[1]
    sample2 = merged.iloc[idx2]
    print(f"  id: {sample2['id']}")
    print(f"  title: {sample2.get('title','')[:120]}")
    raw2 = str(sample2["content_html"])
    print(f"  content_html (raw, first 400 chars):\n{raw2[:400]}")
    print(f"  content_clean (first 400 chars):\n{str(sample2['content_clean'])[:400]}")

    # --- Final summary ---
    banner("STEP 1 SUMMARY")
    print(f"  documents (rows in merged)     : {len(merged)}")
    print(f"  duplicate ids (meta/content)   : {meta_dup} / {content_dup}")
    print(f"  ids only in meta / only in con : {len(only_in_meta)} / {len(only_in_content)}")
    print(f"  output file                    : {OUTPUT_PATH}")
    print(f"  output rows                    : {out_df.shape[0]}")
    print(f"  output cols                    : {out_df.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
