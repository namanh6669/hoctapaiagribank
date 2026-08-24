"""Buổi 14 — prepare_corpus.py

Đọc 3 file nguồn (read-only) trong ../ kb+hops/, chuẩn hoá về dạng
chunk phục vụ retrieval. KHÔNG sửa / ghi đè bất kỳ file nguồn nào.

Input
-----
../ kb+hops/metadata.csv      30 dòng, 17 cột (id, title, so_ky_hieu, ...)
../ kb+hops/content.csv       30 dòng, 2 cột (id, content_html — ô HTML rất lớn)
../ kb+hops/relationships.csv 173 dòng, 10 cột — KHÔNG dùng ở bước này

Output
------
data/processed/chunks_normalized.csv
schema tối thiểu:
    chunk_id         uuid5(document_id, article_no, position) — ổn định qua các lần chạy
    document_id      id từ metadata/content (string hoặc UUID)
    text             nội dung sau strip HTML (UTF-8, đã collapse khoảng trắng)
    source_file      đường dẫn tương đối tới file HTML gốc

giữ thêm (nếu có, không bịa):
    title            metadata.title
    document_type    metadata.loai_van_ban
    chapter          "Chương" gần nhất (Roman/Arabic); để trống nếu không có
    section          dự phòng cho "Mục" — để trống ở bước này
    article          số điều (vd. "5" trong "Điều 5."); trống nếu chunk là preamble
    clause           dự phòng — để trống (regex "Khoản" không tin cậy qua mọi văn bản)
    effective_date   metadata.ngay_co_hieu_luc
    status           metadata.tinh_trang_hieu_luc
    so_ky_hieu       metadata.so_ky_hieu              (giữ vì là mã citation quan trọng)
    ngay_ban_hanh    metadata.ngay_ban_hanh
    co_quan_ban_hanh metadata.co_quan_ban_hanh
    nguoi_ky         metadata.nguoi_ky

Quy tắc chuẩn hoá text
----------------------
- Cắt tag HTML bằng BeautifulSoup + lxml.
- Bỏ <script>, <style>.
- get_text("\\n", strip=True), rồi html.unescape.
- Co cụm space/tab về 1, giữ newline làm ranh giữa đoạn.
- Collapse nhiều blank line về 1.
- Không stemming, không xoá số hiệu Điều/Điều khoản.

Chunking
--------
- Tách đoạn preamble (mọi thứ trước Điều đầu tiên) làm 1 chunk với article="".
- Mỗi "Điều N." match → 1 chunk từ đó đến trước "Điều" tiếp theo (hoặc EOF).
- chapter = "Chương X" gần nhất phía trước.

Usage
-----
.venv/bin/python scripts/prepare_corpus.py
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import re
import sys
import uuid
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ---------- paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Nguồn dữ liệu thực tế đặt tại <project>/ kb+hops  (folder tên có dấu cách đầu).
# Tự dò 2 vị trí: trong project trước, ngoài project sau.
KB_CANDIDATES = [
    PROJECT_ROOT / " kb+hops",          # thực tế hiện tại
    PROJECT_ROOT.parent / " kb+hops",    # theo mô tả cấu trúc ban đầu
]
KB_BASE: Path | None = next((p for p in KB_CANDIDATES if p.is_dir()), None)
if KB_BASE is None:
    raise FileNotFoundError(
        f"Cannot locate ` kb+hops/` directory. Tried: {[str(p) for p in KB_CANDIDATES]}"
    )
DATA_OUT = PROJECT_ROOT / "data" / "processed"
DATA_OUT.mkdir(parents=True, exist_ok=True)

SRC_META = KB_BASE / "metadata.csv"
SRC_CONTENT = KB_BASE / "content.csv"
SRC_REL = KB_BASE / "relationships.csv"  # read-only, not used in this step
OUT_CSV = DATA_OUT / "chunks_normalized.csv"

# default cell-size cap (set by csv library); HTML cells exceed 131_072 by default
csv.field_size_limit(sys.maxsize)

# ---------- regex ----------
RE_DIEU = re.compile(r"(?m)^[ \t]*Điều\s+(\d+)\s*[.:]\s*([^\n]*)")
RE_CHUONG = re.compile(r"(?m)^[ \t]*Chương\s+([IVXLCDM\d]+)\s*[.:]?\s*([^\n]*)")
RE_DIEU_FOOTER = re.compile(r"(?im)\bđiều\s+\d+\s*\.\s*$")  # for trimming trailing dots

# fixed namespace for stable chunk_id across runs
NS = uuid.UUID("9e5e2a5b-22b5-4e58-9a1c-1e2b6f4b7a91")


# ---------- helpers ----------
def strip_html(html_str: str) -> str:
    """HTML -> plain text. Conservative: keep \n between block-level elements,
    collapse repeated whitespace."""
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\s*\n\s*){2,}", "\n\n", text)
    return text.strip()


def chunk_text(plain: str) -> list[dict]:
    """Split plain text into chunks keyed by 'Điều N.' markers.

    Returns list of dicts: {article_no, article_title, chapter_no, text}.
    Each Điều match starts a new chunk; chapter is inherited from the
    most-recent 'Chương X' marker preceding the chunk start.
    """
    if not plain:
        return []

    dieu_matches = list(RE_DIEU.finditer(plain))
    chuong_matches = list(RE_CHUONG.finditer(plain))

    chunks: list[dict] = []

    # preamble before first Điều
    if dieu_matches:
        pre = plain[: dieu_matches[0].start()].strip()
        if pre:
            chunks.append(
                {
                    "article_no": "",
                    "article_title": "",
                    "chapter_no": "",
                    "text": pre,
                }
            )
    else:
        # entire doc as a single chunk
        return [
            {
                "article_no": "",
                "article_title": "",
                "chapter_no": "",
                "text": plain.strip(),
            }
        ]

    for i, m in enumerate(dieu_matches):
        start = m.start()
        end = dieu_matches[i + 1].start() if i + 1 < len(dieu_matches) else len(plain)
        body = plain[start:end].strip()
        article_no = m.group(1)
        article_title = m.group(2).strip() if m.group(2) else ""

        # most recent chapter preceding this Điều
        chapter_no = ""
        for cm in reversed(chuong_matches):
            if cm.start() < start:
                chapter_no = cm.group(1).strip()
                break

        chunks.append(
            {
                "article_no": article_no,
                "article_title": article_title,
                "chapter_no": chapter_no,
                "text": body,
            }
        )

    return chunks


def make_chunk_id(doc_id: str, article_no: str, position: int, length: int) -> str:
    seed = f"{doc_id}|{article_no}|{position}|{length}"
    return str(uuid.uuid5(NS, seed))


# ---------- main ----------
def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare retrieval corpus from kb+hops/")
    ap.add_argument("--out", default=str(OUT_CSV), help="output CSV path")
    args = ap.parse_args()

    if not SRC_META.exists() or not SRC_CONTENT.exists():
        raise FileNotFoundError(
            f"Missing source files under {KB_BASE}. Pre-check must succeed first."
        )

    # ----- load metadata -----
    meta = pd.read_csv(SRC_META, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    meta.columns = [c.strip() for c in meta.columns]
    print(f"[load] metadata     : {len(meta)} rows x {meta.shape[1]} cols")

    # ----- load content (raise csv field limit before this in __main__) -----
    content = pd.read_csv(SRC_CONTENT, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    content.columns = [c.strip() for c in content.columns]
    print(f"[load] content      : {len(content)} rows x {content.shape[1]} cols")

    # existence of expected columns
    if "id" not in meta.columns:
        raise RuntimeError("metadata: 'id' column missing — pre-check violated")
    if "id" not in content.columns or "content_html" not in content.columns:
        raise RuntimeError("content: 'id' / 'content_html' columns missing")
    if "title" not in meta.columns or "loai_van_ban" not in meta.columns:
        raise RuntimeError("metadata: title / loai_van_ban missing")

    # ----- inner join on id -----
    merged = meta.merge(content[["id", "content_html"]], on="id", how="inner")
    print(f"[join] inner-join    : {len(merged)} docs (after join on id)")

    rows: list[dict] = []
    docs_no_html = 0
    docs_with_text = 0

    for _, doc in merged.iterrows():
        doc_id = str(doc["id"])
        raw_html = doc.get("content_html", "")
        if not raw_html or raw_html.strip() == "":
            docs_no_html += 1
            continue
        plain = strip_html(raw_html)
        if not plain:
            docs_no_html += 1
            continue
        docs_with_text += 1

        chunks = chunk_text(plain)
        for pos, c in enumerate(chunks):
            cid = make_chunk_id(
                doc_id=doc_id,
                article_no=c["article_no"],
                position=pos,
                length=len(c["text"]),
            )
            text = c["text"]
            article_title = c["article_title"]
            title = (doc.get("title") or "").strip()
            so_ky_hieu = (doc.get("so_ky_hieu") or "").strip()
            # source_file: đường dẫn tương đối từ PROJECT_ROOT tới file HTML gốc
            try:
                rel_source = SRC_CONTENT.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_source = f"../{SRC_CONTENT.name}"

            rows.append(
                {
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "text": text,
                    "source_file": rel_source,
                    "title": title,
                    "document_type": (doc.get("loai_van_ban") or "").strip(),
                    "chapter": c["chapter_no"],
                    "section": "",
                    "article": c["article_no"],
                    "article_title": article_title,
                    "clause": "",
                    "effective_date": (doc.get("ngay_co_hieu_luc") or "").strip(),
                    "status": (doc.get("tinh_trang_hieu_luc") or "").strip(),
                    "so_ky_hieu": so_ky_hieu,
                    "ngay_ban_hanh": (doc.get("ngay_ban_hanh") or "").strip(),
                    "co_quan_ban_hanh": (doc.get("co_quan_ban_hanh") or "").strip(),
                    "nguoi_ky": (doc.get("nguoi_ky") or "").strip(),
                }
            )

    if not rows:
        raise RuntimeError("No chunks produced — check regex / source encoding.")

    df = pd.DataFrame(rows)

    # ----- validation -----
    empty_text = ((df["text"].isna()) | (df["text"].str.strip() == "")).sum()
    dup = int(df["chunk_id"].duplicated().sum())
    if dup > 0:
        raise RuntimeError(
            f"chunk_id duplicated ({dup} rows). Seed includes position+length, "
            "this should not happen — inspect chunk_text()."
        )

    # ----- write -----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ----- stats -----
    print()
    print("=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    print(f"docs joined (metadata ∩ content)        : {len(merged)}")
    print(f"docs skipped (no usable text)            : {docs_no_html}")
    print(f"docs that produced chunks               : {docs_with_text}")
    print(f"total chunks                             : {len(df)}")
    print(f"unique document_id in chunks            : {df['document_id'].nunique()}")
    print(f"chunks with empty text                   : {empty_text}")
    print(f"duplicate chunk_id                       : {dup}")
    print(f"output                                   : {out_path}")
    print("=" * 60)

    # ----- 3 sample records -----
    print("\n3 sample records (head)\n")
    samples = df.head(3).to_dict(orient="records")
    for i, s in enumerate(samples, 1):
        snippet = s["text"][:240].replace("\n", " | ")
        print(f"--- sample #{i} ---")
        print(f"chunk_id    : {s['chunk_id']}")
        print(f"document_id : {s['document_id']}")
        print(f"title       : {s['title']}")
        print(f"so_ky_hieu  : {s['so_ky_hieu']}")
        print(f"document_type : {s['document_type']}")
        print(f"chapter / article : {s['chapter']!r} / {s['article']!r}")
        print(f"article_title  : {s['article_title'][:80]}")
        print(f"text [0:240]: {snippet}...")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
