"""Tiện ích dùng chung cho các retriever trong buoi_14.

Cung cấp:
- Đường dẫn PROJECT_ROOT và corpus mặc định.
- load_corpus(): đọc data/processed/chunks_normalized.csv.
- make_citation(row): sinh citation `[title (<so_ky_hieu>) | Điều N | chunk_id]`
  hoặc `[title (<so_ky_hieu>) | chunk_id]` cho preamble.
- format_result(): đóng gói kết quả retrieval về schema chuẩn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "processed" / "chunks_normalized.csv"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cache"

CITATION_SEP = " | "

# Cột bắt buộc trong schema ra (xem yêu cầu prompt)
RESULT_COLUMNS = [
    "rank",
    "chunk_id",
    "document_id",
    "text",
    "retrieval_score",
    "retrieval_method",
    "citation",
]


def load_corpus(csv_path: Path | None = None) -> pd.DataFrame:
    """Đọc corpus đã chuẩn hoá. Không sửa, không ghi đè."""
    csv_path = Path(csv_path) if csv_path else DEFAULT_CORPUS
    if not csv_path.exists():
        raise FileNotFoundError(f"Corpus not found at {csv_path}. Run `prepare_corpus.py` first.")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df


def _shorten(s: str, max_len: int = 120) -> str:
    s = s.strip()
    return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "…"


def make_citation(row: pd.Series | dict) -> str:
    """Sinh citation lấy từ metadata thật (title, so_ky_hieu, article, chunk_id).
    Không bịa tên — nếu thiếu thì bỏ phần đó.
    """
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = lambda k: row.get(k, "")  # noqa: E731

    title = (str(get("title") or "")).strip()
    so_ky_hieu = (str(get("so_ky_hieu") or "")).strip()
    doc_id = (str(get("document_id") or "")).strip()
    article = (str(get("article") or "")).strip()
    cid = (str(get("chunk_id") or "")).strip()

    head = title
    if so_ky_hieu and so_ky_hieu not in head:
        head = f"{head} ({so_ky_hieu})" if head else so_ky_hieu
    if not head:
        head = doc_id or "(không rõ)"

    head = _shorten(head, 140)

    parts = [head]
    if article:
        parts.append(f"Điều {article}")
    parts.append(cid)
    return "[" + CITATION_SEP.join(parts) + "]"


def format_result(rank: int, row: pd.Series, score: float, method: str) -> dict:
    return {
        "rank": int(rank),
        "chunk_id": row["chunk_id"],
        "document_id": row["document_id"],
        "text": row["text"],
        "retrieval_score": float(score),
        "retrieval_method": method,
        "citation": make_citation(row),
    }


def results_to_table(results: Iterable[dict]) -> str:
    """In gọn kết quả thành bảng text cho stdout."""
    rows = list(results)
    if not rows:
        return "(no results)"
    lines = []
    for r in rows:
        snippet = (str(r.get("text") or "")[:200]).replace("\n", " ").replace("|", "/")
        lines.append(
            f"  rank={r['rank']:>2d}  score={r['retrieval_score']:.4f}  "
            f"{r['citation']}\n"
            f"    text: {snippet}…"
        )
    return "\n".join(lines)
