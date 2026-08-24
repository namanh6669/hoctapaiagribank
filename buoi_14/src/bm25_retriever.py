"""BM25 lexical retrieval cho buoi_14.

Tokenization giữ được:
- mã văn bản dạng "32/2024/QH15", "01/2014/TT-NHNN" (dấu `/` và `-` dính vào token);
- số điều "5" riêng một token;
- từ tiếng Việt có dấu (Unicode NFC + lowercase).

Mỗi dấu chấm/phẩy/xuống dòng sẽ tách token. Chuẩn NFKC + lowercase để các biến
thể unicode dồn về một dạng.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .common import format_result

# Bao gồm chữ cái Latin cơ bản + toàn bộ dải tiếng Việt có dấu (À-ỹ = U+00C0..U+1EF9).
_TOKEN_RE = re.compile(
    r"[À-ỹA-Za-z0-9]+(?:[/\-\.][À-ỹA-Za-z0-9]+)*",
    re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """Token đơn giản cho BM25: NFC + lowercase + giữ mã có dấu `/` `-` `.`."""
    if not text:
        return []
    text = unicodedata.normalize("NFC", text).lower()
    return [tok for tok in _TOKEN_RE.findall(text) if tok]


class BM25Retriever:
    """BM25Okapi trên toàn bộ chunk corpus. Index build 1 lần, search nhiều lần."""

    def __init__(self, df: pd.DataFrame):
        if "text" not in df.columns:
            raise ValueError("BM25Retriever: thiếu cột 'text'")
        self.df = df.reset_index(drop=True)
        texts = self.df["text"].fillna("").astype(str).tolist()
        self._tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokens)
        self.method = "BM25"

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # Stable sort: tie-break bằng thứ tự xuất hiện trong corpus
        idx_sorted = np.argsort(-scores, kind="stable")
        idx_sorted = [int(i) for i in idx_sorted if scores[int(i)] > 0][:top_k]

        out = []
        for rank, i in enumerate(idx_sorted, 1):
            row = self.df.iloc[i]
            out.append(format_result(rank, row, scores[i], self.method))
        return out
