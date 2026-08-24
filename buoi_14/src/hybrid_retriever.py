"""Hybrid retrieval cho buoi_14 — kết hợp BM25 + Dense bằng Reciprocal Rank Fusion.

Tái sử dụng BM25Retriever + DenseRetriever ở cùng `df` để chắc chắn chỉ search
một corpus duy nhất. Hai retriever đều tự cache/load (BM25 build mỗi lần ~0.2s,
Dense cache hit ~1s).

RRF (Cormack et al. 2009):
    score(c) = Σ_r  1 / (k_rrf + rank_r(c))
trong đó `k_rrf = 60` theo mặc định. Ứng viên xuất hiện ở 1 retriever vẫn có
điểm (rank_r của retriever còn lại = ∞ → 0 đóng góp), không bị loại.

Output schema (xem yêu cầu prompt):
    final_rank, chunk_id, document_id,
    bm25_rank, dense_rank, rrf_score,
    text, citation
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .common import make_citation
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever, DEFAULT_MODEL

DEFAULT_K_RRF = 60


@dataclass
class HybridHit:
    final_rank: int
    chunk_id: str
    document_id: str
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float
    text: str
    citation: str


class HybridRetriever:
    def __init__(
        self,
        df: pd.DataFrame,
        k_rrf: int = DEFAULT_K_RRF,
        # dense_kwargs
        model_name: str = DEFAULT_MODEL,
        dense_kwargs: dict | None = None,
    ):
        if "chunk_id" not in df.columns:
            raise ValueError("HybridRetriever: thiếu cột 'chunk_id'")
        if df["chunk_id"].duplicated().any():
            raise ValueError("HybridRetriever: chunk_id phải duy nhất trong corpus")

        # Cùng df -> chung chunk_id set, đảm bảo không search 2 corpus khác nhau
        self.df = df.reset_index(drop=True)
        self.k_rrf = k_rrf

        self.bm25 = BM25Retriever(self.df)
        self.dense = DenseRetriever(self.df, model_name=model_name, **(dense_kwargs or {}))

        # Index nhanh chunk_id -> row
        self._rows_by_id = {
            str(cid): row for cid, row in zip(self.df["chunk_id"], self.df.to_dict("records"))
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:
        if not query or not query.strip():
            return []
        if candidate_k < top_k:
            raise ValueError("candidate-k phải >= top-k")

        # 1. Lấy candidate set từ 2 retriever (cùng corpus)
        bm_results = self.bm25.search(query, top_k=candidate_k)
        dn_results = self.dense.search(query, top_k=candidate_k)

        bm_rank: dict[str, int] = {r["chunk_id"]: r["rank"] for r in bm_results}
        dn_rank: dict[str, int] = {r["chunk_id"]: r["rank"] for r in dn_results}

        # 2. Hợp nhất theo chunk_id, không duplicate (dict keys)
        all_ids = set(bm_rank) | set(dn_rank)
        if not all_ids:
            return []

        # 3. Tính RRF
        scored: list[tuple[str, float, int | None, int | None]] = []
        for cid in all_ids:
            rrf = 0.0
            bmr = bm_rank.get(cid)
            dnr = dn_rank.get(cid)
            if bmr is not None:
                rrf += 1.0 / (self.k_rrf + bmr)
            if dnr is not None:
                rrf += 1.0 / (self.k_rrf + dnr)
            scored.append((cid, rrf, bmr, dnr))

        # 4. Sắp xếp: RRF giảm dần, tie-break bằng chunk_id để ổn định
        scored.sort(key=lambda x: (-x[1], x[0]))

        # 5. Lấy top_k + format
        out: list[dict] = []
        for final_rank, (cid, rrf, bmr, dnr) in enumerate(scored[:top_k], 1):
            row = self._rows_by_id[cid]
            text = row.get("text", "") or ""
            doc_id = row.get("document_id", "") or ""

            # make_citation nhận dict/Series đều được
            from .common import make_citation as _make_citation
            citation = _make_citation(row)

            out.append(
                {
                    "final_rank": final_rank,
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "bm25_rank": bmr,           # None nếu retriever kia không có
                    "dense_rank": dnr,
                    "rrf_score": rrf,
                    "text": text,
                    "citation": citation,
                }
            )
        return out
