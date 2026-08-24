"""Reranker cho buoi_14 — Cross-Encoder trên top (Hybrid) candidates.

Nguyên tắc:
- Không rerank toàn corpus — chỉ điểm lại candidate từ Hybrid Retriever.
- Trước khi tải model, kiểm tra cache và in thông báo kích thước.
- Nếu model không tải được / torch lỗi / OOM ... → chuyển FALLBACK rõ ràng,
  KHÔNG giả vờ rằng reranking đã chạy.

Default model:
    BAAI/bge-reranker-v2-m3  (multilingual, ~2.3 GB, chất lượng cao)

Lưu cache HuggingFace trong project: env HF_HOME đã được dense_retriever set
mặc định về buoi_14/cache/huggingface, nên reranker cũng hưởng lợi.
"""

from __future__ import annotations

import os
from pathlib import Path

# Ép HF_HOME nằm trong project trước khi import sentence_transformers
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parents[1] / "cache" / "huggingface"),
)

# Silence transformers' docstring-validation noise — xem dense_retriever.py
# (transformers v5+ in [ERROR] qua print(), không qua logger).
import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.models").setLevel(logging.CRITICAL)

from ._quiet_transformers import install_quiet_print_once
install_quiet_print_once()

import numpy as np

from .common import PROJECT_ROOT
from .hybrid_retriever import HybridRetriever

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# Rough size hint để cảnh báo trước khi tải (chỉ là ước lượng)
_MODEL_SIZE_HINT = {
    "BAAI/bge-reranker-v2-m3": "~2.3 GB",
    "BAAI/bge-reranker-base": "~1.1 GB",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "~470 MB",
    "cross-encoder/msmarco-MiniLM-L-6-v2": "~90 MB (English only — không khuyến nghị cho VN)",
}


def _hf_is_cached(model_name: str) -> bool:
    safe = "models--" + model_name.replace("/", "--")
    for root in (
        Path(os.environ.get("HF_HOME", "")),
        Path.home() / ".cache" / "huggingface",
    ):
        if not root:
            continue
        if (Path(root) / "hub" / safe).exists():
            return True
    return False


def _announce(model_name: str) -> None:
    size = _MODEL_SIZE_HINT.get(model_name, "(kích thước chưa rõ)")
    cached = _hf_is_cached(model_name)
    print(f"[rerank] model      : {model_name}")
    print(f"[rerank] size hint  : {size}")
    print(f"[rerank] cache dir  : {os.environ.get('HF_HOME')}/hub/models--{model_name.replace('/','--')}")
    print(f"[rerank] cache hit? : {'YES (sẽ load offline)' if cached else 'NO  (sẽ tải từ HuggingFace)'}")
    if not cached:
        print(f"[rerank] ----------------------------------------------------------------")
        print(f"[rerank] CẢNH BÁO: model chưa có trong cache, sẽ tải về {size}.")
        print(f"[rerank] Để huỷ: Ctrl+C trong vài giây tới.")
        print(f"[rerank] ----------------------------------------------------------------")


# ---------- Base ----------
class Reranker:
    name: str = "base"
    is_fallback: bool = False

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        raise NotImplementedError


# ---------- Cross-Encoder ----------
class CrossEncoderReranker(Reranker):
    is_fallback = False

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, max_length: int = 512):
        self.model_name = model_name
        _announce(model_name)

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except Exception as e:
            raise RuntimeError(
                f"Không import được torch / sentence_transformers: {e}"
            ) from e

        # Chọn device tốt nhất
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        try:
            self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        except Exception as e:
            raise RuntimeError(
                f"Không load được Cross-Encoder '{model_name}': {e}"
            ) from e

        self.device = device
        self.name = f"CrossEncoder:{model_name}"
        print(f"[rerank] loaded OK   on device={device}, max_length={max_length}")

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, (c.get("text") or "")[:5000]) for c in candidates]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        # Inject rerank_score vào candidate dict (không làm mất hybrid_score)
        for c, s in zip(candidates, [float(x) for x in scores]):
            c["rerank_score"] = s
        # Sort giảm dần theo rerank_score
        sorted_cands = sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]
        out = []
        for rank, c in enumerate(sorted_cands, 1):
            out.append(
                {
                    "final_rank": rank,                          # post-rerank position
                    "hybrid_rank": c.get("hybrid_rank"),          # pre-rerank (1..candidate_k)
                    "hybrid_score": c.get("rrf_score"),
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "rerank_score": c["rerank_score"],
                    "text": c.get("text", ""),
                    "citation": c.get("citation", ""),
                    "is_fallback": False,
                }
            )
        return out


# ---------- Fallback (identity) ----------
class FallbackReranker(Reranker):
    name = "FALLBACK identity (không neural reranker — dùng nguyên hybrid_score)"
    is_fallback = True

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        # Identity: giữ thứ tự theo rrf_score của hybrid
        sorted_cands = sorted(
            candidates,
            key=lambda x: -(x.get("rrf_score") or x.get("hybrid_score") or 0.0),
        )[:top_k]
        out = []
        for rank, c in enumerate(sorted_cands, 1):
            hybrid_score = c.get("rrf_score") or c.get("hybrid_score") or 0.0
            out.append(
                {
                    "final_rank": rank,
                    "hybrid_rank": c.get("hybrid_rank"),
                    "hybrid_score": hybrid_score,
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "rerank_score": hybrid_score,  # identity (mark FALLBACK bằng meta)
                    "text": c.get("text", ""),
                    "citation": c.get("citation", ""),
                    "is_fallback": True,
                }
            )
        return out


# ---------- Pipeline ----------
class RerankPipeline:
    """Hybrid retrieval → (chọn) reranker → top-k."""

    def __init__(
        self,
        df,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        force_fallback: bool = False,
        hybrid_kwargs: dict | None = None,
    ):
        self.hybrid = HybridRetriever(df, **(hybrid_kwargs or {}))
        self.force_fallback = force_fallback
        self.reranker: Reranker

        if force_fallback:
            print("[rerank] FORCE FALLBACK theo yêu cầu (--fallback).")
            self.reranker = FallbackReranker()
        else:
            try:
                self.reranker = CrossEncoderReranker(model_name=rerank_model)
            except Exception as e:
                print(f"[rerank] ! Cross-Encoder KHÔNG khả dụng: {e}")
                print("[rerank] ! Chuyển sang FALLBACK identity (không neural reranker).")
                self.reranker = FallbackReranker()

    # trả về (candidates, reranked)
    def search(
        self, query: str, top_k: int = 5, candidate_k: int = 20
    ) -> tuple[list[dict], list[dict], dict]:
        if not query or not query.strip():
            return [], [], {"method": self.reranker.name, "is_fallback": self.reranker.is_fallback}

        hybrid_candidates = self.hybrid.search(query, top_k=candidate_k, candidate_k=candidate_k)
        # hybrid.search trả: final_rank, chunk_id, document_id, bm25_rank, dense_rank,
        # rrf_score, text, citation.
        # Chuẩn hoá: GIỮ final_rank (= vị trí trong hybrid) để in BEFORE table,
        # đồng thời lưu vào hybrid_rank cho reranker truy ngược.
        normalized = []
        for c in hybrid_candidates:
            normalized.append(
                {
                    "final_rank": c.get("final_rank"),  # vị trí trong hybrid
                    "hybrid_rank": c.get("final_rank"),
                    "rrf_score": c.get("rrf_score"),
                    "bm25_rank": c.get("bm25_rank"),
                    "dense_rank": c.get("dense_rank"),
                    "chunk_id": c.get("chunk_id"),
                    "document_id": c.get("document_id"),
                    "text": c.get("text", ""),
                    "citation": c.get("citation", ""),
                }
            )

        reranked = self.reranker.rerank(query, normalized, top_k=top_k)
        meta = {
            "method": self.reranker.name,
            "is_fallback": self.reranker.is_fallback,
        }
        return normalized, reranked, meta
