"""Dense retrieval cho buoi_14 — multilingual-e5-base.

Tiêu chí lựa chọn model:
- Hỗ trợ tiếng Việt (multilingual).
- Đủ nhỏ để chạy ổn trên máy CPU/MPS (Apple Silicon).
- Có cache mạnh mẽ trên Hugging Face.
- Phù hợp cả retrieval asymmetric (passage / query khác prefix).

Cache embeddings tại buoi_14/cache/dense_index/<model_safe>/ bao gồm:
- embeddings.npy : mảng float32 shape (N, D) đã L2-normalized.
- meta.json      : model, dim, n_chunks, signature, created_at.

Cache tự động vô hiệu nếu:
- model name đổi;
- số chunk trong corpus khác;
- danh sách chunk_id theo thứ tự khác (signature md5).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ép HuggingFace cache nằm trong project (không leak ra ngoài buoi_14)
_DEFAULT_HF_HOME = Path(__file__).resolve().parents[1] / "cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))

# Silence transformers' docstring-validation noise (`[ERROR] X is part of Y but
# not documented`) — đây là log nội bộ của transformers v5+ quét image processor
# kwargs của các vision submodule ta không dùng. Logging không đủ vì
# transformers dùng print() thẳng từ auto_docstring.py — phải patch builtins.print.
import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.models").setLevel(logging.CRITICAL)

from ._quiet_transformers import install_quiet_print_once
install_quiet_print_once()

from sentence_transformers import SentenceTransformer

from .common import DEFAULT_CACHE_ROOT, format_result

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


def _safe_name(model: str) -> str:
    return model.replace("/", "__").replace("\\", "__")


def _signature(df: pd.DataFrame) -> str:
    """Hash order-sensitive của các chunk_id. Dùng làm cache key."""
    ids = df["chunk_id"].astype(str).tolist()
    return hashlib.md5("|".join(ids).encode("utf-8")).hexdigest()


@dataclass
class _Cache:
    embeddings: np.ndarray
    meta: dict


def _cache_paths(cache_root: Path, model: str) -> tuple[Path, Path]:
    base = Path(cache_root) / "dense_index" / _safe_name(model)
    return base, base


def _try_load(cache_dir: Path, df: pd.DataFrame, model: str) -> _Cache | None:
    emb_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "meta.json"
    if not (emb_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = np.load(emb_path)
    except Exception:
        return None
    sig = _signature(df)
    if meta.get("model") != model:
        return None
    if meta.get("n_chunks") != len(df):
        return None
    if meta.get("signature") != sig:
        return None
    if emb.shape[0] != len(df):
        return None
    return _Cache(embeddings=emb.astype("float32"), meta=meta)


def _save(cache_dir: Path, embeddings: np.ndarray, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings.astype("float32"))
    (cache_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class DenseRetriever:
    """Cosine retrieval bằng SentenceTransformer + numpy inner-product."""

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = DEFAULT_MODEL,
        cache_root: Path | None = None,
        batch_size: int = 16,
        show_progress: bool = False,
    ):
        if "text" not in df.columns:
            raise ValueError("DenseRetriever: thiếu cột 'text'")
        self.df = df.reset_index(drop=True)
        self.model_name = model_name
        self.method = "DENSE"

        cache_root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
        _, cache_dir = _cache_paths(cache_root, model_name)

        # Khởi tạo model trên device tốt nhất có sẵn
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        # Văn bản pháp luật VN có thể dài; SentenceTransformer tự truncate ở 512 token,
        # nhưng set rõ để chủ động.
        self.model.max_seq_length = 512
        self.device = device

        # Thử cache
        cached = _try_load(cache_dir, self.df, model_name)
        if cached is not None:
            self.embeddings = cached.embeddings
            print(
                f"[dense] cache hit  : {cache_dir}  "
                f"(n={self.embeddings.shape[0]}, dim={self.embeddings.shape[1]})"
            )
            return

        # Build embeddings
        print(
            f"[dense] building   : {len(self.df)} chunks · model={model_name} · device={device}"
        )
        texts = self.df["text"].fillna("").astype(str).tolist()
        # E5 mong đợi "passage: " cho tài liệu; "query: " cho câu hỏi
        e5_texts = [(E5_PASSAGE_PREFIX + t) if t else E5_PASSAGE_PREFIX for t in texts]

        t0 = time.time()
        embs = self.model.encode(
            e5_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalized -> cosine = inner product
        )
        dt = time.time() - t0
        print(f"[dense] encoded in {dt:.1f}s, shape={embs.shape}")

        self.embeddings = embs.astype("float32")
        meta = {
            "model": model_name,
            "n_chunks": int(embs.shape[0]),
            "dim": int(embs.shape[1]),
            "dtype": "float32",
            "signature": _signature(self.df),
            "device": device,
            "batch_size": batch_size,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save(cache_dir, self.embeddings, meta)
        print(f"[dense] cached to {cache_dir}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        # E5 prefix cho query
        q_e5 = E5_QUERY_PREFIX + query
        q_emb = self.model.encode(
            [q_e5],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype("float32")

        scores = self.embeddings @ q_emb  # cosine vì đã L2-normalized
        idx_sorted = np.argsort(-scores, kind="stable")
        idx_sorted = [int(i) for i in idx_sorted[:top_k]]

        out = []
        for rank, i in enumerate(idx_sorted, 1):
            row = self.df.iloc[i]
            out.append(format_result(rank, row, scores[i], self.method))
        return out
