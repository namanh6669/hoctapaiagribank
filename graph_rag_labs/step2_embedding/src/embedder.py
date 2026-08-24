"""Vector embedding for the chunks produced in Bước 1.

The default model is ``thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5`` — a
Vietnamese SBERT distilled from mmarco, with cosine-similarity training that
matches the Vietnamese legal/banking vocabulary well. The model is small
enough (~90 MB, 384-dim) to run on CPU in a few seconds.

Environment notes
-----------------

The module pins CPU execution so the demo works on machines without a GPU:

* ``device = "cpu"`` is passed to ``SentenceTransformer``.
* We force ``torch.set_num_threads`` from ``STEP2_THREADS`` (default 4) so
  students on laptops do not see their entire CPU pinned.
* No CUDA / MPS paths are exercised.

If you want a different model (e.g. ``intfloat/multilingual-e5-small``),
pass ``model_name=...`` to :func:`embed_chunks`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"
DEFAULT_THREADS = int(os.environ.get("STEP2_THREADS", "4"))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """Embeddings + side info, ready to be persisted or inspected."""

    chunk_ids: list[str]
    vectors: np.ndarray  # shape (N, dim), dtype float32
    model_name: str
    dim: int
    elapsed_seconds: float
    normalized: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_chunks(
    chunks: list[dict],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    normalize: bool = True,
    include_heading_path: bool = True,
    verbose: bool = True,
) -> EmbeddingResult:
    """Embed the chunks loaded from Bước 1.

    Parameters
    ----------
    chunks:
        The list returned by ``chunker.chunk_document`` and persisted to
        ``chunks.json``.
    model_name:
        HuggingFace repo id of the SentenceTransformer to load.
    batch_size:
        Number of chunks per encoder batch. Small batches keep memory low
        for CPU machines.
    normalize:
        If True, L2-normalise each vector (recommended for cosine sim).
    include_heading_path:
        Prepend the chunk's breadcrumb to the text before embedding. This
        gives the encoder the structural context ("Chương II / Điều 5")
        which improves retrieval on legal texts.
    """
    import torch  # noqa: WPS433 — imported lazily so the module imports cleanly
    from sentence_transformers import SentenceTransformer  # noqa: WPS433

    # ---- 1. CPU guard rails --------------------------------------------
    torch.set_num_threads(DEFAULT_THREADS)
    if verbose:
        print(f"  PyTorch          : {torch.__version__}")
        print(f"  CUDA khả dụng     : {torch.cuda.is_available()}")
        print(f"  Số thread CPU    : {torch.get_num_threads()}")

    # ---- 2. Load model on CPU -----------------------------------------
    if verbose:
        print(f"  Đang tải model   : {model_name}")
    model = SentenceTransformer(model_name, device="cpu")
    if verbose:
        # ``get_embedding_dimension`` is the modern name; the legacy
        # ``get_sentence_embedding_dimension`` raises a deprecation warning
        # in sentence-transformers >= 5.x.
        dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        print(f"  Kích thước vector : {dim_fn()}")
        print(f"  max_seq_length    : {model.max_seq_length}")

    # ---- 3. Build the text corpus --------------------------------------
    texts = [_build_text(c, include_heading=include_heading_path) for c in chunks]
    ids = [c["id"] for c in chunks]

    # ---- 4. Encode -----------------------------------------------------
    if verbose:
        print(f"  Số chunk cần nhúng: {len(texts)}")
        print(f"  batch_size        : {batch_size}")

    t0 = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0

    if verbose:
        per_chunk = elapsed / max(len(texts), 1) * 1000
        print(f"  Thời gian embed   : {elapsed:.2f}s ({per_chunk:.1f} ms / chunk)")
        print(f"  Shape output      : {vectors.shape}  dtype={vectors.dtype}")

    return EmbeddingResult(
        chunk_ids=ids,
        vectors=vectors,
        model_name=model_name,
        dim=int(vectors.shape[1]),
        elapsed_seconds=elapsed,
        normalized=normalize,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_embeddings(
    result: EmbeddingResult,
    *,
    chunks: list[dict],
    output_dir: Path,
    write_npz: bool = True,
    write_json_meta: bool = True,
) -> dict[str, Path]:
    """Persist vectors + per-chunk metadata to disk.

    Outputs
    -------
    embeddings.npz
        ``vectors`` (N, dim) float32 + ``ids`` (N,) string.
    embeddings_meta.json
        Per-chunk payload: ``id``, ``kind``, ``title``, ``heading_path``,
        ``parent_id``, ``next_id``, ``dim``, ``model``.
    embeddings_report.json
        Top-level summary used by the demo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    if write_npz:
        npz_path = output_dir / "embeddings.npz"
        np.savez_compressed(
            npz_path,
            vectors=result.vectors,
            ids=np.array(result.chunk_ids),
        )
        paths["npz"] = npz_path

    if write_json_meta:
        meta_path = output_dir / "embeddings_meta.json"
        by_id = {c["id"]: c for c in chunks}
        rows = []
        for cid in result.chunk_ids:
            c = by_id.get(cid, {})
            rows.append(
                {
                    "id": cid,
                    "kind": c.get("kind"),
                    "title": c.get("title"),
                    "heading_path": c.get("heading_path", []),
                    "parent_id": c.get("parent_id"),
                    "next_id": c.get("next_id"),
                    "dim": result.dim,
                    "model": result.model_name,
                    "normalized": result.normalized,
                }
            )
        meta_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["meta"] = meta_path

    report_path = output_dir / "embeddings_report.json"
    report_path.write_text(
        json.dumps(
            {
                "model": result.model_name,
                "dim": result.dim,
                "n_vectors": int(result.vectors.shape[0]),
                "dtype": str(result.vectors.dtype),
                "normalized": result.normalized,
                "elapsed_seconds": round(result.elapsed_seconds, 4),
                "ms_per_chunk": round(result.elapsed_seconds / max(len(result.chunk_ids), 1) * 1000, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["report"] = report_path
    return paths


def load_embeddings(npz_path: Path) -> tuple[list[str], np.ndarray]:
    """Read back the persisted vectors + ids."""
    data = np.load(npz_path, allow_pickle=False)
    return list(data["ids"]), data["vectors"]


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def _build_text(chunk: dict, *, include_heading: bool) -> str:
    """Compose the text fed to the encoder.

    Containers (chapter/article/...) carry no body text of their own, so
    embedding them helps structural retrieval: the encoder still learns the
    location of every clause.
    """
    parts: list[str] = []
    if include_heading:
        path = chunk.get("heading_path") or []
        if path:
            parts.append(" | ".join(path))
    body = (chunk.get("text") or "").strip()
    title = (chunk.get("title") or "").strip()
    if body:
        parts.append(body)
    elif title:
        parts.append(title)
    return "\n".join(parts) if parts else title or ""


# ---------------------------------------------------------------------------
# Similarity helpers (used by the demo)
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two equally-L2-normalised matrices."""
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def topk_similar(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    *,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, indices)`` of the top-k most similar rows."""
    sims = cosine_similarity(query_vec, matrix)[0]
    order = np.argsort(-sims)[:k]
    return sims[order], order