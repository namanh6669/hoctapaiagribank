"""Query-side embedding for the multi-hop retriever.

The encoder is the same Vietnamese MSMARCO-MiniLM model used in Bước 2,
so chunks and queries live in the same vector space. CPU-only.
"""
from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"


class QueryEmbedder:
    """Lazy wrapper around the SentenceTransformer so importing this module
    doesn't pay the model-load cost."""

    def __init__(self, model_name: str = DEFAULT_MODEL, threads: int = 4) -> None:
        self.model_name = model_name
        self.threads = threads
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            torch.set_num_threads(self.threads)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, query: str) -> list[float]:
        model = self._load()
        vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vec.astype("float32").tolist()