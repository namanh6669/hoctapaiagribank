"""Demo entrypoint for Bước 2.

Reads the chunks produced by Bước 1, embeds them with
``thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5`` on CPU, and prints:

* model + device diagnostics (so the student sees the CPU pin working),
* batch stats (vector shape, dtype, ms / chunk),
* a few representative cosine-similarity lookups between chunks,
* a small semantic-search demo on a hand-picked Vietnamese query.

Run from ``step2_embedding/``:

    python -m src.run_demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .embedder import (
    DEFAULT_MODEL,
    embed_chunks,
    save_embeddings,
    topk_similar,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# Default input = Bước 1 output. Allow override via argv[1].
DEFAULT_INPUT = PROJECT.parent / "step1_chunking" / "output" / "chunks.json"
OUTPUT_DIR = PROJECT / "output"
SAMPLE_DIR = OUTPUT_DIR / "sample"


# ---------------------------------------------------------------------------
# Demo queries — picked from the same Thông tư so the student sees the
# encoder doing its job on Vietnamese legal text.
# ---------------------------------------------------------------------------

DEMO_QUERIES: list[str] = [
    "Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ",
    "Khách hàng nào được giữ nguyên nhóm nợ?",
    "Thời hạn cơ cấu lại tối đa là bao lâu?",
]


def run(input_path: Path = DEFAULT_INPUT) -> None:
    print("=" * 78)
    print("BƯỚC 2 — TẠO VECTOR NHÚNG (EMBEDDING)")
    print("=" * 78)

    if not input_path.exists():
        sys.exit(
            f"[LỖI] Không tìm thấy {input_path}. "
            "Hãy chạy Bước 1 trước (python -m src.run_demo ở step1_chunking/)."
        )

    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"\n[1] Đọc chunks từ : {input_path.name}")
    print(f"    Tổng số chunk   : {len(chunks)}")

    # ---- 2. Embedding ---------------------------------------------------
    print("\n" + "-" * 78)
    print("[2] EMBEDDING (CPU-only)")
    print("-" * 78)
    result = embed_chunks(chunks, model_name=DEFAULT_MODEL, batch_size=32)

    # ---- 3. Persist ----------------------------------------------------
    print("\n" + "-" * 78)
    print("[3] LƯU KẾT QUẢ")
    print("-" * 78)
    paths = save_embeddings(result, chunks=chunks, output_dir=OUTPUT_DIR)
    for label, path in paths.items():
        print(f"  {label:<8} -> {path}")

    # ---- 4. Vectors sanity-check --------------------------------------
    print("\n" + "-" * 78)
    print("[4] KIỂM TRA VECTOR (5 CHUNK ĐẦU)")
    print("-" * 78)
    _print_vector_samples(chunks, result)

    # ---- 5. Cosine-similarity lookups ---------------------------------
    print("\n" + "-" * 78)
    print("[5] ĐỘ TƯƠNG ĐỒNG COSINE GIỮA CÁC CHUNK")
    print("-" * 78)
    _print_self_similarity(chunks, result)

    # ---- 6. Semantic search demo ---------------------------------------
    print("\n" + "-" * 78)
    print("[6] TÌM KIẾM NGỮ NGHĨA (CÂU TRUY VẤN TIẾNG VIỆT)")
    print("-" * 78)
    _print_semantic_search(chunks, result, queries=DEMO_QUERIES, k=3)

    # ---- 7. Sample JSON for downstream consumers ----------------------
    sample_path = SAMPLE_DIR / "embeddings_sample.json"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(_sample_metadata(chunks, result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[7] ĐÃ GHI sample metadata -> {sample_path}")


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------


def _print_vector_samples(chunks: list[dict], result) -> None:
    by_id = {c["id"]: c for c in chunks}
    print(f"  Tổng vector : {result.vectors.shape[0]} × {result.vectors.shape[1]}")
    print(f"  dtype       : {result.vectors.dtype}")
    print(f"  L2-normalised : {result.normalized}")
    print()
    print("  5 vector đầu (8 chiều đầu + L2-norm):")
    for i in range(min(5, len(result.chunk_ids))):
        cid = result.chunk_ids[i]
        c = by_id[cid]
        v = result.vectors[i]
        preview = "[" + ", ".join(f"{x:+.4f}" for x in v[:8]) + ", …]"
        norm = float(np.linalg.norm(v))
        print(f"    [{i}] {cid}  norm={norm:.4f}")
        print(f"        kind={c['kind']:<10} title={c['title'][:55]!r}")
        print(f"        vec[:8] = {preview}")


def _print_self_similarity(chunks: list[dict], result) -> None:
    """Print cosine sim of the first paragraph with the 5 most similar chunks."""
    by_id = {c["id"]: c for c in chunks}

    # Pick the first paragraph (skip the document root, which has no text).
    first_para_idx = next(
        (i for i, c in enumerate(chunks) if c.get("text")),
        None,
    )
    if first_para_idx is None:
        print("  (Không có chunk văn bản để so sánh)")
        return

    query = result.vectors[first_para_idx : first_para_idx + 1]
    scores, idxs = topk_similar(query, result.vectors, k=6)
    query_chunk = chunks[first_para_idx]

    print(f"  Query chunk : {query_chunk['id']}")
    print(f"  kind/title  : {query_chunk['kind']} | {query_chunk['title'][:60]!r}")
    print(f"  heading     : {' > '.join(query_chunk.get('heading_path', []))}")
    print()
    print("  Top-5 chunk giống nhất:")
    for rank, (s, i) in enumerate(zip(scores[:5], idxs[:5]), start=1):
        if i == first_para_idx:
            label = "(chính nó)"
        else:
            label = ""
        c = chunks[i]
        print(f"    #{rank}  score={s:+.4f}  {label}")
        print(f"         {c['id']}  {c['kind']:<10}  {c['title'][:55]!r}")
        print(f"         heading: {' > '.join(c.get('heading_path', []))}")


def _print_semantic_search(
    chunks: list[dict],
    result,
    *,
    queries: list[str],
    k: int = 3,
) -> None:
    """Run a few Vietnamese queries through the same encoder and print hits."""
    # Lazy import to keep the import cost out of the cold path.
    from sentence_transformers import SentenceTransformer

    print(f"  Model truy vấn: {result.model_name}")
    model = SentenceTransformer(result.model_name, device="cpu")

    for q in queries:
        print()
        print(f"  Query: {q!r}")
        q_vec = model.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=result.normalized,
        ).astype(np.float32)
        scores, idxs = topk_similar(q_vec, result.vectors, k=k)
        for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
            c = chunks[i]
            print(f"    #{rank}  score={s:+.4f}  [{c['kind']}] {c['title'][:55]!r}")
            print(f"         heading: {' > '.join(c.get('heading_path', []))}")


def _sample_metadata(chunks: list[dict], result) -> dict:
    """Tiny JSON snapshot: 3 vectors + their chunk metadata."""
    rows = []
    for i in range(min(3, len(result.chunk_ids))):
        cid = result.chunk_ids[i]
        c = next(x for x in chunks if x["id"] == cid)
        rows.append(
            {
                "id": cid,
                "kind": c["kind"],
                "title": c["title"],
                "heading_path": c.get("heading_path", []),
                "vec_first8": [round(float(x), 6) for x in result.vectors[i, :8].tolist()],
                "l2_norm": round(float(np.linalg.norm(result.vectors[i])), 6),
            }
        )
    return {
        "model": result.model_name,
        "dim": result.dim,
        "normalized": result.normalized,
        "samples": rows,
    }


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    run(arg)