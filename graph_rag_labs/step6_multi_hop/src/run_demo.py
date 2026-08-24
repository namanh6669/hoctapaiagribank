"""Demo entrypoint for Bước 6 — Multi-hop retrieval.

Run from ``step6_multi_hop/``:

    python -m src.run_demo
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import load_settings
from .retriever import MultiHopRetriever, QueryResult


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT_DIR = PROJECT / "output"


# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------


DEMO_QUERIES: list[str] = [
    "Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ",
    "Thông tư 06/2023 sửa đổi những điều nào của Thông tư 39/2016?",
    "Căn cứ pháp lý để ban hành Thông tư 02/2023/TT-NHNN",
    "Trách nhiệm của Ngân hàng Nhà nước trong việc quản lý cho vay",
]


def main() -> None:
    print("=" * 78)
    print("BƯỚC 6 — MULTI-HOP VECTOR + GRAPH RETRIEVAL")
    print("=" * 78)

    settings = load_settings()
    print(f"\n[1] Kết nối Neo4j")
    print(f"    URI          : {settings.uri}")
    print(f"    user         : {settings.user}")
    print(f"    database     : {settings.database}")
    print(f"    vector_index : {settings.vector_index}")

    with MultiHopRetriever() as retriever:
        # ---- 2. Run each query with multiple hop settings -----------
        print("\n" + "-" * 78)
        print("[2] TRUY VẤN ĐA BƯỚC (vector + CAN_CU/THAY_THE/HOP_NHAT)")
        print("-" * 78)

        all_results = []
        for query in DEMO_QUERIES:
            for hops in (0, 1, 2):
                print(f"\n  Query: {query!r}")
                print(f"  num_hops={hops}")
                result = retriever.search(query, top_k=3, num_hops=hops)
                _print_result(result)
                all_results.append({"query": query, "hops": hops, "result": _result_to_dict(result)})

    # ---- 3. Persist -------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "multi_hop_results.json"
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[3] ĐÃ GHI: {out_path}")


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _print_result(result: QueryResult) -> None:
    if not result.chunks:
        print("    (no chunks returned)")
        return

    print(f"  → {len(result.documents)} document(s) | {len(result.chunks)} chunk(s)")
    print()
    print("  Documents theo graph path:")
    for d in result.documents[:6]:
        marker = "[ROOT]" if d.is_root else "      "
        path = " -> ".join(d.path[:3]) + (" -> ..." if len(d.path) > 3 else "")
        via = f" via {d.via_relationship}" if d.via_relationship else ""
        print(f"    {marker} {d.doc_id:<20} hops={d.hops} score={d.score:.3f}{via}")
        print(f"             {path}")
        print(f"             {d.title[:60]!r}")

    print()
    print("  Top chunks:")
    for c in result.chunks[:5]:
        snippet = (c.text or c.title).replace("\n", " ")[:80]
        print(f"    [{c.source:<10}] score={c.score:+.3f} [{c.kind}] {snippet}…")


def _result_to_dict(result: QueryResult) -> dict:
    return {
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "kind": c.kind,
                "title": c.title,
                "text": c.text,
                "heading_path": c.heading_path,
                "parent_doc_id": c.parent_doc_id,
                "score": c.score,
                "source": c.source,
                "via_doc": c.via_doc,
            }
            for c in result.chunks
        ],
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "doc_type": d.doc_type,
                "is_root": d.is_root,
                "score": d.score,
                "hops": d.hops,
                "via_relationship": d.via_relationship,
                "path": d.path,
            }
            for d in result.documents
        ],
    }


if __name__ == "__main__":
    main()