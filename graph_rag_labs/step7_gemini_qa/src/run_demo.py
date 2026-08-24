"""Demo entrypoint for Bước 7 — Multi-hop + Gemini QA.

Pipeline:

    query (vi)
      → step6 MultiHopRetriever → QueryResult (chunks + documents)
      → prompt_builder.build_messages(query, result)
      → GeminiClient.generate(messages)
      → answer + citations

Run from ``step7_gemini_qa/``:

    python -m src.run_demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make step6 importable as a sibling package.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT.parent))  # graph_rag_labs/

from step6_multi_hop.src.retriever import MultiHopRetriever
from step7_gemini_qa.src.config import load_settings
from step7_gemini_qa.src.gemini_client import GeminiClient
from step7_gemini_qa.src.prompt_builder import build_messages


OUTPUT_DIR = PROJECT / "output"


DEMO_QUERIES: list[str] = [
    "Điều kiện để tổ chức tín dụng được cơ cấu lại thời hạn trả nợ là gì?",
    "Thông tư 06/2023 sửa đổi những điều nào của Thông tư 39/2016?",
    "Căn cứ pháp lý để ban hành Thông tư 02/2023/TT-NHNN gồm những văn bản nào?",
    "Trách nhiệm của Ngân hàng Nhà nước trong quản lý cho vay theo Thông tư 02/2023?",
]


def main() -> None:
    print("=" * 78)
    print("BƯỚC 7 — MULTI-HOP + GEMINI QA")
    print("=" * 78)

    settings = load_settings()
    print(f"\n[1] Cấu hình")
    print(f"    Neo4j            : {settings.neo4j_uri}")
    print(f"    Vector index     : {settings.vector_index}")
    print(f"    Gemini model     : {settings.gemini_model}")
    print(f"    GEMINI_API_KEY   : {'***' + settings.gemini_api_key[-4:] if len(settings.gemini_api_key) > 4 else 'set'}")

    # ---- 2. Build clients ----------------------------------------
    print("\n" + "-" * 78)
    print("[2] KẾT NỐI")
    print("-" * 78)
    retriever = MultiHopRetriever(
        database=settings.neo4j_database,
        vector_index=settings.vector_index,
    )
    gemini = GeminiClient(settings.gemini_api_key, settings.gemini_model)
    print(f"    Neo4j connected : {settings.neo4j_uri}")
    print(f"    Gemini loaded   : {settings.gemini_model}")

    # ---- 3. Run each query ----------------------------------------
    print("\n" + "-" * 78)
    print("[3] TRUY VẤN + TẠO CÂU TRẢ LỜI")
    print("-" * 78)

    all_results: list[dict] = []
    for query in DEMO_QUERIES:
        print(f"\n  Q: {query}")
        result = retriever.search(query, top_k=4, num_hops=1)
        print(f"  Context: {len(result.documents)} doc(s), {len(result.chunks)} chunk(s)")
        for c in result.chunks[:5]:
            print(f"    [{c.source:<10}] score={c.score:+.3f} [{c.kind:<10}] len(text)={len(c.text or '')} {c.text[:80]!r}")
        for c in result.chunks[:5]:
            print(f"    [{c.source:<10}] score={c.score:+.3f} [{c.kind:<10}] {c.parent_doc_id[:8]}…  {(c.text or c.title)[:60].replace(chr(10), ' ')!r}")

        messages = build_messages(query, result, max_chunks=8)
        gen = gemini.generate(messages, temperature=0.2, max_output_tokens=2000)

        print(f"\n  Gemini ({gen.model}) — {gen.elapsed_ms:.0f} ms, "
              f"in={gen.input_tokens} out={gen.output_tokens} tokens")
        print("  ---")
        for line in gen.text.splitlines():
            print(f"  | {line}")
        print("  ---")

        all_results.append({
            "query": query,
            "context": {
                "n_documents": len(result.documents),
                "n_chunks": len(result.chunks),
                "chunks": [
                    {
                        "source": c.source,
                        "score": c.score,
                        "kind": c.kind,
                        "doc_id": c.parent_doc_id,
                        "preview": (c.text or c.title)[:200],
                    }
                    for c in result.chunks
                ],
            },
            "answer": gen.text,
            "answer_meta": {
                "model": gen.model,
                "elapsed_ms": gen.elapsed_ms,
                "input_tokens": gen.input_tokens,
                "output_tokens": gen.output_tokens,
            },
        })

    # ---- 4. Persist -----------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "qa_results.json"
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[4] ĐÃ GHI: {out_path}")

    retriever.close()


if __name__ == "__main__":
    main()