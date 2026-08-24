"""End-to-end demo for Bước 1.

Reads a markdown legal document, runs the cleaning step, then the
hierarchical chunking step, and prints a sample of the result to the
console so the structure is visible at a glance.

Run from the ``step1_chunking`` folder:

    python -m src.run_demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .cleaner import clean_markdown, summarise
from .chunker import chunk_document, Chunk


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "output"
SAMPLE_OUT = OUTPUT_DIR / "sample"

DEFAULT_INPUT = DATA_DIR / "TT_02_2023_NHNN.md"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run(input_path: Path = DEFAULT_INPUT) -> list[Chunk]:
    print("=" * 78)
    print(f"BƯỚC 1 — LÀM SẠCH HTML & CHUNKING PHÂN CẤP")
    print(f"Tệp đầu vào  : {input_path.name}")
    print("=" * 78)

    raw_md = input_path.read_text(encoding="utf-8")
    print(f"\n[1] Markdown gốc: {len(raw_md):,} ký tự\n")

    # ---- 2. Cleaning ----------------------------------------------------
    print("-" * 78)
    print("[2] LÀM SẠCH HTML")
    print("-" * 78)
    cleaned = clean_markdown(raw_md)
    for line in summarise(cleaned):
        print(f"  {line}")
    print()

    # Show a tiny before/after HTML fragment so the diff is visible.
    _print_before_after(raw_md, cleaned)

    # Demo the cleaning step on a noisy synthetic HTML fragment so the
    # user can see what the cleaner strips. Markdown input rarely carries
    # any of the typical web noise (scripts, classes, ads), so a separate
    # example is more illustrative.
    _demo_noisy_html()

    # ---- 3. Chunking ----------------------------------------------------
    print("-" * 78)
    print("[3] CHUNKING PHÂN CẤP")
    print("-" * 78)
    chunks = chunk_document(
        cleaned.soup,
        doc_title="Thông tư 02/2023/TT-NHNN",
        source_doc=input_path.name,
    )
    _print_chunk_summary(chunks)
    _print_sample_tree(chunks)
    _print_sample_chunks(chunks)

    # ---- 4. Persist ----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUT.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in chunks]
    out_path = OUTPUT_DIR / "chunks.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sample_path = SAMPLE_OUT / "chunks_sample.json"
    sample_path.write_text(
        json.dumps(payload[:8], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[4] ĐÃ GHI:")
    print(f"  - Toàn bộ chunks -> {out_path}")
    print(f"  - 8 chunk đầu    -> {sample_path}")

    # ---- 5. NEXT walk --------------------------------------------------
    _print_next_relations(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _print_before_after(raw_md: str, cleaned) -> None:
    """Print a small before/after fragment so the user can see what changed."""
    # Take the first 600 chars of markdown and the first 600 chars of cleaned HTML.
    md_fragment = raw_md[:600].replace("\n", " ")
    html_fragment = str(cleaned.soup)[:600]
    print("  Trước (markdown đầu vào — 600 ký tự đầu):")
    print(f"    {md_fragment}…")
    print()
    print("  Sau (HTML đã làm sạch — 600 ký tự đầu):")
    print(f"    {html_fragment}…")
    print()


def _print_chunk_summary(chunks: list[Chunk]) -> None:
    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    print(f"  Tổng số chunk        : {len(chunks)}")
    for kind in ("document", "chapter", "section", "article", "subarticle", "paragraph", "list", "table"):
        if kind in by_kind:
            print(f"    {kind:<12} : {by_kind[kind]}")
    total_chars = sum(c.char_count for c in chunks)
    leaves = [c for c in chunks if c.kind in {"paragraph", "list", "table"}]
    avg_chars = total_chars // max(len(leaves), 1)
    print(f"  Tổng ký tự (chunk leaf): {total_chars:,}")
    print(f"  Trung bình ký tự/leaf  : {avg_chars:,}")
    print()


def _print_sample_tree(chunks: list[Chunk]) -> None:
    """Print the document's hierarchical tree (only the first few levels)."""
    print("  CÂY PHÂN CẤP (3 lớp đầu):")
    by_id = {c.id: c for c in chunks}
    root = chunks[0]
    print(f"  └─ [{root.kind}] {root.title}")
    for cid in root.children_ids[:5]:
        chap = by_id.get(cid)
        if chap is None:
            continue
        print(f"      └─ [{chap.kind}] {chap.title}")
        for cid2 in chap.children_ids[:3]:
            sub = by_id.get(cid2)
            if sub is None:
                continue
            print(f"          └─ [{sub.kind}] {sub.title[:60]}")
    print()


def _print_sample_chunks(chunks: list[Chunk]) -> None:
    """Print 4 representative chunks in full so the structure is visible."""
    print("-" * 78)
    print("[4] MẪU CHUNK CHI TIẾT (chapter / article / paragraph / table)")
    print("-" * 78)
    chapter = next((c for c in chunks if c.kind == "chapter"), None)
    article = next((c for c in chunks if c.kind == "article"), None)
    paragraph = next((c for c in chunks if c.kind == "paragraph"), None)
    table = next((c for c in chunks if c.kind == "table"), None)

    for label, chunk in (
        ("CHƯƠNG", chapter),
        ("ĐIỀU", article),
        ("ĐOẠN VĂN", paragraph),
        ("BẢNG", table),
    ):
        if chunk is None:
            continue
        print(f"\n  ┌── {label} :: {chunk.title}")
        print(f"  │ id           : {chunk.id}")
        print(f"  │ kind         : {chunk.kind}")
        print(f"  │ depth        : {chunk.depth}")
        print(f"  │ heading_path :")
        for hop in chunk.heading_path:
            print(f"  │   └─ {hop}")
        print(f"  │ parent_id    : {chunk.parent_id}")
        if chunk.children_ids:
            preview = chunk.children_ids[:3] + (["…"] if len(chunk.children_ids) > 3 else [])
            print(f"  │ children_ids : {preview}")
        print(f"  │ next_id      : {chunk.next_id}")
        print(f"  │ char_count   : {chunk.char_count}")
        print(f"  │ text (120 ký tự đầu):")
        body = chunk.text.replace("\n", " ")[:120]
        print(f"  │   {body}{'…' if len(chunk.text) > 120 else ''}")
        print(f"  └──")


def _print_next_relations(chunks: list[Chunk]) -> None:
    """Walk a few NEXT links so the linear reading flow is visible."""
    print()
    print("-" * 78)
    print("[5] THỬ NGHIỆM QUAN HỆ NEXT (6 node đầu)")
    print("-" * 78)
    by_id = {c.id: c for c in chunks}
    cursor = chunks[0].next_id
    steps = 0
    while cursor and steps < 6:
        node = by_id.get(cursor)
        if node is None:
            break
        print(f"  → [{node.kind}] {node.title[:55]:<55} | parent={node.parent_id}")
        cursor = node.next_id
        steps += 1


def _demo_noisy_html() -> None:
    """Show the cleaner on a tiny synthetic HTML fragment.

    Markdown sources rarely carry scripts / inline styles / classes /
    on-click handlers, so the real-world reduction is small. This
    synthetic example makes the cleaning rules visible.
    """
    print()
    print("-" * 78)
    print("[2.5] DEMO NHANH — CLEANER TRÊN HTML NHIỄU (SYNTHETIC)")
    print("-" * 78)

    noisy = """
    <html>
      <head>
        <style>body { font-family: Arial; color: #333 }</style>
        <script>console.log('tracking');</script>
      </head>
      <body>
        <nav class="navbar" id="top-nav">
          <a href="/home" onclick="track()">Trang chủ</a>
          <a href="/about">Giới thiệu</a>
        </nav>
        <div class="ad-banner" data-slot="top" data-campaign="x42">
          <img src="banner.png" alt="Quảng cáo" />
          <p style="color:red; font-size:9pt">Ưu đãi lãi suất!</p>
        </div>

        <h1 class="title" id="main-title">Thông tư 02/2023/TT-NHNN</h1>
        <p style="text-align:center"><em>Ngày ban hành: 23/4/2023</em></p>

        <h2 onclick="expand()">Chương I<br/>QUY ĐỊNH CHUNG</h2>
        <p class="lead" data-id="42">Nội dung chính của thông tư...</p>

        <table class="data" border="1" cellpadding="5" bgcolor="#eee">
          <thead><tr><th>Mục</th><th>Nội dung</th></tr></thead>
          <tbody>
            <tr><td>1</td><td>Cơ cấu lại thời hạn trả nợ</td></tr>
            <tr><td>2</td><td>Giữ nguyên nhóm nợ</td></tr>
          </tbody>
        </table>

        <footer class="site-footer">
          <p>© 2023 Ngân hàng Nhà nước Việt Nam</p>
        </footer>
      </body>
    </html>
    """
    print(f"\n  HTML gốc (đoạn trích, {len(noisy):,} ký tự):")
    for line in noisy.strip().splitlines()[:8]:
        print(f"    {line.rstrip()}")
    print("    … (còn nữa) …")

    from .cleaner import clean_html
    cleaned = clean_html(noisy)
    print(f"\n  Kết quả làm sạch:")
    for line in summarise(cleaned):
        print(f"    {line}")
    print("\n  HTML sau khi sạch (toàn bộ):")
    cleaned_html = str(cleaned.soup)
    for line in cleaned_html.strip().splitlines():
        print(f"    {line.rstrip()}")


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    run(arg)