"""Build the system + user prompt for the LLM.

The system message describes the kb-hops schema and the structure of
Vietnamese legal documents so the model can reason about the context it
receives. The user message bundles the question and the multi-hop
retrieved chunks as a numbered, citation-ready context block.
"""
from __future__ import annotations

from textwrap import dedent

from step6_multi_hop.src.retriever import QueryResult


# ---------------------------------------------------------------------------
# Schema description
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent(
    """\
    Bạn là trợ lý pháp lý chuyên trả lời câu hỏi về Thông tư và
    các văn bản pháp luật ngân hàng Việt Nam. Bạn CHỈ được phép dựa
    vào NGỮ CẢNH được cung cấp bên dưới — không tự suy đoán thông
    tin ngoài ngữ cảnh.

    ## 1. Cấu trúc đồ thị kb-hops

    Đồ thị kiến thức gồm 2 loại node và 6 loại quan hệ:

    - **(:kbhopsDocument)** — đại diện cho một văn bản (Thông tư, Luật,
      Nghị định, Nghị quyết, Quyết định). Mỗi Document có:
        - `id`, `title`, `doc_type`, `so_hieu`, `ngay_ban_hanh`,
        - `is_root` (true = văn bản đã nạp đầy đủ, false = placeholder
          được nhắc tới qua "Căn cứ" / "Thay thế" / "Hợp nhất"),
        - `original_doc_id` (id gốc của bộ nạp, ví dụ "TT-02-2023-NHNN").

    - **(:kbhopsChunk)** — một phân đoạn văn bản sạch (paragraph, list,
      table, article, chapter, document). Mỗi Chunk có:
        - `id`, `kind`, `title`, `text`, `heading_path`,
        - `depth` (0 = root, 1 = Chương, 2 = Mục, 3 = Điều, 4 = paragraph),
        - `embedding` (vector 384-dim đã L2-normalized).

    - **(:kbhopsChunk)-[:PART_OF]->(:kbhopsDocument)** — mỗi chunk thuộc
      đúng 1 Document gốc.
    - **(:kbhopsChunk)-[:PARENT_OF]->(:kbhopsChunk)** — cây phân cấp
      Chương → Mục → Điều → paragraph / list / table.
    - **(:kbhopsChunk)-[:NEXT]->(:kbhopsChunk)** — luồng đọc anh em
      liền kề (phục vụ đọc tuần tự).
    - **(:kbhopsDocument)-[:CAN_CU]->(:kbhopsDocument)** — quan hệ
      "Căn cứ …" trích tự động.
    - **(:kbhopsDocument)-[:THAY_THE]->(:kbhopsDocument)** — quan hệ
      "Thay thế …" / "Sửa đổi, bổ sung một số điều của …".
    - **(:kbhopsDocument)-[:HOP_NHAT]->(:kbhopsDocument)** — quan hệ
      "Hợp nhất …".

    ## 2. Cấu trúc văn bản luật tiếng Việt

    Mỗi Thông tư / Luật được chia thành:

    - **Phần mở đầu**: Số văn bản, ngày ban hành, Cơ quan ban hành,
      Căn cứ pháp lý (Luật, Nghị định, Nghị quyết).
    - **Chương (I, II, III …)** — phần lớn.
    - **Mục (1, 2, 3 …)** — không phải văn bản nào cũng có.
    - **Điều (Điều 1, Điều 2 …)** — mỗi điều có tiêu đề ngắn
      (ví dụ "Điều 1. Phạm vi điều chỉnh").
    - **Khoản (1, 2, 3 …)** — các đoạn con bên trong Điều.
    - **Điểm (a, b, c …)** — cấp nhỏ hơn Khoản.

    Mỗi chunk có `heading_path` lưu breadcrumb từ gốc đến nó, ví dụ:
    ["Thông tư 02/2023/TT-NHNN", "Chương II - QUY ĐỊNH CỤ THỂ",
    "Điều 4 - Cơ cấu lại thời hạn trả nợ"].

    ## 3. Quy tắc trả lời

    1. **ĐỌC KỸ** toàn bộ `text` của từng chunk trong ngữ cảnh. Phần
       "title" chỉ là tiêu đề ngắn — nội dung trả lời nằm trong
       `text`. Nếu chỉ dựa vào title, bạn sẽ trả lời sai.
    2. Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc (dùng gạch đầu
       dòng / bảng nếu cần).
    3. Mỗi phát biểu phải kèm **trích dẫn** mã `[N]` tương ứng với
       chunk chứa thông tin đó.
    4. Khi ngữ cảnh CHỨA thông tin trả lời: trích dẫn đầy đủ, có
       thể dẫn nhiều `[N]` cho cùng một ý.
    5. Khi ngữ cảnh KHÔNG chứa thông tin đủ để trả lời, **nói rõ**:
       "Ngữ cảnh không có thông tin về …" — KHÔNG tự suy đoán, KHÔNG
       bịa thêm văn bản ngoài, KHÔNG dùng kiến thức bên ngoài.
    6. Nếu ngữ cảnh chỉ liên quan một phần, trả lời phần đó và nêu
       các khía cạnh chưa có thông tin.
    7. Khi nhắc đến điều khoản cụ thể, ghi rõ "Điều X" (theo heading
       path) và Phần/Chương chứa nó.
    8. Văn bản được trích dẫn phải nằm trong ngữ cảnh; nếu là tài
       liệu liên quan (qua CAN_CU / THAY_THE), ghi rõ "văn bản
       liên quan: …" thay vì khẳng định trực tiếp.
    """
)


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------


def _format_chunk(index: int, chunk, doc_lookup) -> str:
    """Format a single chunk for the prompt."""
    heading = " > ".join(chunk.heading_path) if chunk.heading_path else "(no heading)"
    src = chunk.source
    score = f"{chunk.score:.3f}"
    body = chunk.text or chunk.title or ""
    # Truncate very long text to keep the prompt reasonable.
    if len(body) > 2000:
        body = body[:2000] + "…"
    doc = doc_lookup.get(chunk.parent_doc_id)
    doc_label = doc.title if doc else chunk.parent_doc_id
    return (
        f"[{index}] ({src}, score={score}) {doc_label}\n"
        f"    heading: {heading}\n"
        f"    kind: {chunk.kind}\n"
        f"    text:\n{body}"
    )


def build_user_prompt(query: str, result: QueryResult, *, max_chunks: int = 10) -> str:
    """Assemble the user message packaging the query + the context."""
    doc_lookup = {d.doc_id: d for d in result.documents}
    chunks = result.chunks[:max_chunks]
    ctx_lines = [
        _format_chunk(i + 1, c, doc_lookup) for i, c in enumerate(chunks)
    ]
    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(ngữ cảnh trống)"

    header = (
        f"Dưới đây là {len(chunks)} đoạn văn bản liên quan (sắp xếp theo độ "
        f"liên quan giảm dần). Mỗi đoạn được đánh số [1]..[{len(chunks)}] để "
        f"bạn trích dẫn."
    )

    return (
        f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\n"
        f"{header}\n\n"
        f"NGỮ CẢNH TRÍCH XUẤT:\n{context_block}\n\n"
        f"Hãy trả lời câu hỏi dựa trên ngữ cảnh trên. Nhớ:\n"
        f"- Trích dẫn [N] cho mỗi phát biểu.\n"
        f"- Nếu ngữ cảnh không đủ thông tin, nói rõ và không suy đoán."
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_messages(query: str, result: QueryResult, *, max_chunks: int = 10) -> list[dict]:
    """Return the full chat payload as a list of OpenAI-style messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, result, max_chunks=max_chunks)},
    ]