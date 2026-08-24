"""Trích xuất quan hệ cấp tài liệu từ text.

Quét toàn bộ ``Chunk.heading_path`` + ``Chunk.text`` để tìm các cụm:

* ``Căn cứ <tên văn bản>``         → ``[:CAN_CU]``
* ``Thay thế <tên văn bản>``       → ``[:THAY_THE]``
* ``Hợp nhất <tên văn bản>``       → ``[:HOP_NHAT]``

Mỗi văn bản được nhắc tới sẽ trở thành một ``Document`` placeholder (chúng
ta không có nội dung của nó, chỉ có tiêu đề + (nếu match được) số hiệu +
ngày ban hành). Quan hệ nối ``Document`` gốc → ``Document`` được nhắc tới.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

# Numbering patterns that appear inside a legal reference, e.g.
#   "39/2016/TT-NHNN", "102/2022/NĐ-CP", "59/NQ-CP"
_SO_HIEU_RE = re.compile(
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐÂĂÊÔƠƯÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴ\-]+",
    re.UNICODE,
)

# Capture "<type keyword> <reference>" — the reference includes the
# short code + (optional) the descriptive name until a comma, semicolon
# or end-of-clause. The keyword MUST start with a capital letter to
# avoid matching body prose like "căn cứ vào phương án ...".
_REF_BLOCK_RE = re.compile(
    r"(?P<keyword>Căn cứ|"
    r"Thay thế|"
    r"Sửa đổi, bổ sung(?:\smột\ssố\sđiều)?\scủa|"
    r"Hợp nhất)\s+"
    r"(?P<ref>[^.;:\n]+(?:;[^.;:\n]+)*)"
)

# Matches date forms used in the references, e.g. "ngày 23 tháng 4 năm 2023".
_DATE_RE = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)


# Map of the original keyword (after lowercasing + stripping accents) to
# the canonical relationship kind.
_KEYWORD_MAP: dict[str, str] = {
    "can cu": "CAN_CU",
    "thay the": "THAY_THE",
    "hop nhat": "HOP_NHAT",
    # "Sửa đổi, bổ sung một số điều của" semantically means "this doc
    # modifies that doc" — treat it as a supersede/THAY_THE-style link.
    "sua doi, bo sung": "THAY_THE",
    "sua doi, bo sung mot so dieu cua": "THAY_THE",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentRef:
    """One external document reference (placeholder)."""

    canonical_id: str       # stable id we use for the Document node
    title: str              # human-readable title
    doc_type: str           # "luat" | "nghi_dinh" | "thong_tu" | "nghi_quyet" | "unknown"
    so_hieu: str | None     # "39/2016/TT-NHNN" if detected
    ngay_ban_hanh: str | None  # ISO date if detected


@dataclass
class DocumentRelationships:
    """Edges to create from the source Document to referenced Documents."""

    can_cu: list[DocumentRef] = field(default_factory=list)
    thay_the: list[DocumentRef] = field(default_factory=list)
    hop_nhat: list[DocumentRef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_relationships(
    *,
    document_text: str,
    document_heading: str = "",
    scan_body: bool = False,
    preamble_chars: int = 1500,
    max_title_len: int = 220,
    skip_titles: tuple[str, ...] = (
        "thông tư này",
        "thong tu nay",
        "quy định nội bộ",
        "quy dinh noi bo",
        "nội bộ",
        "noi bo",
    ),
) -> DocumentRelationships:
    """Scan a document and return its document-level edges.

    The extractor keeps references that look like real legal documents
    (they carry a "số hiệu" code like ``39/2016/TT-NHNN``) or whose
    cleaned title is reasonably short. Body prose that the regex catches
    by accident (e.g. "vào phương án sử dụng vốn…") is dropped because
    it has no ``số hiệu`` and its title is too long.
    """
    haystack = (
        "\n".join([document_heading, document_text]) if scan_body
        else "\n".join([document_heading, document_text[:preamble_chars]])
    ).strip()

    rels = DocumentRelationships()
    seen: set[tuple[str, str]] = set()
    skip_norm = {_normalise(s) for s in skip_titles}

    # Real references start with one of these doc-type prefixes
    # (accent-stripped). Anything else is body prose the regex caught.
    doc_prefixes = (
        "luat", "nghi dinh", "thong tu", "nghi quyet",
        "phap lenh", "quyet dinh", "phap lenh",
    )

    for match in _REF_BLOCK_RE.finditer(haystack):
        keyword = _normalise(match.group("keyword"))
        rel_kind = _KEYWORD_MAP.get(keyword)
        if rel_kind is None:
            continue
        ref = DocumentRef(*_split_reference(match.group("ref")))

        title_norm = _normalise(ref.title)
        if any(skip in title_norm for skip in skip_norm):
            continue
        # Keep refs that EITHER have a recognised "số hiệu" OR start
        # with a known document-type prefix (Luật, Nghị định, ...) OR
        # whose title is short enough to plausibly be a doc title.
        starts_with_doc = any(title_norm.startswith(p) for p in doc_prefixes)
        if not ref.so_hieu and not starts_with_doc and len(ref.title) > max_title_len:
            continue

        key = (rel_kind, ref.canonical_id)
        if key in seen:
            continue
        seen.add(key)

        if rel_kind == "CAN_CU":
            rels.can_cu.append(ref)
        elif rel_kind == "THAY_THE":
            rels.thay_the.append(ref)
        elif rel_kind == "HOP_NHAT":
            rels.hop_nhat.append(ref)

    return rels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics for keyword matching."""
    replacements = str.maketrans(
        "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ",
        "aadeoouaaaaaaaaaaaaaaaeeeeeeeeeeiiiiiooooooooooooooouuuuuuuuuuyyyyy",
    )
    return text.translate(replacements).strip().lower()


def _split_reference(raw: str) -> tuple[str, str, str, str | None, str | None]:
    """Turn a raw reference fragment into a structured ``DocumentRef``.

    Returns ``(canonical_id, title, doc_type, so_hieu, ngay_ban_hanh)``.
    """
    text = raw.strip().rstrip(",;").strip()

    # Số hiệu — first match wins.
    so_match = _SO_HIEU_RE.search(text)
    so_hieu = _normalise_so_hieu(so_match.group(0)) if so_match else None

    # Date.
    date_match = _DATE_RE.search(text)
    ngay = (
        f"{date_match.group(3)}-{int(date_match.group(1)):02d}-{int(date_match.group(2)):02d}"
        if date_match
        else None
    )

    # Title — the cleaned reference text without the so_hieu / date noise.
    title = text
    if so_hieu:
        title = re.sub(re.escape(so_match.group(0)), "", title, count=1).strip()
    if date_match:
        title = _DATE_RE.sub("", title, count=1).strip(" ,;.")
    # Collapse repeated whitespace.
    title = re.sub(r"\s+", " ", title).strip(" ,;.")

    doc_type = _detect_doc_type(so_hieu, title)

    # Build the slug. Strip suffixes like "và Luật sửa đổi..." /
    # "; Luật sửa đổi..." so the same law dedupes across documents.
    slug = so_hieu or _slugify(_strip_trailing_conjunctions(title))

    return slug, title, doc_type, so_hieu, ngay


_CONJUNCTION_RE = re.compile(
    r"\s*(?:;|và|&)\s+.*$", flags=re.IGNORECASE
)


def _strip_trailing_conjunctions(title: str) -> str:
    """Drop '; Luật sửa đổi…' / 'và Nghị định…' from the title's tail."""
    return _CONJUNCTION_RE.sub("", title).strip(" ,;.")


def _normalise_so_hieu(s: str) -> str:
    """Strip whitespace inside a document code."""
    return re.sub(r"\s+", "", s).upper()


def _slugify(text: str) -> str:
    """Make a stable id from a title (ASCII, lower, dash-joined)."""
    norm = _normalise(text)
    norm = re.sub(r"[^a-z0-9]+", "-", norm).strip("-")
    return f"doc-{norm}" if norm else "doc-unknown"


def _detect_doc_type(so_hieu: str | None, title: str) -> str:
    """Guess the document type from the so_hieu suffix."""
    if so_hieu:
        upper = so_hieu.upper()
        if "/TT-" in upper:
            return "thong_tu"
        if "/ND-CP" in upper or "/NĐ-CP" in upper:
            return "nghi_dinh"
        if "/NQ-" in upper:
            return "nghi_quyet"
        if "/QĐ" in upper or "/QD-" in upper:
            return "quyet_dinh"
        if "/PL-" in upper:
            return "phap_lenh"
    t = _normalise(title)
    if t.startswith("luat "):
        return "luat"
    if t.startswith("nghi dinh ") or t.startswith("nghi-dinh "):
        return "nghi_dinh"
    if t.startswith("thong tu ") or t.startswith("thong-tu "):
        return "thong_tu"
    if t.startswith("nghi quyet ") or t.startswith("nghi-quyet "):
        return "nghi_quyet"
    return "unknown"