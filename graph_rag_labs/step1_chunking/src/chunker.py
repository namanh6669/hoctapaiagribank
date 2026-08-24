"""Hierarchical chunker for legal/banking Vietnamese documents.

The input is a *cleaned* BeautifulSoup tree (see :mod:`cleaner`). The chunker
walks the tree and produces a flat list of ``Chunk`` objects that follow the
canonical hierarchy used by Vietnamese legal texts:

    Document
      └── Chương (Chapter)
            └── Mục (Section)        # optional
                  └── Điều (Article)
                        └── Paragraph / List / Table

Two structural relations are emitted in addition to the parent/child links:

* ``parent_id`` / ``children_ids`` — the natural containment tree.
* ``next_id`` — links each chunk to the next sibling that follows it in the
  reading order. This preserves the linear reading flow (``NEXT`` edge in
  Graph-RAG terminology) without bloating every chunk with full HTML.

Each chunk only carries the text it owns; HTML boilerplate is not stored on
the node. The Document root holds the document title (the first ``h1`` /
``h2`` it sees, falling back to a caller-supplied title).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from bs4 import BeautifulSoup, NavigableString, Tag


# ---------------------------------------------------------------------------
# Heading patterns (Vietnamese legal text)
# ---------------------------------------------------------------------------

# Each pattern captures (number, title). Number may be empty.
_CHUONG_RE = re.compile(r"^Chương\s+([IVXLC0-9]+)\b\.?\s*(.*)$", re.IGNORECASE)
_MUC_RE = re.compile(r"^Mục\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
_DIEU_RE = re.compile(r"^Điều\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)

# Default heading-level mapping when the document doesn't use keywords. The
# level decides the chunk kind if no Vietnamese keyword matched.
_HEADING_LEVEL_KIND: dict[int, str] = {
    1: "document",
    2: "chapter",
    3: "article",
    4: "subarticle",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single node in the hierarchical chunk tree.

    The ``heading_path`` is the breadcrumb the chunk sits under, e.g.
    ``["THÔNG TƯ 02/2023/TT-NHNN", "Chương II", "Điều 5"]``. It is convenient
    for both retrieval and human-readable reports.
    """

    id: str
    kind: str  # "document" | "chapter" | "section" | "article" | "paragraph" | "table" | "list"
    title: str
    text: str
    heading_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    next_id: str | None = None

    # Light metadata so downstream consumers don't need to re-parse the HTML.
    depth: int = 0
    char_count: int = 0
    source_doc: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "heading_path": list(self.heading_path),
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "next_id": self.next_id,
            "depth": self.depth,
            "char_count": self.char_count,
            "source_doc": self.source_doc,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    soup: BeautifulSoup,
    *,
    doc_title: str | None = None,
    source_doc: str | None = None,
) -> list[Chunk]:
    """Walk a cleaned soup and emit hierarchical chunks.

    The returned list is in reading order. The first chunk is the document
    root, followed by its children, then their children, etc.
    """
    chunks: list[Chunk] = []
    next_id = _id_factory()

    # ---- Document root --------------------------------------------------
    root = Chunk(
        id=next_id(),
        kind="document",
        title=doc_title or _detect_document_title(soup),
        text="",  # The root carries no text; content lives in children.
        heading_path=[],
        depth=0,
        source_doc=source_doc,
    )
    chunks.append(root)

    # Build a *linear* walk of body elements, keeping track of the current
    # chapter / section / article at every step. We assign parent links on
    # the fly so the chunk list can be returned without a second pass.
    #
    # `subtitle_tail` collects standalone heading text (e.g. "QUY ĐỊNH CHUNG"
    # rendered as its own ``<h2>`` after "Chương I") that should be appended
    # to the most recently emitted chapter.
    current_chapter = root
    current_section = root
    current_article = root
    last_leaf: Chunk | None = None  # for NEXT linking

    for elem in _iter_block_elements(soup):
        kind, title, title_only, subtitle = _classify(elem)
        if kind == "ignored":
            continue

        if kind == "subtitle":
            # The previous chapter is the natural host for this descriptive
            # title. Retroactively merge it into the chapter's title and
            # propagate the change into every descendant heading_path.
            _merge_subtitle_into_last_chapter(chunks, title)
            continue

        if kind == "document":
            # An <h1> inside the body usually refines the document title.
            if title and not root.title:
                root.title = title
            continue

        if kind in {"chapter", "section", "article", "subarticle"}:
            chunk = _make_container(
                next_id,
                kind=kind,
                title=title_only or title,
                full_title=title,
                parent_id=root.id,
                source_doc=source_doc,
                depth=_kind_depth(kind),
            )
            chunks.append(chunk)
            root.children_ids.append(chunk.id)

            # Capture the *parent* context for heading_path BEFORE we move
            # the current_* pointers forward.
            parent_chapter = current_chapter
            parent_section = current_section
            parent_article = current_article

            # Wire up the parent chain.
            if kind == "chapter":
                current_chapter = chunk
                current_section = chunk
                current_article = chunk
            elif kind == "section":
                chunk.parent_id = current_chapter.id
                current_chapter.children_ids.append(chunk.id)
                current_section = chunk
                current_article = chunk
            elif kind == "article":
                chunk.parent_id = current_section.id
                current_section.children_ids.append(chunk.id)
                current_article = chunk
            elif kind == "subarticle":
                chunk.parent_id = current_article.id
                current_article.children_ids.append(chunk.id)

            # Build the heading path (document -> chapter -> ... -> leaf).
            chunk.heading_path = _build_heading_path(
                root,
                parent_chapter,
                parent_section,
                parent_article,
                chunk,
            )

            # NEXT link: previous leaf (or previous container) -> this chunk.
            if last_leaf is not None:
                last_leaf.next_id = chunk.id
            last_leaf = chunk
            continue

        # Leaf-level element: paragraph, list, table.
        leaf = _make_leaf(next_id, elem, kind, source_doc)
        leaf.parent_id = current_article.id
        leaf.heading_path = _build_heading_path(
            root,
            current_chapter,
            current_section if current_section.id != current_chapter.id else None,
            current_article if current_article.id != current_section.id else None,
            leaf,
        )
        current_article.children_ids.append(leaf.id)
        chunks.append(leaf)

        if last_leaf is not None:
            last_leaf.next_id = leaf.id
        last_leaf = leaf

    # ---- Final NEXT to the root ----------------------------------------
    if last_leaf is not None and last_leaf.next_id is None:
        last_leaf.next_id = root.id

    # The Document root points to the very first chunk after it. Without
    # this the linear walk has nowhere to start.
    if root.children_ids:
        root.next_id = root.children_ids[0]

    return chunks


# ---------------------------------------------------------------------------
# Element classification
# ---------------------------------------------------------------------------


def _classify(elem: Tag) -> tuple[str, str, str, str]:
    """Return ``(kind, full_title, title_only, subtitle)`` for a block element.

    * ``kind`` is the structural role: ``chapter`` / ``section`` / ``article``
      / ``paragraph`` / ``list`` / ``table`` / ``subtitle`` / ``ignored``.
    * ``full_title`` includes any number prefix ("Chương II").
    * ``title_only`` is the descriptive part after the number.
    * ``subtitle`` is set only for "subtitle" headings — descriptive lines
      like "QUY ĐỊNH CHUNG" that immediately follow a chapter number and
      should be merged into the chapter title rather than emitted as a
      separate chunk.
    """
    if elem.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        text = elem.get_text(" ", strip=True)
        if not text:
            return "ignored", "", "", ""

        for pattern, kind in (
            (_CHUONG_RE, "chapter"),
            (_MUC_RE, "section"),
            (_DIEU_RE, "article"),
        ):
            m = pattern.match(text)
            if m:
                number, rest = m.group(1), m.group(2).strip(" .:-")
                full = f"{_kind_vn(kind)} {number}" + (f" - {rest}" if rest else "")
                return kind, full, rest or text, ""

        # A heading that has no Chương/Mục/Điều number is treated as a
        # *subtitle* (descriptive title for the most recent container).
        # ``h1`` is special: it refines the document title.
        if elem.name == "h1":
            return "document", text, text, ""

        return "subtitle", text, text, text

    if elem.name in {"p"}:
        return "paragraph", "", "", ""

    if elem.name in {"ul", "ol"}:
        return "list", "", "", ""

    if elem.name == "table":
        return "table", "", "", ""

    return "ignored", "", "", ""


def _make_container(
    id_factory,
    *,
    kind: str,
    title: str,
    full_title: str,
    parent_id: str,
    source_doc: str | None,
    depth: int,
) -> Chunk:
    chunk = Chunk(
        id=id_factory(),
        kind=kind,
        title=full_title,
        text="",  # Containers carry no body text; their children do.
        parent_id=parent_id,
        depth=depth,
        source_doc=source_doc,
        char_count=len(full_title),
    )
    return chunk


def _merge_subtitle_into_last_chapter(chunks: list[Chunk], subtitle: str) -> None:
    """Append ``subtitle`` to the most recently emitted chapter.

    Also rewrites every descendant's ``heading_path`` so the breadcrumb
    stays in sync with the new chapter title. If no chapter has been
    emitted yet (e.g. subtitle precedes the first "Chương"), the function
    is a no-op.
    """
    # Find the last chunk with kind == "chapter"
    last_chapter: Chunk | None = None
    for c in reversed(chunks):
        if c.kind == "chapter":
            last_chapter = c
            break
    if last_chapter is None:
        return

    # Avoid duplicating the same subtitle if we run twice.
    if subtitle in last_chapter.title:
        return

    new_title = f"{last_chapter.title} - {subtitle}"
    last_chapter.title = new_title
    last_chapter.char_count = len(new_title)

    # Propagate the change into descendants that include the chapter title.
    old_token = last_chapter.title.split(" - ")[0]  # e.g. "Chương II"
    for c in chunks:
        if c.id == last_chapter.id:
            continue
        if c.heading_path and old_token in c.heading_path[1] if len(c.heading_path) > 1 else False:
            # Rebuild the heading_path slot 1 (the chapter slot).
            new_path = list(c.heading_path)
            new_path[1] = new_title
            c.heading_path = new_path
        elif c.heading_path and len(c.heading_path) >= 2 and c.heading_path[1] == old_token:
            new_path = list(c.heading_path)
            new_path[1] = new_title
            c.heading_path = new_path


def _make_leaf(
    id_factory,
    elem: Tag,
    kind: str,
    source_doc: str | None,
) -> Chunk:
    text = elem.get_text("\n", strip=True)
    # Compress runs of blank lines inside a leaf.
    text = re.sub(r"\n{2,}", "\n", text).strip()
    preview = text[:60] + ("..." if len(text) > 60 else "")
    return Chunk(
        id=id_factory(),
        kind=kind,
        title=preview,
        text=text,
        depth=_kind_depth(kind),
        source_doc=source_doc,
        char_count=len(text),
    )


def _build_heading_path(
    root: Chunk,
    chapter: Chunk,
    section: Chunk | None,
    article: Chunk | None,
    leaf: Chunk,
) -> list[str]:
    """Build the breadcrumb path the leaf sits under."""
    path: list[str] = []
    if root is leaf:
        return path
    if root.title:
        path.append(root.title)
    if chapter is not leaf and chapter is not root:
        path.append(chapter.title)
    if section is not None and section is not leaf and section is not chapter and section is not root:
        path.append(section.title)
    if article is not None and article is not leaf and article is not section and article is not chapter:
        path.append(article.title)
    if leaf.kind in {"paragraph", "list", "table"}:
        path.append(leaf.title)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_document_title(soup: BeautifulSoup) -> str:
    """Best-effort detection of the document title from the first heading."""
    for tag in soup.find_all(["h1", "h2"]):
        text = tag.get_text(" ", strip=True)
        if text:
            return text
    return "Untitled Document"


def _iter_block_elements(soup: BeautifulSoup) -> Iterator[Tag]:
    """Yield top-level structural elements in reading order.

    We descend into the soup one level at a time: anything inside a ``<body>``
    if present, otherwise the whole document. We skip empty whitespace.
    """
    root = soup.find("body") or soup
    for child in root.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                # Stray text without a tag -> treat as a paragraph.
                yield _wrap_text_as_paragraph(soup, text)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"}:
            yield child


def _wrap_text_as_paragraph(soup: BeautifulSoup, text: str) -> Tag:
    new_p = soup.new_tag("p")
    new_p.string = text
    return new_p


def _kind_depth(kind: str) -> int:
    return {
        "document": 0,
        "chapter": 1,
        "section": 2,
        "article": 3,
        "subarticle": 4,
        "paragraph": 4,
        "list": 4,
        "table": 4,
    }.get(kind, 5)


def _kind_vn(kind: str) -> str:
    return {
        "chapter": "Chương",
        "section": "Mục",
        "article": "Điều",
        "subarticle": "Điều",
    }.get(kind, kind)


def _id_factory() -> "Iterator[str]":
    counter = {"n": 0}

    def gen() -> str:
        counter["n"] += 1
        return f"c{counter['n']:04d}-{uuid.uuid4().hex[:6]}"

    return gen