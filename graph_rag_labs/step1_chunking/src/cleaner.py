"""HTML/Markdown cleaning module for hierarchical chunking.

Takes raw HTML (or markdown that has been wrapped in HTML) and strips the
heavy, decorative parts while keeping the structural spine of the document:

- Headings (``h1`` - ``h6``) — they encode the legal hierarchy (Chương, Mục,
  Điều, ...).
- Paragraphs (``p``) — body text.
- Tables (``table``) — including ``thead`` / ``tbody`` / ``tr`` / ``td`` /
  ``th``.
- Lists (``ul`` / ``ol``) — for enumerated clauses (Khoản 1., Khoản 2., ...).

Everything else is dropped: ``script``, ``style``, ``nav``, ``aside``,
``header``, ``footer``, ``form``, inline styles, classes, ids, ``onclick``
handlers, ``data-*`` payloads, ``<a>`` wrappers around numbers, image banners,
etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


# ---------------------------------------------------------------------------
# Tag policy
# ---------------------------------------------------------------------------

# Tags we keep verbatim (structural only). Everything inside a kept tag is
# normalised but we never drop the tag itself.
KEPT_TAGS: frozenset[str] = frozenset(
    {
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "br", "hr",
        "ul", "ol", "li",
        "table", "thead", "tbody", "tfoot",
        "tr", "th", "td", "caption",
        "strong", "em", "b", "i", "u",
    }
)

# Tags we drop entirely (along with their content). These are decorative or
# behavioural wrappers that pollute the chunk text.
DROPPED_TAGS: frozenset[str] = frozenset(
    {
        "script", "style", "noscript", "iframe", "object", "embed",
        "nav", "aside", "header", "footer", "form", "button",
        "svg", "canvas", "video", "audio", "source", "track",
        "figure", "figcaption", "picture",
        "link", "meta",
    }
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CleanedDocument:
    """Output of the cleaning step.

    Attributes
    ----------
    soup:
        The BeautifulSoup tree after stripping. The caller can keep walking
        it for chunking.
    raw_size:
        Size of the original HTML in characters (for the before/after report).
    cleaned_size:
        Size of the cleaned HTML in characters.
    removed_tags:
        Counter of tags that were dropped, useful for the report.
    """

    soup: BeautifulSoup
    raw_size: int
    cleaned_size: int
    removed_tags: dict[str, int]

    @property
    def reduction_pct(self) -> float:
        if self.raw_size == 0:
            return 0.0
        return round((1 - self.cleaned_size / self.raw_size) * 100, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_html(html: str) -> CleanedDocument:
    """Strip a raw HTML document down to its structural spine.

    The function is intentionally conservative: it never deletes a heading or
    paragraph, it only removes decorative wrappers and inline noise (classes,
    styles, comments, JS handlers, ...).
    """
    if not html or not html.strip():
        empty = BeautifulSoup("", "lxml")
        return CleanedDocument(empty, 0, 0, {})

    raw_size = len(html)
    soup = BeautifulSoup(html, "lxml")

    removed: dict[str, int] = {}

    # 1) Strip comments (e.g. ``<!-- ad slot --->``)
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # 2) Drop tags we never want to keep (script, style, nav, ...)
    for tag in soup.find_all(True):
        name = tag.name.lower() if tag.name else ""
        if name in DROPPED_TAGS:
            removed[name] = removed.get(name, 0) + 1
            tag.decompose()

    # 3) For everything that survived, strip noisy attributes
    for tag in soup.find_all(True):
        name = tag.name.lower() if tag.name else ""
        if name not in KEPT_TAGS:
            # Unknown / decorative tag (div, span, section, article, ...):
            # unwrap it — keep its *children* but lose the wrapper.
            tag.unwrap()
            continue
        _strip_attributes(tag)

    # 4) Collapse empty paragraphs (often left after attribute stripping).
    for p in soup.find_all("p"):
        if not _has_meaningful_text(p):
            p.decompose()

    # 5) Tidy whitespace inside the remaining tree.
    _normalise_whitespace(soup)

    cleaned_html = str(soup)
    return CleanedDocument(
        soup=soup,
        raw_size=raw_size,
        cleaned_size=len(cleaned_html),
        removed_tags=removed,
    )


def clean_markdown(md: str) -> CleanedDocument:
    """Convenience wrapper: parse markdown, render it to HTML, then clean.

    The output is therefore identical in shape to ``clean_html`` output, so
    downstream chunking doesn't care about the source format.
    """
    import markdown as _md

    rendered = _md.markdown(
        md,
        extensions=["extra", "tables", "sane_lists", "nl2br"],
    )
    return clean_html(rendered)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_NOISY_ATTRS = re.compile(
    r"^(class|id|style|data-[\w-]+|on\w+|srcset|sizes|color|face|"
    r"width|height|align|valign|bgcolor|border|cellpadding|cellspacing|"
    r"target|rel|aria-[\w-]+|role)$",
    re.IGNORECASE,
)


def _strip_attributes(tag: Tag) -> None:
    """Drop every attribute that doesn't carry semantic value."""
    for attr in list(tag.attrs):
        if _NOISY_ATTRS.match(attr):
            del tag.attrs[attr]


def _has_meaningful_text(tag: Tag) -> bool:
    text = tag.get_text(" ", strip=True)
    return bool(text) and not _PAGE_NUMBER_RE.fullmatch(text)


_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")


def _normalise_whitespace(soup: BeautifulSoup) -> None:
    """Collapse runs of whitespace inside text nodes; keep block boundaries."""
    for element in soup.find_all(True):
        # Iterate children (NavigableString + Tag) in place
        new_children: list = []
        for child in list(element.children):
            if isinstance(child, NavigableString):
                # Collapse multiple whitespace to single space, keep newlines.
                cleaned = re.sub(r"[ \t ]+", " ", str(child))
                cleaned = re.sub(r"\n[ \t]*", "\n", cleaned)
                cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
                if cleaned.strip():
                    new_children.append(NavigableString(cleaned))
                # else: drop empty text node
            else:
                new_children.append(child)
        # Replace children — preserve at least one if anything remains
        if new_children and new_children != list(element.children):
            element.clear()
            for child in new_children:
                element.append(child)


# ---------------------------------------------------------------------------
# Reporting helpers (used by the demo entrypoint)
# ---------------------------------------------------------------------------


def summarise(doc: CleanedDocument) -> Iterable[str]:
    """Yield human-friendly lines describing the cleaning result."""
    yield f"Kích thước gốc    : {doc.raw_size:>8,} ký tự"
    yield f"Kích thước sau   : {doc.cleaned_size:>8,} ký tự"
    yield f"Giảm             : {doc.reduction_pct:>8} %"
    if doc.removed_tags:
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(doc.removed_tags.items()))
        yield f"Tag đã loại bỏ   : {pairs}"
    else:
        yield "Tag đã loại bỏ   : (không có)"