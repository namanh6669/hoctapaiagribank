#!/usr/bin/env python3
"""Pipeline demo Buổi 5: đọc PDF, OCR fallback và thử nghiệm chunking.

Phạm vi cố ý đơn giản cho bài học:
- Không tạo embedding.
- Không lưu vector database.
- Không gọi LLM.
- Không ghi đè PDF gốc trong datademo/.

Luồng chính:
1. Đọc PDF trong datademo/.
2. Thử lấy text layer bằng PyMuPDF.
3. Nếu PyMuPDF không dùng được hoặc text có dấu hiệu lỗi/rỗng, render trang ra ảnh
   để kiểm tra fallback và gửi toàn bộ file PDF lên LlamaParse OCR.
4. Chuẩn hoá Unicode NFC.
5. Ghi raw text + chunk + báo cáo vào output/ khi dùng --write.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pymupdf  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - lỗi được báo trong runtime
    pymupdf = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "datademo"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"
PDF_GLOB = "*.pdf"
LANGUAGE = "vi"

# Không đọc/in giá trị key; chỉ kiểm tra key có tồn tại hay không.
LLAMA_KEY_NAMES = ("LLAMA_CLOUD_API_KEY", "LLAMA_PARSE_API_KEY", "LLAMA_API_KEY")


@dataclass
class PageRecord:
    source: str
    page: int
    text: str
    ocr_used: bool
    language: str = LANGUAGE
    warnings: list[str] = field(default_factory=list)


@dataclass
class RawDocument:
    source: str
    pages: list[PageRecord]
    extraction_method: str
    warnings: list[str] = field(default_factory=list)
    raw_markdown_full: str | None = None


@dataclass
class ChunkRecord:
    chunk_id: str
    strategy: str
    source: str
    page_start: int
    page_end: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyStats:
    strategy: str
    chunk_count: int
    min_length: int
    max_length: int
    avg_length: float


@dataclass
class PipelineReport:
    generated_at: str
    dry_run: bool
    documents: list[dict[str, Any]]
    stats: list[StrategyStats]
    warnings: list[str]
    output_dir: str


def normalize_nfc(text: str) -> str:
    """Chuẩn hoá Unicode NFC và xuống dòng kiểu Unix."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def compact_spaces(text: str) -> str:
    """Giữ ngắt đoạn nhưng gom khoảng trắng thừa trong từng dòng."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return normalize_nfc("\n".join(lines)).strip()


def text_quality_warnings(text: str) -> list[str]:
    """Phát hiện text layer rỗng/lỗi font/lỗi encoding/ký tự lạ ở mức heuristic."""
    warnings: list[str] = []
    stripped = text.strip()
    if not stripped:
        return ["text_layer_empty"]

    total = len(stripped)
    replacement_count = stripped.count("�")
    control_count = sum(1 for ch in stripped if unicodedata.category(ch) in {"Cc", "Cf"} and ch not in "\n\t")
    private_count = sum(1 for ch in stripped if unicodedata.category(ch) == "Co")
    letter_count = sum(1 for ch in stripped if ch.isalpha())
    printable_count = sum(1 for ch in stripped if ch.isprintable() or ch in "\n\t")

    if replacement_count:
        warnings.append("replacement_character_found")
    if control_count / total > 0.01:
        warnings.append("too_many_control_characters")
    if private_count / total > 0.005:
        warnings.append("private_use_font_glyphs_found")
    if printable_count / total < 0.95:
        warnings.append("too_many_non_printable_characters")
    if letter_count < 20 and total > 80:
        warnings.append("too_few_letters_for_text_page")

    # Dấu hiệu mojibake hay gặp khi tiếng Việt bị decode sai.
    mojibake_patterns = ("Ã", "Â", "Ä", "Æ", "Ð", "ð", "þ", "\x00")
    mojibake_hits = sum(stripped.count(pattern) for pattern in mojibake_patterns)
    if mojibake_hits / max(total, 1) > 0.01:
        warnings.append("possible_encoding_or_font_error")

    # Một số PDF tiếng Việt nhúng font khiến PyMuPDF trả text kiểu "CQNG HOAXA",
    # "DQc lQp", "th6ng tu"... Dùng từ khoá lỗi phổ biến để bắt fallback OCR.
    broken_vietnamese_patterns = (
        "CQNG",
        "HOAXA",
        "HQI",
        "NGHiA",
        "DQc",
        "Hqnh",
        "phric",
        "Th6ng",
        "tli6u",
        "ngdn",
        "kh6",
        "nhrlnh",
        "nudc",
    )
    broken_hits = sum(stripped.count(pattern) for pattern in broken_vietnamese_patterns)
    vietnamese_diacritics = sum(1 for ch in stripped if ch in "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ")
    if broken_hits >= 3 and vietnamese_diacritics / max(letter_count, 1) < 0.02:
        warnings.append("possible_vietnamese_font_substitution_error")

    # Nhiều ký tự đơn lẻ cách nhau bất thường thường là text layer bị tách glyph.
    single_char_tokens = re.findall(r"(?<!\w)\w(?!\w)", stripped)
    tokens = re.findall(r"\w+", stripped)
    if len(tokens) >= 30 and len(single_char_tokens) / len(tokens) > 0.65:
        warnings.append("possible_broken_glyph_spacing")

    return warnings


def page_records_from_pymupdf(pdf_path: Path) -> tuple[list[PageRecord], list[str]]:
    """Trích text từng trang bằng PyMuPDF và trả về cảnh báo chất lượng."""
    if pymupdf is None:
        return [], ["pymupdf_import_failed"]

    warnings: list[str] = []
    records: list[PageRecord] = []
    try:
        with pymupdf.open(pdf_path) as doc:
            if doc.page_count == 0:
                return [], ["pdf_has_no_pages"]
            for index, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text")
                except Exception as exc:  # noqa: BLE001 - cần fallback OCR rõ ràng
                    text = ""
                    page_warnings = [f"pymupdf_page_extract_failed:{exc.__class__.__name__}"]
                else:
                    text = compact_spaces(text)
                    page_warnings = text_quality_warnings(text)
                records.append(
                    PageRecord(
                        source=pdf_path.name,
                        page=index,
                        text=normalize_nfc(text),
                        ocr_used=False,
                        warnings=page_warnings,
                    )
                )
                warnings.extend(f"page_{index}:{warning}" for warning in page_warnings)
    except Exception as exc:  # noqa: BLE001
        return [], [f"pymupdf_open_failed:{exc.__class__.__name__}"]

    return records, warnings


def has_unusable_page(records: list[PageRecord], warnings: list[str]) -> bool:
    """Nếu một trang có vấn đề thì OCR lại toàn bộ file, đúng yêu cầu bài."""
    if not records:
        return True
    if warnings:
        return True
    return any(not record.text.strip() or record.warnings for record in records)


def render_pdf_pages(pdf_path: Path, render_dir: Path, dpi: int = 180) -> list[str]:
    """Render trang PDF ra ảnh PNG để minh hoạ/kiểm tra fallback OCR.

    Ảnh chỉ được ghi trong output/, không động đến PDF gốc.
    """
    rendered: list[str] = []
    if pymupdf is None:
        return rendered

    render_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = render_dir / f"page_{index:04d}.png"
            pixmap.save(image_path)
            rendered.append(str(image_path.relative_to(PROJECT_DIR)))
    return rendered


def load_llama_api_key_presence() -> str | None:
    """Load .env trong src/ nhưng không trả/in giá trị key."""
    if load_dotenv is not None:
        load_dotenv(SRC_DIR / ".env", override=False)

    import os

    for name in LLAMA_KEY_NAMES:
        if os.getenv(name):
            return name
    return None


def get_llama_api_key() -> str:
    """Lấy key để gọi API nhưng không log giá trị."""
    if load_dotenv is not None:
        load_dotenv(SRC_DIR / ".env", override=False)

    import os

    for name in LLAMA_KEY_NAMES:
        value = os.getenv(name)
        if value:
            return value
    names = ", ".join(LLAMA_KEY_NAMES)
    raise RuntimeError(f"Không tìm thấy API key trong src/.env hoặc môi trường. Cần một trong: {names}")


async def llamaparse_pdf_to_markdown(pdf_path: Path) -> str:
    """Gửi toàn bộ PDF lên LlamaParse và lấy markdown_full.

    Dùng đúng kiểu client async từ llama-cloud; không gọi LLM riêng.
    """
    try:
        from llama_cloud import AsyncLlamaCloud
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Không import được llama_cloud: {exc}") from exc

    client = AsyncLlamaCloud(api_key=get_llama_api_key())
    file_obj = await client.files.create(file=str(pdf_path), purpose="parse")
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        expand=["markdown_full"],
    )
    markdown = getattr(result, "markdown_full", None)
    if not markdown:
        raise RuntimeError("LlamaParse không trả về markdown_full")
    return normalize_nfc(str(markdown))


def split_markdown_to_pages(markdown: str, source: str) -> list[PageRecord]:
    """Tách markdown OCR thành page records nếu có marker trang; nếu không có thì page=1."""
    text = normalize_nfc(markdown).strip()
    if not text:
        return [PageRecord(source=source, page=1, text="", ocr_used=True, warnings=["llamaparse_empty_markdown"])]

    page_pattern = re.compile(r"(?:^|\n)\s*(?:-{0,3}\s*)?(?:page|trang)\s+(\d+)\s*(?:-{0,3})?\s*\n", re.IGNORECASE)
    matches = list(page_pattern.finditer(text))
    if not matches:
        return [PageRecord(source=source, page=1, text=compact_spaces(text), ocr_used=True)]

    records: list[PageRecord] = []
    for idx, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_text = compact_spaces(text[start:end])
        records.append(PageRecord(source=source, page=page_number, text=page_text, ocr_used=True))
    return records


async def extract_raw_document(pdf_path: Path, output_dir: Path, *, write: bool, force_ocr: bool = False) -> RawDocument:
    """Trích raw text từ PDF bằng PyMuPDF hoặc OCR fallback."""
    py_records, py_warnings = page_records_from_pymupdf(pdf_path)
    warnings = list(py_warnings)

    should_ocr = force_ocr or has_unusable_page(py_records, py_warnings)
    if not should_ocr:
        return RawDocument(source=pdf_path.name, pages=py_records, extraction_method="pymupdf", warnings=warnings)

    warnings.append("fallback_to_llamaparse_for_whole_file")
    if write:
        try:
            rendered = render_pdf_pages(pdf_path, output_dir / "rendered_pages" / pdf_path.stem)
            if rendered:
                warnings.append(f"rendered_pages:{len(rendered)}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"render_pages_failed:{exc.__class__.__name__}")

    markdown = await llamaparse_pdf_to_markdown(pdf_path)
    ocr_records = split_markdown_to_pages(markdown, pdf_path.name)
    for record in ocr_records:
        record.text = normalize_nfc(record.text)
        record.ocr_used = True
        record.language = LANGUAGE
    return RawDocument(
        source=pdf_path.name,
        pages=ocr_records,
        extraction_method="llamaparse_ocr",
        warnings=warnings,
        raw_markdown_full=markdown,
    )


def iter_pdf_paths(data_dir: Path, only: list[str] | None = None) -> list[Path]:
    if only:
        paths = [data_dir / name for name in only]
    else:
        paths = sorted(data_dir.glob(PDF_GLOB))
    return [path for path in paths if path.suffix.lower() == ".pdf"]


def pages_to_full_text(pages: list[PageRecord]) -> str:
    blocks = [f"\n\n[PAGE {page.page}]\n{page.text}" for page in pages]
    return normalize_nfc("".join(blocks)).strip()


def find_page_span(text: str, pages: list[PageRecord]) -> tuple[int, int]:
    """Ước lượng page_start/page_end cho chunk theo nội dung."""
    page_numbers: list[int] = []
    remaining = text
    for page in pages:
        sample = page.text[: min(len(page.text), 120)].strip()
        if sample and sample in remaining:
            page_numbers.append(page.page)
        elif page.text and any(part and part in page.text for part in re.split(r"\n+", text[:300])[:3]):
            page_numbers.append(page.page)
    if not page_numbers:
        return pages[0].page, pages[-1].page
    return min(page_numbers), max(page_numbers)


def chunk_fixed_size(doc: RawDocument, *, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    text = pages_to_full_text(doc.pages)
    if not text:
        return chunks

    start = 0
    index = 1
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = normalize_nfc(text[start:end]).strip()
        if chunk_text:
            page_start, page_end = find_page_span(chunk_text, doc.pages)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{Path(doc.source).stem}::fixed::{index:04d}",
                    strategy="fixed-size",
                    source=doc.source,
                    page_start=page_start,
                    page_end=page_end,
                    text=chunk_text,
                    metadata={"chunk_size": chunk_size, "overlap": overlap, "char_start": start, "char_end": end},
                )
            )
            index += 1
        if end == len(text):
            break
        start += step
    return chunks


def paragraph_blocks(page: PageRecord) -> list[str]:
    text = normalize_nfc(page.text).strip()
    if not text:
        return []
    blocks = re.split(r"\n\s*\n+", text)
    if len(blocks) == 1:
        # Fallback cho PDF chỉ có single newline: gom các dòng liên tiếp thành đoạn nhỏ.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = []
        current: list[str] = []
        for line in lines:
            current.append(line)
            if re.search(r"[.!?;:]$", line) or len(" ".join(current)) > 700:
                blocks.append(" ".join(current))
                current = []
        if current:
            blocks.append(" ".join(current))
    return [compact_spaces(block) for block in blocks if block.strip()]


def chunk_semantic(doc: RawDocument, *, target_size: int, max_size: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    index = 1

    def flush(reason: str) -> None:
        nonlocal buffer, page_start, page_end, index
        if not buffer or page_start is None or page_end is None:
            return
        text = normalize_nfc("\n\n".join(buffer)).strip()
        for part in split_large_text(text, max_size):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{Path(doc.source).stem}::semantic::{index:04d}",
                    strategy="semantic",
                    source=doc.source,
                    page_start=page_start,
                    page_end=page_end,
                    text=part,
                    metadata={"boundary": reason, "target_size": target_size, "max_size": max_size},
                )
            )
            index += 1
        buffer = []
        page_start = None
        page_end = None

    for page in doc.pages:
        for block in paragraph_blocks(page):
            for block_part in split_large_text(block, max_size):
                block_len = len(block_part)
                current_len = len("\n\n".join(buffer))
                if buffer and current_len >= target_size:
                    flush("paragraph_or_blank_line")
                elif buffer and current_len + block_len > max_size:
                    flush("max_size_guard_at_paragraph_boundary")

                if page_start is None:
                    page_start = page.page
                page_end = page.page
                buffer.append(block_part)

                if len(block_part) >= max_size:
                    flush("single_large_paragraph")
    flush("end_of_document")
    return chunks


HEADING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chapter", re.compile(r"^\s*(CHƯƠNG|Chương)\s+([IVXLCDM]+|\d+)\b.*", re.IGNORECASE)),
    ("section", re.compile(r"^\s*(MỤC|Mục)\s+([IVXLCDM]+|\d+)\b.*", re.IGNORECASE)),
    ("article", re.compile(r"^\s*(ĐIỀU|Điều)\s+\d+\b.*", re.IGNORECASE)),
    ("clause", re.compile(r"^\s*\d+\.\s+.+")),
    ("point", re.compile(r"^\s*[a-zđ]\)\s+.+", re.IGNORECASE)),
)


def detect_heading(line: str) -> tuple[str, str] | None:
    clean = compact_spaces(line)
    if not clean or len(clean) > 220:
        return None
    clean = clean.strip("#* _")
    for level, pattern in HEADING_PATTERNS:
        if pattern.match(clean):
            return level, clean
    return None


def split_large_text(text: str, max_size: int) -> list[str]:
    """Cắt đoạn quá lớn theo câu/dòng, không cắt giữa chừng nếu tránh được."""
    text = normalize_nfc(text).strip()
    if len(text) <= max_size:
        return [text] if text else []

    units = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    parts: list[str] = []
    buffer: list[str] = []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_size:
            if buffer:
                parts.append(" ".join(buffer).strip())
                buffer = []
            for start in range(0, len(unit), max_size):
                parts.append(unit[start : start + max_size].strip())
            continue

        candidate = (" ".join(buffer + [unit])).strip()
        if buffer and len(candidate) > max_size:
            parts.append(" ".join(buffer).strip())
            buffer = [unit]
        else:
            buffer.append(unit)
    if buffer:
        parts.append(" ".join(buffer).strip())
    return [part for part in parts if part]


def chunk_hierarchical(doc: RawDocument, *, max_size: int) -> tuple[list[ChunkRecord], list[str]]:
    chunks: list[ChunkRecord] = []
    warnings: list[str] = []
    current_lines: list[str] = []
    current_meta: dict[str, Any] = {"heading_path": []}
    current_start_page: int | None = None
    current_end_page: int | None = None
    heading_path: dict[str, str | None] = {"chapter": None, "section": None, "article": None, "clause": None, "point": None}
    index = 1
    heading_found = False

    order = ["chapter", "section", "article", "clause", "point"]

    def path_list() -> list[dict[str, str]]:
        return [{"level": key, "title": value} for key, value in heading_path.items() if value]

    def reset_lower(level: str) -> None:
        if level not in order:
            return
        start = order.index(level) + 1
        for key in order[start:]:
            heading_path[key] = None

    def flush(reason: str) -> None:
        nonlocal current_lines, current_meta, current_start_page, current_end_page, index
        if not current_lines or current_start_page is None or current_end_page is None:
            return
        text = normalize_nfc("\n".join(current_lines)).strip()
        metadata = dict(current_meta)
        metadata["boundary"] = reason
        for part in split_large_text(text, max_size):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{Path(doc.source).stem}::hierarchical::{index:04d}",
                    strategy="hierarchical",
                    source=doc.source,
                    page_start=current_start_page,
                    page_end=current_end_page,
                    text=part,
                    metadata=metadata,
                )
            )
            index += 1
        current_lines = []
        current_meta = {"heading_path": path_list()}
        current_start_page = None
        current_end_page = None

    for page in doc.pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        for line in lines:
            heading = detect_heading(line)
            if heading:
                level, title = heading
                if level == "clause" and not heading_path["article"]:
                    heading = None
                elif level == "point" and not (heading_path["clause"] or heading_path["article"]):
                    heading = None

            if heading:
                heading_found = True
                level, title = heading
                flush(f"new_{level}")
                reset_lower(level)
                heading_path[level] = title
                current_meta = {"heading_level": level, "heading": title, "heading_path": path_list()}
            elif not current_lines and current_meta == {"heading_path": []}:
                current_meta = {"heading_path": path_list()}

            if current_start_page is None:
                current_start_page = page.page
            current_end_page = page.page
            current_lines.append(line)

            if len("\n".join(current_lines)) >= max_size:
                flush("max_size_guard_inside_structure")
    flush("end_of_document")

    if not heading_found:
        warnings.append("Không phát hiện cấu trúc Chương/Mục/Điều/Khoản/Điểm; hierarchical không bịa heading, chỉ dùng chunk tuần tự theo nội dung.")
    return chunks, warnings


def compute_stats(chunks: Iterable[ChunkRecord]) -> list[StrategyStats]:
    grouped: dict[str, list[int]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.strategy, []).append(len(chunk.text))
    stats: list[StrategyStats] = []
    for strategy, lengths in sorted(grouped.items()):
        stats.append(
            StrategyStats(
                strategy=strategy,
                chunk_count=len(lengths),
                min_length=min(lengths) if lengths else 0,
                max_length=max(lengths) if lengths else 0,
                avg_length=round(statistics.mean(lengths), 2) if lengths else 0.0,
            )
        )
    return stats


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def write_raw_outputs(doc: RawDocument, output_dir: Path) -> None:
    stem = Path(doc.source).stem
    raw_dir = output_dir / "raw"
    write_json(raw_dir / f"{stem}.pages.json", [asdict(page) for page in doc.pages])
    write_json(
        raw_dir / f"{stem}.meta.json",
        {
            "source": doc.source,
            "extraction_method": doc.extraction_method,
            "warnings": doc.warnings,
            "language": LANGUAGE,
            "page_count": len(doc.pages),
        },
    )
    text = "\n\n".join(f"[PAGE {page.page}]\n{page.text}" for page in doc.pages)
    (raw_dir / f"{stem}.txt").write_text(normalize_nfc(text), encoding="utf-8")
    if doc.raw_markdown_full is not None:
        (raw_dir / f"{stem}.llamaparse.md").write_text(doc.raw_markdown_full, encoding="utf-8")


def build_chunks(documents: list[RawDocument], args: argparse.Namespace) -> tuple[list[ChunkRecord], list[str]]:
    chunks: list[ChunkRecord] = []
    warnings: list[str] = []
    for doc in documents:
        chunks.extend(chunk_fixed_size(doc, chunk_size=args.fixed_size, overlap=args.overlap))
        chunks.extend(chunk_semantic(doc, target_size=args.semantic_target, max_size=args.semantic_max))
        hierarchical, hierarchical_warnings = chunk_hierarchical(doc, max_size=args.hierarchical_max)
        chunks.extend(hierarchical)
        warnings.extend(f"{doc.source}: {warning}" for warning in hierarchical_warnings)
    return chunks, warnings


def print_stats(stats: list[StrategyStats]) -> None:
    print("\nThống kê chunk theo chiến lược:")
    if not stats:
        print("- Chưa có chunk nào.")
        return
    for item in stats:
        print(
            f"- {item.strategy}: count={item.chunk_count}, "
            f"min={item.min_length}, max={item.max_length}, avg={item.avg_length} ký tự"
        )


def example_metadata(chunks: list[ChunkRecord]) -> dict[str, Any] | None:
    for chunk in chunks:
        if chunk.metadata:
            return asdict(chunk)
    return asdict(chunks[0]) if chunks else None


async def run_pipeline(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    pdf_paths = iter_pdf_paths(data_dir, args.pdf)
    warnings: list[str] = []

    if not pdf_paths:
        print(f"Không tìm thấy PDF trong {data_dir}")
        return 1

    key_name = load_llama_api_key_presence()
    if key_name:
        print(f"Đã thấy API key LlamaParse qua biến {key_name} (không in giá trị).")
    else:
        print("Chưa thấy API key LlamaParse; chỉ chạy được PDF có text layer tốt, hoặc sẽ lỗi khi cần OCR.")

    documents: list[RawDocument] = []
    for pdf_path in pdf_paths:
        print(f"\nĐang xử lý: {pdf_path.name}")
        try:
            doc = await extract_raw_document(pdf_path, output_dir, write=args.write, force_ocr=args.force_ocr)
        except Exception as exc:  # noqa: BLE001
            warning = f"{pdf_path.name}: extraction_failed:{exc.__class__.__name__}: {exc}"
            warnings.append(warning)
            print(f"  FAIL: {warning}")
            continue
        documents.append(doc)
        print(f"  method={doc.extraction_method}, pages={len(doc.pages)}, warnings={len(doc.warnings)}")
        for warning in doc.warnings[:8]:
            print(f"  cảnh báo: {warning}")
        if len(doc.warnings) > 8:
            print(f"  ... thêm {len(doc.warnings) - 8} cảnh báo")
        if args.write:
            write_raw_outputs(doc, output_dir)

    if not documents:
        print("Không có tài liệu nào được xử lý thành công.")
        return 1

    chunks, chunk_warnings = build_chunks(documents, args)
    warnings.extend(chunk_warnings)
    stats = compute_stats(chunks)
    print_stats(stats)

    sample = example_metadata(chunks)
    if sample:
        print("\nVí dụ metadata/chunk đầu ra:")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:2000])

    for warning in chunk_warnings:
        print(f"Cảnh báo chunking: {warning}")

    report = PipelineReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dry_run=not args.write,
        documents=[
            {
                "source": doc.source,
                "extraction_method": doc.extraction_method,
                "page_count": len(doc.pages),
                "warnings": doc.warnings,
            }
            for doc in documents
        ],
        stats=stats,
        warnings=warnings,
        output_dir=str(output_dir),
    )

    if args.write:
        write_json(output_dir / "chunks" / "chunks.json", [asdict(chunk) for chunk in chunks])
        write_json(output_dir / "reports" / "chunk_report.json", asdict(report))
        print(f"\nĐã ghi output vào: {output_dir}")
    else:
        print("\nDry-run: chưa ghi raw/chunk/report. Thêm --write để lưu vào output/.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buổi 5 - PDF raw extraction + chunking demo")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Thư mục chứa PDF đầu vào")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Thư mục output để lưu raw/chunk/report")
    parser.add_argument("--pdf", action="append", help="Tên file PDF cụ thể trong data-dir; có thể lặp lại")
    parser.add_argument("--write", action="store_true", help="Ghi output ra đĩa; mặc định chỉ dry-run")
    parser.add_argument("--force-ocr", action="store_true", help="Bỏ qua text layer và OCR toàn bộ file bằng LlamaParse")
    parser.add_argument("--fixed-size", type=int, default=1200, help="Số ký tự mỗi chunk fixed-size")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap ký tự cho fixed-size")
    parser.add_argument("--semantic-target", type=int, default=1200, help="Mục tiêu kích thước chunk semantic")
    parser.add_argument("--semantic-max", type=int, default=1800, help="Ngưỡng tối đa chunk semantic trước khi flush")
    parser.add_argument("--hierarchical-max", type=int, default=2200, help="Ngưỡng tối đa chunk hierarchical")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fixed_size <= 0:
        raise ValueError("--fixed-size phải > 0")
    if args.overlap < 0:
        raise ValueError("--overlap phải >= 0")
    if args.overlap >= args.fixed_size:
        raise ValueError("--overlap phải nhỏ hơn --fixed-size")
    if args.semantic_target <= 0 or args.semantic_max <= 0 or args.hierarchical_max <= 0:
        raise ValueError("Các kích thước semantic/hierarchical phải > 0")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    raise SystemExit(main())
