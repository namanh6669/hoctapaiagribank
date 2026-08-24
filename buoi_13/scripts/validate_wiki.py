#!/usr/bin/env python3
"""Kiểm thử Wiki Risk Graph và sinh outputs/wiki_validation_report.md.

Đầu vào:
    outputs/entities.csv           (đã chuẩn hoá)
    outputs/relations.csv         (đã chuẩn hoá)
    wiki/**/*.md                  (đã sinh bởi build_wiki.py)

Đầu ra:
    outputs/wiki_validation_report.md

Các kiểm tra (9):
    1.  Tổng số file Markdown.
    2.  Tổng số wikilink.
    3.  Wikilink trỏ tới trang không tồn tại.
    4.  Entity bị trùng ID.
    5.  Trang có ID nhưng không tồn tại trong entities.csv.
    6.  Relation có source hoặc target không tồn tại trong entities.csv.
    7.  RuiRo không có bất kỳ KiemSoat nào.
    8.  RuiRo không có bất kỳ SuKienRuiRo nào.
    9.  Trang không có liên kết ra trang khác (orphan page).

Sau khi chạy, cuối báo cáo có phần phân loại:
    - Lỗi chương trình (sửa được bằng cách vá build_wiki.py)
    - Lỗi / đặc điểm dữ liệu (phản ánh đúng data/*.csv, không tự vá)
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO / "wiki"
ENTITIES_PATH = REPO / "outputs" / "entities.csv"
RELATIONS_PATH = REPO / "outputs" / "relations.csv"
REPORT_PATH = REPO / "outputs" / "wiki_validation_report.md"

WIKILINK_RE = re.compile(r"(?<!\\)\[\[([^\]]+?)\]\]")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Tách YAML frontmatter (rất đơn giản), trả về (dict, phần còn lại)."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    fm: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            fm[key.strip()] = val
    return fm, content[m.end():]


def iter_md_files() -> list[Path]:
    return sorted(WIKI_DIR.rglob("*.md"))


def resolve_wikilink(target: str) -> Path | None:
    """Trả về Path nếu `[[target]]` resolve được trong wiki/, ngược lại None."""
    t = target.strip()
    if "/" in t:
        cand = WIKI_DIR / f"{t}.md"
        return cand if cand.exists() else None
    matches = sorted(WIKI_DIR.rglob(f"{t}.md"))
    return matches[0] if matches else None


def main() -> int:
    # ===== Đọc dữ liệu nguồn =====
    entities = read_csv(ENTITIES_PATH)
    relations = read_csv(RELATIONS_PATH)

    entity_by_id: dict[str, dict[str, str]] = {e["id"]: e for e in entities}
    rels_by_src: dict[str, list[dict[str, str]]] = defaultdict(list)
    rels_by_tgt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in relations:
        rels_by_src[r["source_id"]].append(r)
        rels_by_tgt[r["target_id"]].append(r)

    # ===== Thu thập thông tin wiki =====
    md_files = iter_md_files()
    md_relpaths = [p.relative_to(REPO).as_posix() for p in md_files]

    # (4) Duplicate IDs
    id_counter = Counter(e["id"] for e in entities)
    dup_ids = sorted(i for i, c in id_counter.items() if c > 1)

    # (5) Pages có ID nhưng không có trong entities.csv
    pages_with_id: list[tuple[str, str]] = []  # (relpath, id)
    pages_id_unknown: list[tuple[str, str]] = []
    for path in md_files:
        content = path.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(content)
        eid = fm.get("id", "").strip()
        if eid:
            pages_with_id.append((path.relative_to(REPO).as_posix(), eid))
            if eid not in entity_by_id:
                pages_id_unknown.append((path.relative_to(REPO).as_posix(), eid))

    # (6) Relation không resolve
    orphan_rel_src = [r for r in relations
                      if r["source_id"] and r["source_id"] not in entity_by_id]
    orphan_rel_tgt = [r for r in relations
                      if r["target_id"] and r["target_id"] not in entity_by_id]

    # (7) RuiRo không có KiemSoat
    ruiro_no_control: list[str] = []
    for e in entities:
        if e["type"] != "RuiRo":
            continue
        has_mit = any(
            r.get("relationship_type") == "MITIGATES"
            for r in rels_by_tgt.get(e["id"], [])
        )
        if not has_mit:
            ruiro_no_control.append(e["id"])

    # (8) RuiRo không có SuKienRuiRo
    ruiro_no_event: list[str] = []
    for e in entities:
        if e["type"] != "RuiRo":
            continue
        has_obs = any(
            r.get("relationship_type") == "OBSERVED_AS"
            for r in rels_by_src.get(e["id"], [])
        )
        if not has_obs:
            ruiro_no_event.append(e["id"])

    # (1), (2), (3), (9) — wikilink + orphan
    total_wikilinks = 0
    broken_wikilinks: list[tuple[str, str]] = []  # (source_relpath, target)
    outgoing_per_page: dict[str, int] = defaultdict(int)
    for path in md_files:
        content = path.read_text(encoding="utf-8")
        _, body = parse_frontmatter(content)
        targets = WIKILINK_RE.findall(body)
        n = len(targets)
        total_wikilinks += n
        rel = path.relative_to(REPO).as_posix()
        outgoing_per_page[rel] = n
        for raw in targets:
            target = raw.split("|", 1)[0].strip()
            if resolve_wikilink(target) is None:
                broken_wikilinks.append((rel, target))

    # (9) orphan = page có 0 outgoing (kể cả README/Home vì nếu chúng không link ra, đó là bug)
    orphan_pages = [
        rel for rel in md_relpaths if outgoing_per_page.get(rel, 0) == 0
    ]

    # ===== Tạo báo cáo Markdown =====
    L: list[str] = []
    L += ["# Wiki Risk Graph — Validation Report", ""]
    L.append(
        "_Báo cáo này được sinh tự động bởi `scripts/validate_wiki.py`, "
        "đọc `outputs/entities.csv`, `outputs/relations.csv` và `wiki/**/*.md`._"
    )
    L.append("")

    # Tóm tắt
    L += ["## Tóm tắt", ""]
    L.append(f"- Tổng file Markdown: **{len(md_files)}**")
    L.append(f"- Tổng entity trong CSV: **{len(entities)}** "
             f"({sum(1 for e in entities if e['type'] == 'RuiRo')} RuiRo, "
             f"{sum(1 for e in entities if e['type'] == 'KiemSoat')} KiemSoat, "
             f"{sum(1 for e in entities if e['type'] == 'SuKienRuiRo')} SuKienRuiRo)")
    L.append(f"- Tổng relation trong CSV: **{len(relations)}**")
    L.append(f"- Tổng wikilink trong body các trang: **{total_wikilinks}**")
    L.append("")

    def status(num: int, title: str, ok: bool, bullets: list[str] | None = None) -> None:
        L.append(f"### {num}. {title}")
        L.append("")
        L.append("✅ **OK**" if ok else "❌ **Có vấn đề**")
        if bullets:
            for b in bullets:
                L.append(f"- {b}")
        L.append("")

    status(1, "Tổng số file Markdown", True,
           [f"Đếm được **{len(md_files)}** file (bao gồm `Home.md`, README và các trang entity)."])

    status(2, "Tổng số wikilink", True,
           [f"Tổng `[[...]]` trong **body** các trang: **{total_wikilinks}**."])

    # 3
    if not broken_wikilinks:
        status(3, "Wikilink trỏ tới trang không tồn tại", True,
               ["Mọi `[[target]]` đều resolve được trong `wiki/`."])
    else:
        rows = "\n".join(
            f"| `{src}` | `[[{tgt}]]` |" for src, tgt in broken_wikilinks
        )
        L += [
            "### 3. Wikilink trỏ tới trang không tồn tại",
            "",
            f"❌ **Có {len(broken_wikilinks)} wikilink lỗi:**",
            "",
            "| Trang nguồn | Target |",
            "| --- | --- |",
            rows,
            "",
        ]

    # 4
    if not dup_ids:
        status(4, "Entity bị trùng ID trong `entities.csv`", True)
    else:
        L += [
            "### 4. Entity bị trùng ID trong `entities.csv`",
            "",
            f"❌ **Có {len(dup_ids)} ID trùng:**",
            "",
        ]
        for i in dup_ids:
            L.append(f"- `{i}` ({id_counter[i]} lần)")
        L.append("")

    # 5
    if not pages_id_unknown:
        status(5, "Trang wiki có ID không tồn tại trong `entities.csv`", True,
               [f"Đã kiểm tra **{len(pages_with_id)}** trang có frontmatter `id`."])
    else:
        L += [
            "### 5. Trang wiki có ID không tồn tại trong `entities.csv`",
            "",
            f"❌ **Có {len(pages_id_unknown)} trang có ID lạ:**",
            "",
            "| Trang | ID |",
            "| --- | --- |",
        ]
        for path, eid in pages_id_unknown:
            L.append(f"| `{path}` | `{eid}` |")
        L.append("")

    # 6
    if not orphan_rel_src and not orphan_rel_tgt:
        status(6, "Relation có source/target không tồn tại", True)
    else:
        L += [
            "### 6. Relation có source/target không tồn tại trong `entities.csv`",
            "",
            f"❌ **Có {len(orphan_rel_src) + len(orphan_rel_tgt)} vi phạm:**",
            "",
        ]
        if orphan_rel_src:
            L.append(f"- Orphan `source_id` ({len(orphan_rel_src)}):")
            for r in orphan_rel_src[:10]:
                L.append(
                    f"  - `{r['source_id']}` → `{r['target_id']}` ({r['relationship_type']})"
                )
        if orphan_rel_tgt:
            L.append(f"- Orphan `target_id` ({len(orphan_rel_tgt)}):")
            for r in orphan_rel_tgt[:10]:
                L.append(
                    f"  - `{r['source_id']}` → `{r['target_id']}` ({r['relationship_type']})"
                )
        L.append("")

    # 7
    if not ruiro_no_control:
        status(7, "RuiRo không có KiemSoat nào", True)
    else:
        L += [
            "### 7. RuiRo không có KiemSoat nào",
            "",
            f"⚠️  **Dữ liệu:** Có **{len(ruiro_no_control)}** RuiRo chưa có kiểm soát "
            f"trong `relations.csv` (phản ánh đúng nguồn — không tự vá):",
            "",
        ]
        for rid in ruiro_no_control:
            L.append(f"- `{rid}`")
        L.append("")

    # 8
    if not ruiro_no_event:
        status(8, "RuiRo không có SuKienRuiRo nào", True)
    else:
        L += [
            "### 8. RuiRo không có SuKienRuiRo nào",
            "",
            f"❌ **Có {len(ruiro_no_event)} RuiRo không có sự kiện:**",
            "",
        ]
        for rid in ruiro_no_event:
            L.append(f"- `{rid}`")
        L.append("")

    # 9
    if not orphan_pages:
        status(9, "Trang không có liên kết ra trang khác (orphan)", True)
    else:
        L += [
            "### 9. Trang không có liên kết ra trang khác (orphan)",
            "",
            f"❌ **Có {len(orphan_pages)} trang không có wikilink nào trong body:**",
            "",
        ]
        for p in orphan_pages:
            L.append(f"- `{p}`")
        L.append("")

    # ===== Phân loại lỗi =====
    L += ["## Phân loại lỗi", ""]
    code_bugs: list[str] = []
    data_issues: list[str] = []

    if broken_wikilinks:
        code_bugs.append(
            f"#3 wikilink lỗi — **{len(broken_wikilinks)}** liên kết "
            f"không resolve được (nghi code `build_wiki.py` sinh link sai)."
        )
    if pages_id_unknown:
        code_bugs.append(
            f"#5 trang có `id` không có trong `entities.csv` — "
            f"**{len(pages_id_unknown)}** trang (nghi code sinh ID lạ)."
        )
    if orphan_rel_src or orphan_rel_tgt:
        code_bugs.append(
            f"#6 relation có source/target không khớp entities — "
            f"**{len(orphan_rel_src) + len(orphan_rel_tgt)}** dòng."
        )
    if orphan_pages:
        code_bugs.append(
            f"#9 trang orphan — **{len(orphan_pages)}** trang "
            f"không có wikilink nào trong body."
        )
    if dup_ids:
        data_issues.append(
            f"#4 ID trùng trong `entities.csv` — **{len(dup_ids)}** ID."
        )
    if ruiro_no_control:
        data_issues.append(
            f"#7 RuiRo không có kiểm soát — **{len(ruiro_no_control)}**: "
            f"{', '.join(ruiro_no_control)}."
        )
    if ruiro_no_event:
        data_issues.append(
            f"#8 RuiRo không có sự kiện — **{len(ruiro_no_event)}**: "
            f"{', '.join(ruiro_no_event)}."
        )

    L += ["### Lỗi chương trình (có thể vá bằng cách sửa `build_wiki.py`)", ""]
    if not code_bugs:
        L.append("- _(không có)_")
    else:
        for b in code_bugs:
            L.append(f"- {b}")
    L.append("")

    L += ["### Lỗi / đặc điểm dữ liệu (phản ánh đúng `data/*.csv`, KHÔNG tự vá)", ""]
    if not data_issues:
        L.append("- _(không có)_")
    else:
        for d in data_issues:
            L.append(f"- {d}")
    L.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")

    # Tóm tắt console
    print("=" * 70)
    print(" Wiki Risk Graph — Validate Wiki ")
    print("=" * 70)
    print(f"\nĐã ghi: {REPORT_PATH}")
    print()
    print(f"File MD       : {len(md_files)}")
    print(f"Wikilink (body): {total_wikilinks}")
    print(f"Broken wiki   : {len(broken_wikilinks)}")
    print(f"Dup IDs       : {len(dup_ids)}")
    print(f"ID unknown    : {len(pages_id_unknown)}")
    print(f"Orphan rels   : src={len(orphan_rel_src)} tgt={len(orphan_rel_tgt)}")
    print(f"RuiRo no KS   : {len(ruiro_no_control)} -> {ruiro_no_control}")
    print(f"RuiRo no SK   : {len(ruiro_no_event)} -> {ruiro_no_event}")
    print(f"Orphan pages  : {len(orphan_pages)} -> {orphan_pages}")
    print()
    print("Phân loại lỗi:")
    print(f"  - Lỗi chương trình: {len(code_bugs)}")
    print(f"  - Lỗi dữ liệu    : {len(data_issues)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
