#!/usr/bin/env python3
"""Sinh wiki/ (Obsidian-flavored Markdown) từ outputs/entities.csv + relations.csv.

Cấu trúc đầu ra:
    wiki/
    ├── Home.md
    ├── risks/
    │   ├── README.md
    │   ├── RR-001.md      (một file cho mỗi RuiRo)
    │   └── ...
    ├── controls/
    │   ├── README.md
    │   ├── KS-001.md      (một file cho mỗi KiemSoat)
    │   └── ...
    └── events/
        ├── README.md
        ├── SK-001.md      (một file cho mỗi SuKienRuiRo)
        └── ...

Quy ước:
- Tên file = ID (RR/KS/SK + số) — an toàn filesystem, không phụ thuộc locale.
- Obsidian wikilink `[[ID]]` hoặc `[[ID|Friendly Name]]` — resolve theo basename
  theo mặc định của Obsidian, hiển thị theo alias.
- H1 trong mỗi trang = tên thân thiện (Obsidian dùng làm page title).
- Quan hệ dựng duy nhất từ `relations.csv` — không tự bịa.
- owner_unit_id / owner_role_id giữ nguyên mã, không suy đoán tên.
- Không tự đổi `verification_status`.

Cách chạy:
    python scripts/build_wiki.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENTITIES_PATH = REPO / "outputs" / "entities.csv"
RELATIONS_PATH = REPO / "outputs" / "relations.csv"
WIKI_DIR = REPO / "wiki"

SUB_RISKS = WIKI_DIR / "risks"
SUB_CONTROLS = WIKI_DIR / "controls"
SUB_EVENTS = WIKI_DIR / "events"

HEADER = "=" * 70


# ===== I/O =====
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_entities(rows: list[dict[str, str]]) -> tuple[dict, dict]:
    by_id = {r["id"]: r for r in rows}
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return by_id, by_type


def index_relations(rows: list[dict[str, str]]) -> tuple[list, dict, dict]:
    by_src: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_tgt: dict[str, list[dict[str, str]]] = defaultdict(list)
    edges = list(rows)
    for r in rows:
        by_src[r["source_id"]].append(r)
        by_tgt[r["target_id"]].append(r)
    return edges, by_src, by_tgt


# ===== YAML & Markdown helpers =====
def yaml_value(v: str) -> str:
    """Serialize một scalar cho YAML frontmatter một cách an toàn."""
    s = "" if v is None else str(v)
    if not s:
        return '""'
    needs_quote = (
        any(c in s for c in (":", "#", '"', "'", "\n", "\\"))
        or s.lower() in {"null", "true", "false", "yes", "no"}
        or s.strip() != s
        or s[:1] in {"-", "?", "*", "&", "!", "|", ">", "%", "@", "`"}
    )
    if needs_quote:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def yaml_frontmatter(props: dict) -> str:
    """Trả về khối YAML frontmatter, bỏ qua field rỗng."""
    lines = ["---"]
    for key, val in props.items():
        if val in (None, ""):
            continue
        lines.append(f"{key}: {yaml_value(val)}")
    lines.append("---")
    return "\n".join(lines)


def wikilink(target_id: str, alias: str | None = None) -> str:
    """Obsidian wikilink. `[[ID]]` hoặc `[[ID|alias]]`."""
    if alias and alias.strip() and alias.strip() != target_id:
        safe = alias.replace("|", "\\|")
        return f"[[{target_id}|{safe}]]"
    return f"[[{target_id}]]"


def count_wikilinks(content: str) -> int:
    """Đếm số `[[` không bị escape."""
    return len(re.findall(r"(?<!\\)\[\[", content))


# ===== Edge metadata =====
def render_edges_block(edges: list[dict[str, str]]) -> list[str]:
    """3 dòng con cho mỗi quan hệ: type / quote / verification_status."""
    lines = []
    for e in edges:
        rt = e.get("relationship_type", "") or "_(không rõ)_"
        quote = e.get("evidence_quote", "")
        vs = e.get("verification_status", "") or "_(không rõ)_"
        lines.append(f"  - relationship_type: `{rt}`")
        lines.append(f"  - evidence_quote: {(quote and '> ' + quote) or '_(trống)_'}")
        lines.append(f"  - verification_status: `{vs}`")
    return lines


# ===== Page builders =====
def build_ruiro_page(entity: dict, edges_in: list[dict], edges_out: list[dict],
                     by_id: dict) -> str:
    eid = entity["id"]
    name = entity.get("name") or eid

    fm = yaml_frontmatter({
        "id": eid,
        "type": "RuiRo",
        "name": name,
        "category": entity.get("category", ""),
        "inherent_level": entity.get("inherent_level", ""),
        "residual_level": entity.get("residual_level", ""),
        "owner_unit_id": entity.get("owner_unit_id", ""),
        "verification_status": entity.get("verification_status", ""),
        "data_origin": entity.get("data_origin", ""),
    })

    body = [fm, "", f"# {name}", ""]
    body.append(f"**Mã:** `{eid}`  ")
    body.append(f"**Phân loại:** {entity.get('category') or '_(không rõ)_'}  ")
    body.append(
        f"**Mức rủi ro cố hữu (inherent):** "
        f"{entity.get('inherent_level') or '_(không rõ)_'}  "
    )
    body.append(
        f"**Mức rủi ro còn lại (residual):** "
        f"{entity.get('residual_level') or '_(không rõ)_'}  "
    )
    body.append(
        f"**Đơn vị sở hữu (chỉ mã):** "
        f"`{entity.get('owner_unit_id') or '_(không rõ)_'}`"
    )
    body.append("")

    desc = entity.get("description", "")
    if desc:
        body += ["## Mô tả", "", desc, ""]

    cause = entity.get("cause", "")
    event_d = entity.get("event", "")
    impact = entity.get("impact", "")
    body += [
        "## Nguyên nhân → Sự kiện → Hậu quả",
        "",
        f"- **Nguyên nhân:** {cause or '_(chưa rõ)_'}",
        f"- **Sự kiện:** {event_d or '_(chưa rõ)_'}",
        f"- **Hậu quả:** {impact or '_(chưa rõ)_'}",
        "",
    ]

    # Kiểm soát liên quan — incoming MITIGATES
    body += ["## Kiểm soát liên quan", ""]
    mit_in = [e for e in edges_in if e.get("relationship_type") == "MITIGATES"]
    if not mit_in:
        body.append(
            "> _(Chưa có kiểm soát nào được khai báo trong `relations.csv` cho rủi ro này — "
            "quan sát từ dữ liệu, không tự suy đoán.)_"
        )
    else:
        for e in mit_in:
            ctrl = by_id.get(e["source_id"])
            if not ctrl:
                continue
            body.append(f"- {wikilink(ctrl['id'], ctrl.get('name'))}")
            body += render_edges_block([e])
        body.append("")

    # Sự kiện liên quan — outgoing OBSERVED_AS
    body += ["## Sự kiện liên quan", ""]
    obs_out = [e for e in edges_out if e.get("relationship_type") == "OBSERVED_AS"]
    if not obs_out:
        body.append(
            "> _(Chưa có sự kiện nào được quan sát trong `relations.csv`.)_"
        )
    else:
        for e in obs_out:
            sk = by_id.get(e["target_id"])
            if not sk:
                continue
            label = (sk.get("description") or sk["id"])[:60]
            body.append(f"- {wikilink(sk['id'], label)}")
            body += render_edges_block([e])
        body.append("")

    body += [
        "---",
        "",
        f"_verification_status: `{entity.get('verification_status', '')}` · "
        f"data_origin: `{entity.get('data_origin', '')}`_",
    ]
    return "\n".join(body) + "\n"


def build_kiemsoat_page(entity: dict, edges_out: list[dict], by_id: dict) -> str:
    eid = entity["id"]
    name = entity.get("name") or eid

    fm = yaml_frontmatter({
        "id": eid,
        "type": "KiemSoat",
        "name": name,
        "control_type": entity.get("control_type", ""),
        "frequency": entity.get("frequency", ""),
        "effectiveness": entity.get("effectiveness", ""),
        "owner_role_id": entity.get("owner_role_id", ""),
        "verification_status": entity.get("verification_status", ""),
        "data_origin": entity.get("data_origin", ""),
    })

    body = [fm, "", f"# {name}", ""]
    body.append(f"**Mã:** `{eid}`  ")
    body.append(f"**Loại kiểm soát:** {entity.get('control_type') or '_(không rõ)_'}  ")
    body.append(f"**Tần suất:** {entity.get('frequency') or '_(không rõ)_'}  ")
    body.append(f"**Hiệu quả:** {entity.get('effectiveness') or '_(không rõ)_'}  ")
    body.append(
        f"**Vai trò phụ trách (chỉ mã):** "
        f"`{entity.get('owner_role_id') or '_(không rõ)_'}`"
    )
    body.append("")

    desc = entity.get("description", "")
    if desc:
        body += ["## Mô tả", "", desc, ""]

    # Rủi ro liên quan — outgoing MITIGATES
    body += ["## Rủi ro mà kiểm soát này MITIGATES", ""]
    mit_out = [e for e in edges_out if e.get("relationship_type") == "MITIGATES"]
    if not mit_out:
        body.append("> _(Chưa có rủi ro nào trong `relations.csv`.)_")
    else:
        for e in mit_out:
            rr = by_id.get(e["target_id"])
            if not rr:
                continue
            body.append(f"- {wikilink(rr['id'], rr.get('name'))}")
            body += render_edges_block([e])
        body.append("")

    body += [
        "---",
        "",
        f"_verification_status: `{entity.get('verification_status', '')}` · "
        f"data_origin: `{entity.get('data_origin', '')}`_",
    ]
    return "\n".join(body) + "\n"


def build_sukien_page(entity: dict, parent_rr: dict | None,
                      parent_edge: dict | None) -> str:
    eid = entity["id"]

    fm = yaml_frontmatter({
        "id": eid,
        "type": "SuKienRuiRo",
        "risk_id": entity.get("risk_id", ""),
        "occurred_at": entity.get("occurred_at", ""),
        "discovered_at": entity.get("discovered_at", ""),
        "severity": entity.get("severity", ""),
        "loss_amount_vnd": entity.get("loss_amount_vnd", ""),
        "verification_status": entity.get("verification_status", ""),
        "data_origin": entity.get("data_origin", ""),
    })

    body = [fm, "", f"# Sự kiện {eid}", ""]
    body.append(f"**Mã sự kiện:** `{eid}`  ")
    body.append(f"**Mã rủi ro (từ `risk_id`):** `{entity.get('risk_id') or '_(không rõ)_'}`  ")
    body.append(f"**Ngày xảy ra:** {entity.get('occurred_at') or '_(không rõ)_'}  ")
    body.append(f"**Ngày phát hiện:** {entity.get('discovered_at') or '_(không rõ)_'}  ")
    body.append(f"**Mức độ:** {entity.get('severity') or '_(không rõ)_'}  ")
    raw_loss = entity.get("loss_amount_vnd", "")
    try:
        loss_str = f"{int(raw_loss):,}".replace(",", ".") if raw_loss else "_(không rõ)_"
        loss_str += " VND" if raw_loss else ""
    except ValueError:
        loss_str = raw_loss or "_(không rõ)_"
    body.append(f"**Tổn thất:** {loss_str}")
    body.append("")

    desc = entity.get("description", "")
    if desc:
        body += ["## Mô tả", "", desc, ""]

    body += ["## Rủi ro tương ứng", ""]
    if parent_rr and parent_edge:
        body.append(f"- {wikilink(parent_rr['id'], parent_rr.get('name'))}")
        body += render_edges_block([parent_edge])
    else:
        body.append(
            "> _(Không tìm thấy quan hệ `OBSERVED_AS` tương ứng trong `relations.csv`.)_"
        )
    body.append("")

    body += [
        "---",
        "",
        f"_verification_status: `{entity.get('verification_status', '')}` · "
        f"data_origin: `{entity.get('data_origin', '')}`_",
    ]
    return "\n".join(body) + "\n"


def build_index_page(label_vi: str, entities: list[dict]) -> str:
    body = [
        f"# Danh sách {label_vi}",
        "",
        f"_Tổng: **{len(entities)}** mục._",
        "",
        "| # | Mã | Liên kết | Trạng thái |",
        "| --- | --- | --- | --- |",
    ]
    for i, e in enumerate(entities, 1):
        name = e.get("name") or e["id"]
        link = wikilink(e["id"], name)
        vs = e.get("verification_status") or "_(không rõ)_"
        body.append(f"| {i} | `{e['id']}` | {link} | `{vs}` |")
    body.append("")
    return "\n".join(body) + "\n"


def find_example_path(by_type: dict, edges_by_src: dict,
                      edges_by_tgt: dict) -> tuple[str | None, str | None, str | None]:
    """Tìm một RuiRo vừa có MITIGATES (in), vừa có OBSERVED_AS (out)."""
    for rr in by_type.get("RuiRo", []):
        eid = rr["id"]
        ks = next(
            (e["source_id"] for e in edges_by_tgt.get(eid, [])
             if e.get("relationship_type") == "MITIGATES"),
            None,
        )
        sk = next(
            (e["target_id"] for e in edges_by_src.get(eid, [])
             if e.get("relationship_type") == "OBSERVED_AS"),
            None,
        )
        if ks and sk:
            return ks, eid, sk
    return None, None, None


def build_home(by_type: dict, edges: list, edge_counts: Counter) -> str:
    n_total = sum(len(v) for v in by_type.values())

    fm = yaml_frontmatter({
        "title": "Wiki Risk Graph — Trang chủ",
        "type": "Home",
    })

    body = [fm, "", "# Wiki Risk Graph", ""]
    body += [
        "Wiki tri thức cho **Wiki Risk Graph MVP** — được sinh tự động từ "
        "`outputs/entities.csv` + `outputs/relations.csv`.",
        "",
        "## Thống kê",
        "",
        f"- **Tổng node:** {n_total}",
    ]
    for t in ("RuiRo", "KiemSoat", "SuKienRuiRo"):
        body.append(f"  - `{t}`: {len(by_type.get(t, []))}")
    body.append(f"- **Tổng edge:** {len(edges)}")
    for rt, c in sorted(edge_counts.items(), key=lambda x: -x[1]):
        body.append(f"  - `{rt}`: {c}")
    body.append("")

    body += [
        "## Danh sách",
        "",
        f"- [[risks/README|Danh sách rủi ro ({len(by_type.get('RuiRo', []))})]]",
        f"- [[controls/README|Danh sách kiểm soát ({len(by_type.get('KiemSoat', []))})]]",
        f"- [[events/README|Danh sách sự kiện rủi ro ({len(by_type.get('SuKienRuiRo', []))})]]",
        "",
    ]

    # Đường đi minh họa (tự chọn từ dữ liệu)
    body += [
        "## Đường đi minh họa",
        "",
        "Một đường đi đầy đủ **KiemSoat → RuiRo → SuKienRuiRo** có trong dữ liệu:",
        "",
        "_(placeholder — sẽ được main() thay bằng đường đi thực tế)_",
        "",
    ]
    body += [
        "## Cam kết dữ liệu",
        "",
        "- Wiki này được sinh tự động — không tự bịa quan hệ.",
        "- Không suy đoán tên đơn vị/vai trò từ mã (`owner_unit_id`, `owner_role_id`).",
        "- Mọi `verification_status` giữ nguyên giá trị gốc.",
        "",
    ]
    return "\n".join(body) + "\n"


# ===== Main =====
def main() -> int:
    print(HEADER)
    print(" Wiki Risk Graph — Build wiki/ from outputs/ ")
    print(HEADER)

    entities = read_csv(ENTITIES_PATH)
    relations = read_csv(RELATIONS_PATH)
    by_id, by_type = index_entities(entities)
    edges, edges_by_src, edges_by_tgt = index_relations(relations)
    edge_counts = Counter(e["relationship_type"] for e in edges)

    # Tạo thư mục & dọn file .md cũ (chỉ top-level trong mỗi thư mục)
    for d in (SUB_RISKS, SUB_CONTROLS, SUB_EVENTS):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.md"):
            old.unlink()
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    for old in WIKI_DIR.glob("*.md"):
        old.unlink()

    total_pages = 0
    total_wikilinks = 0

    # --- RuiRo pages ---
    for rr in by_type.get("RuiRo", []):
        eid = rr["id"]
        md = build_ruiro_page(
            rr,
            edges_by_tgt.get(eid, []),
            edges_by_src.get(eid, []),
            by_id,
        )
        (SUB_RISKS / f"{eid}.md").write_text(md, encoding="utf-8")
        total_pages += 1
        total_wikilinks += count_wikilinks(md)

    # --- KiemSoat pages ---
    for ks in by_type.get("KiemSoat", []):
        eid = ks["id"]
        md = build_kiemsoat_page(ks, edges_by_src.get(eid, []), by_id)
        (SUB_CONTROLS / f"{eid}.md").write_text(md, encoding="utf-8")
        total_pages += 1
        total_wikilinks += count_wikilinks(md)

    # --- SuKienRuiRo pages ---
    for sk in by_type.get("SuKienRuiRo", []):
        eid = sk["id"]
        parent = None
        parent_edge = None
        for e in edges_by_tgt.get(eid, []):
            if e.get("relationship_type") == "OBSERVED_AS":
                cand = by_id.get(e["source_id"])
                if cand:
                    parent = cand
                    parent_edge = e
                    break
        md = build_sukien_page(sk, parent, parent_edge)
        (SUB_EVENTS / f"{eid}.md").write_text(md, encoding="utf-8")
        total_pages += 1
        total_wikilinks += count_wikilinks(md)

    # --- Index pages ---
    for sub, vi_label, etype in [
        (SUB_RISKS, "rủi ro (RuiRo)", "RuiRo"),
        (SUB_CONTROLS, "kiểm soát (KiemSoat)", "KiemSoat"),
        (SUB_EVENTS, "sự kiện rủi ro (SuKienRuiRo)", "SuKienRuiRo"),
    ]:
        idx_md = build_index_page(vi_label, by_type.get(etype, []))
        (sub / "README.md").write_text(idx_md, encoding="utf-8")
        total_pages += 1
        total_wikilinks += count_wikilinks(idx_md)

    # --- Home.md ---
    home_md = build_home(by_type, edges, edge_counts)
    # Lấy example path và chèn vào Home.md
    ks_id, rr_id, sk_id = find_example_path(by_type, edges_by_src, edges_by_tgt)
    if ks_id and rr_id and sk_id:
        rr_name = by_id[rr_id].get("name", "")
        example_lines = [
            "[[" + ks_id + "]] → [[" + rr_id + "]] → [[" + sk_id + "]]",
            "",
            f"- **{ks_id}** — kiểm soát liên quan (mở trang để xem tên).",
            f"- **{rr_id}** — {rr_name}",
            f"- **{sk_id}** — sự kiện quan sát được.",
        ]
        home_md = home_md.replace(
            "_(placeholder — sẽ được main() thay bằng đường đi thực tế)_",
            "\n".join(example_lines),
        )
    (WIKI_DIR / "Home.md").write_text(home_md, encoding="utf-8")
    total_pages += 1
    total_wikilinks += count_wikilinks(home_md)

    # ===== Báo cáo =====
    print(f"\nWiki đã ghi vào: {WIKI_DIR}")
    print()
    print(
        f"  risks/      : {len(by_type.get('RuiRo', [])):>3} trang + README.md"
    )
    print(
        f"  controls/   : {len(by_type.get('KiemSoat', [])):>3} trang + README.md"
    )
    print(
        f"  events/     : {len(by_type.get('SuKienRuiRo', [])):>3} trang + README.md"
    )
    print(
        f"  Home.md     :   1 trang"
    )
    print()
    print(f"Tổng số trang: {total_pages}")
    print(f"Tổng số wikilink phát sinh: {total_wikilinks}")

    print()
    print("Đường đi minh họa (lấy thực tế từ dữ liệu):")
    if ks_id and rr_id and sk_id:
        print(f"  KiemSoat : {ks_id}  →  wiki/controls/{ks_id}.md")
        print(f"  RuiRo    : {rr_id}  →  wiki/risks/{rr_id}.md")
        print(f"  SuKien   : {sk_id}  →  wiki/events/{sk_id}.md")
        rr_name = by_id[rr_id].get("name", "")
        print(f"  Nhãn đường đi: [[{ks_id}]] → [[{rr_id}]] → [[{sk_id}]]")
        print(f"  Tên rủi ro: {rr_name}")
    print()
    print(HEADER)
    print("Done.")
    print(HEADER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
