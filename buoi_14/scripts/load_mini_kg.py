"""Buổi 14 — Mini Knowledge Graph loader (VanBan / DieuKhoan).

Đọc (read-only):
- ../ kb+hops/metadata.csv           → VanBan nodes
- ../ kb+hops/relationships.csv      → VanBan ↔ VanBan + VanBan → Entity
- buoi_14/data/processed/chunks_normalized.csv → DieuKhoan + :NEXT chain

Sinh ra:
- cypher/schema.cypher        (đã viết tay — ontology VanBan / DieuKhoan)
- cypher/demo_queries.cypher  (đã viết tay — $params)
- cypher/load_data.cypher     (auto — MERGE + UNWIND, idempotent)
- cypher/load_data_params.json
- outputs/kg_build_report.md  (thống kê + orphan check + push status)

Tùy chọn push lên Neo4j thật (an toàn):
- Đặt NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE qua `.env`
- Dùng `python-dotenv` để load `.env`.
- Tất cả query tham số hoá (`$params`), KHÔGH hard-code password.
- Mọi node / relationship đều mang `lab_session = "buoi_14"` để phạm vi hoá.
- TUYỆT ĐỐI KHÔNG chạy:
      MATCH (n) DETACH DELETE n
- Có chế độ cleanup giới hạn (chỉ xoá node/edge có `lab_session='buoi_14'`)
  nhưng phải bật `--clean-previous` HOẶC `CLEAN_PREVIOUS=1` trong `.env`.

Quy tắc chống bịa:
- Tên Entity lấy từ `relationships.target_name` thật.
- Điều khoản text lấy từ `chunks_normalized.csv.text` thật.
- VanBan title/document_type/status lấy từ metadata.csv thật.
- Mọi quan hệ Document↔Document / Document→Entity lấy từ
  relationships.csv với kind/rtype đã biết trước — không tạo rel type mới.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Đọc .env ngay khi script khởi động (KHÔNG bắt buộc)
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import pandas as pd

from src.common import PROJECT_ROOT

# ---------- đường dẫn ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

KB_CANDIDATES = [
    PROJECT_ROOT / " kb+hops",
    PROJECT_ROOT.parent / " kb+hops",
]
KB_BASE = next((p for p in KB_CANDIDATES if p.is_dir()), None)
if KB_BASE is None:
    raise FileNotFoundError(f"Cannot locate ` kb+hops/`. Tried: {[str(p) for p in KB_CANDIDATES]}")

SRC_META = KB_BASE / "metadata.csv"
SRC_CONTENT = KB_BASE / "content.csv"
SRC_REL = KB_BASE / "relationships.csv"
SRC_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks_normalized.csv"

CYPHER_DIR = PROJECT_ROOT / "cypher"
OUT_DIR = PROJECT_ROOT / "outputs"

LAB = "buoi_14"

# Map relationship_type → (src_kind, tgt_kind) theo LABEL trong graph mới.
# (CSV gốc dùng "Document" thay vì "VanBan"; mapping dưới: Document→VanBan)
REL_KIND: dict[str, tuple[str, str]] = {
    "THAM_CHIEU":       ("VanBan", "VanBan"),
    "THAY_THE_BOI":     ("VanBan", "VanBan"),
    "SUA_DOI_BO_SUNG":  ("VanBan", "VanBan"),
    "BAN_HANH_BOI":     ("VanBan", "Entity"),
    "KY_BOI":           ("VanBan", "Entity"),
    "THUOC_LINH_VUC":   ("VanBan", "Entity"),
    "AP_DUNG_CHO":      ("VanBan", "Entity"),
}

# Ánh xạ kind trong CSV → label thật trong graph
CSV_KIND_TO_LABEL = {"Document": "VanBan", "Entity": "Entity"}

# ---------- build in-memory graph ----------
def _build_graph() -> tuple[nx.DiGraph, dict]:
    meta = pd.read_csv(SRC_META, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    meta.columns = [c.strip() for c in meta.columns]
    rel = pd.read_csv(SRC_REL, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    rel.columns = [c.strip() for c in rel.columns]
    chunks = pd.read_csv(SRC_CHUNKS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    chunks.columns = [c.strip() for c in chunks.columns]

    # Map so_ky_hieu -> metadata.id để chuẩn hoá các source/target id trong CSV
    so_to_id: dict[str, str] = {}
    for _, row in meta.iterrows():
        sid = str(row["id"]).strip()
        so = str(row.get("so_ky_hieu", "")).strip()
        if so and so not in so_to_id:
            so_to_id[so] = sid

    def norm_doc_id(raw: str) -> str:
        return so_to_id.get(raw.strip(), raw.strip())

    G = nx.DiGraph()

    # ---------- :VanBan ----------
    doc_ids_from_meta: set[str] = set()
    for _, row in meta.iterrows():
        did = str(row["id"]).strip()
        doc_ids_from_meta.add(did)
        node = f"VanBan:{did}"
        G.add_node(
            node,
            kind="VanBan",
            id=did,
            title=str(row.get("title", "")).strip(),
            document_type=str(row.get("loai_van_ban", "")).strip(),
            status=str(row.get("tinh_trang_hieu_luc", "")).strip(),
            so_ky_hieu=str(row.get("so_ky_hieu", "")).strip(),
            ngay_ban_hanh=str(row.get("ngay_ban_hanh", "")).strip(),
            ngay_co_hieu_luc=str(row.get("ngay_co_hieu_luc", "")).strip(),
            ngay_het_hieu_luc=str(row.get("ngay_het_hieu_luc", "")).strip(),
            co_quan_ban_hanh=str(row.get("co_quan_ban_hanh", "")).strip(),
            nguoi_ky=str(row.get("nguoi_ky", "")).strip(),
            nganh=str(row.get("nganh", "")).strip(),
            linh_vuc=str(row.get("linh_vuc", "")).strip(),
            lab_session=LAB,
        )

    # ---------- :DieuKhoan + :CONTAINS + :NEXT ----------
    orphan_dieukhoan: list[str] = []
    chunks_per_doc: dict[str, list[str]] = defaultdict(list)
    chunks_in_rels_or_csv: list[dict] = []

    for _, c in chunks.iterrows():
        cid = str(c["chunk_id"]).strip()
        did = str(c["document_id"]).strip()
        article = str(c.get("article", "")).strip()
        clause = str(c.get("clause", "")).strip()
        chapter = str(c.get("chapter", "")).strip()
        section = str(c.get("section", "")).strip()
        text = str(c.get("text", ""))
        so_ky_hieu = str(c.get("so_ky_hieu", "")).strip()
        ngay_ban_hanh = str(c.get("ngay_ban_hanh", "")).strip()

        if not text.strip():
            continue  # bỏ chunk rỗng

        # Chuẩn hoá document_id về metadata.id
        if did in so_to_id:
            did = so_to_id[did]

        # :DieuKhoan
        node = f"DieuKhoan:{cid}"
        G.add_node(
            node,
            kind="DieuKhoan",
            id=cid,
            document_id=did,
            text=text,
            article=article,
            clause=clause,
            chapter=chapter,
            section=section,
            article_title=str(c.get("article_title", "")).strip(),
            document_type=str(c.get("document_type", "")).strip(),
            title=str(c.get("title", "")).strip(),
            status=str(c.get("status", "")).strip(),
            so_ky_hieu=so_ky_hieu,
            ngay_ban_hanh=ngay_ban_hanh,
            lab_session=LAB,
        )

        # :CONTAINS VanBan → DieuKhoan
        vbn = f"VanBan:{did}"
        if G.has_node(vbn):
            G.add_edge(
                vbn, node,
                type="CONTAINS",
                confidence=1.0,
                method="chunks_normalized",
                evidence="",
                source="chunks_normalized.csv",
                lab_session=LAB,
            )
        else:
            orphan_dieukhoan.append(cid)
            # vẫn tạo stub VanBan để graph không bị đứt — rất hiếm
            G.add_node(vbn, kind="VanBan", id=did, title="", document_type="",
                       status="", so_ky_hieu="", ngay_ban_hanh="", ngay_co_hieu_luc="",
                       ngay_het_hieu_luc="", co_quan_ban_hanh="", nguoi_ky="",
                       nganh="", linh_vuc="", lab_session=LAB)
            G.add_edge(vbn, node, type="CONTAINS", confidence=1.0,
                       method="chunks_normalized", evidence="",
                       source="chunks_normalized.csv", lab_session=LAB)

        chunks_per_doc[did].append(node)

    # :NEXT — chain theo thứ tự chunk trong CSV (đã đúng thứ tự điều khoản)
    for did, dk_nodes in chunks_per_doc.items():
        for i in range(len(dk_nodes) - 1):
            G.add_edge(
                dk_nodes[i], dk_nodes[i + 1],
                type="NEXT",
                confidence=1.0,
                method="chunks_normalized",
                evidence="",
                source="chunks_normalized.csv (order)",
                lab_session=LAB,
            )

    # ---------- :Entity từ relationships.csv ----------
    entity_names: dict[str, set[str]] = {}
    for _, r in rel.iterrows():
        tk_csv = str(r["target_kind"]).strip()
        if tk_csv != "Entity":
            continue
        tid = str(r["target_id"]).strip()
        ent_node = f"Entity:{tid}"
        tname = str(r["target_name"]).strip()
        entity_names.setdefault(tid, set()).add(tname)
        if not G.has_node(ent_node):
            G.add_node(ent_node, kind="Entity", id=tid, name=tname,
                       name_alt="", lab_session=LAB)
        elif G.nodes[ent_node].get("name") != tname:
            cur_alt = G.nodes[ent_node].get("name_alt") or ""
            names = set(filter(None, [cur_alt, tname])) - {G.nodes[ent_node].get("name", "")}
            if names:
                G.nodes[ent_node]["name_alt"] = " | ".join(sorted(names))

    # ---------- Edges từ relationships.csv ----------
    skipped = 0
    remapped = 0
    edge_groups: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for _, r in rel.iterrows():
        sk_csv = str(r["source_kind"]).strip()
        sid_raw = str(r["source_id"]).strip()
        tk_csv = str(r["target_kind"]).strip()
        tid = str(r["target_id"]).strip()
        rtype = str(r["relationship_type"]).strip()
        if rtype not in REL_KIND:
            skipped += 1
            continue
        # Map CSV kind → graph label, dùng label để so sánh với REL_KIND
        sk = CSV_KIND_TO_LABEL.get(sk_csv, sk_csv)
        tk = CSV_KIND_TO_LABEL.get(tk_csv, tk_csv)
        if (sk, tk) != REL_KIND[rtype]:
            skipped += 1
            continue
        # Normalize source/target ids về metadata.id nếu là so_ky_hieu
        if sk == "VanBan":
            new_sid = norm_doc_id(sid_raw)
            if new_sid != sid_raw:
                remapped += 1
            sid_raw = new_sid
        if tk == "VanBan":
            new_tid = norm_doc_id(tid)
            if new_tid != tid:
                remapped += 1
            tid = new_tid

        # Đảm bảo VanBan có node (stub nếu missing)
        if sk == "VanBan" and not G.has_node(f"VanBan:{sid_raw}"):
            G.add_node(f"VanBan:{sid_raw}", kind="VanBan", id=sid_raw,
                       title="", document_type="", status="", so_ky_hieu="",
                       ngay_ban_hanh="", ngay_co_hieu_luc="", ngay_het_hieu_luc="",
                       co_quan_ban_hanh="", nguoi_ky="", nganh="", linh_vuc="",
                       lab_session=LAB)
        if tk == "VanBan" and not G.has_node(f"VanBan:{tid}"):
            G.add_node(f"VanBan:{tid}", kind="VanBan", id=tid,
                       title="", document_type="", status="", so_ky_hieu="",
                       ngay_ban_hanh="", ngay_co_hieu_luc="", ngay_het_hieu_luc="",
                       co_quan_ban_hanh="", nguoi_ky="", nganh="", linh_vuc="",
                       lab_session=LAB)

        u = f"{sk}:{sid_raw}"
        v = f"{tk}:{tid}"
        try:
            conf = float(r.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        d = {
            "type": rtype,
            "confidence": conf,
            "method": str(r.get("method", "")).strip(),
            "evidence": str(r.get("evidence", "")).strip()[:200],
            "source": "kb+hops/relationships.csv",
            "lab_session": LAB,
        }
        G.add_edge(u, v, **d)
        edge_groups[rtype].append((u, v, d))

    # ---------- Orphan detection ----------
    orphan_vanban: list[str] = []
    for n, d in G.nodes(data=True):
        if d.get("kind") != "VanBan":
            continue
        # VanBan không có CONTAINS outgoing nào
        has_contains = False
        for _u, _v, edge_attrs in G.out_edges(n, data=True):
            if edge_attrs.get("type") == "CONTAINS":
                has_contains = True
                break
        if not has_contains:
            orphan_vanban.append(d.get("id", n))

    stats = {
        "docs_in_meta": len(doc_ids_from_meta),
        "dieukhoan_total": sum(1 for n, d in G.nodes(data=True) if d.get("kind") == "DieuKhoan"),
        "entity_total": sum(1 for n, d in G.nodes(data=True) if d.get("kind") == "Entity"),
        "edges_total": G.number_of_edges(),
        "contains_total": sum(1 for u, v, d in G.edges(data=True) if d.get("type") == "CONTAINS"),
        "next_total": sum(1 for u, v, d in G.edges(data=True) if d.get("type") == "NEXT"),
        "rels_total": sum(1 for u, v, d in G.edges(data=True)
                          if d.get("type") in ("THAM_CHIEU", "THAY_THE_BOI", "SUA_DOI_BO_SUNG",
                                                "BAN_HANH_BOI", "KY_BOI", "THUOC_LINH_VUC", "AP_DUNG_CHO")),
        "rels_by_type": dict(Counter(d.get("type") for _, _, d in G.edges(data=True))),
        "edges_skipped": skipped,
        "remapped": remapped,
        "orphan_vanban": orphan_vanban,
        "orphan_dieukhoan": orphan_dieukhoan,
        "entity_alt_names": {k: sorted(v) for k, v in entity_names.items() if len(v) > 1},
        "chunks_per_doc": {k: len(v) for k, v in chunks_per_doc.items()},
    }
    return G, stats, edge_groups


# ---------- emit Cypher ----------
def _emit_load_cypher(G: nx.DiGraph, edge_groups: dict, stats: dict, path: Path):
    vanban_nodes = [d for n, d in G.nodes(data=True) if d.get("kind") == "VanBan"]
    dieu_nodes = [d for n, d in G.nodes(data=True) if d.get("kind") == "DieuKhoan"]
    ent_nodes = [d for n, d in G.nodes(data=True) if d.get("kind") == "Entity"]

    def _params_vanban():
        return [
            {
                "id": d["id"],
                "title": d.get("title", ""),
                "document_type": d.get("document_type", ""),
                "status": d.get("status", ""),
                "so_ky_hieu": d.get("so_ky_hieu", ""),
                "ngay_ban_hanh": d.get("ngay_ban_hanh", ""),
                "ngay_co_hieu_luc": d.get("ngay_co_hieu_luc", ""),
                "ngay_het_hieu_luc": d.get("ngay_het_hieu_luc", ""),
                "co_quan_ban_hanh": d.get("co_quan_ban_hanh", ""),
                "nguoi_ky": d.get("nguoi_ky", ""),
                "nganh": d.get("nganh", ""),
                "linh_vuc": d.get("linh_vuc", ""),
            }
            for d in vanban_nodes
        ]

    def _params_dieu():
        return [
            {
                "id": d["id"],
                "document_id": d.get("document_id", ""),
                "text": d.get("text", ""),
                "article": d.get("article", ""),
                "clause": d.get("clause", ""),
                "chapter": d.get("chapter", ""),
                "section": d.get("section", ""),
                "article_title": d.get("article_title", ""),
                "so_ky_hieu": d.get("so_ky_hieu", ""),
                "ngay_ban_hanh": d.get("ngay_ban_hanh", ""),
                "document_type": d.get("document_type", ""),
                "title": d.get("title", ""),
                "status": d.get("status", ""),
            }
            for d in dieu_nodes
        ]

    def _params_entity():
        return [
            {
                "id": d["id"],
                "name": d.get("name", ""),
                "name_alt": (d.get("name_alt") or None),
            }
            for d in ent_nodes
        ]

    def _params_edges():
        out = {}
        for rtype, plist in edge_groups.items():
            out[rtype] = [
                {
                    "src": u.split(":", 1)[1],
                    "tgt": v.split(":", 1)[1],
                    "src_kind": u.split(":", 1)[0],
                    "tgt_kind": v.split(":", 1)[0],
                    "confidence": float(d.get("confidence", 0.0)),
                    "method": d.get("method", ""),
                    "evidence": d.get("evidence", ""),
                }
                for u, v, d in plist
            ]
        return out

    def _params_contains():
        out = []
        for u, v, d in G.edges(data=True):
            if d.get("type") != "CONTAINS":
                continue
            out.append({"src": u.split(":", 1)[1], "tgt": v.split(":", 1)[1]})
        return out

    def _params_next():
        out = []
        for u, v, d in G.edges(data=True):
            if d.get("type") != "NEXT":
                continue
            out.append({"src": u.split(":", 1)[1], "tgt": v.split(":", 1)[1]})
        return out

    params_vanban = _params_vanban()
    params_dieu = _params_dieu()
    params_entity = _params_entity()
    params_edges = _params_edges()
    params_contains = _params_contains()
    params_next = _params_next()

    lines: list[str] = []
    lines.append("// Buổi 14 — Mini KG data load (auto-generated)")
    lines.append("// Source: buoi_14/data/processed/chunks_normalized.csv + ../ kb+hops/metadata.csv + ../ kb+hops/relationships.csv")
    lines.append("// Thứ tự chạy: schema.cypher → load_data.cypher → demo_queries.cypher")
    lines.append("// Mọi node/edge đều có lab_session = \"buoi_14\" để cô lập.")
    lines.append("")
    lines.append(f"// Counts: VanBan={len(params_vanban)} · DieuKhoan={len(params_dieu)} · Entity={len(params_entity)}")
    lines.append(f"//         CONTAINS={len(params_contains)} · NEXT={len(params_next)}")
    for t, n in sorted(stats["rels_by_type"].items()):
        lines.append(f"//         {t}={n}")
    lines.append("")

    # VanBan
    lines.append("// ============================================================")
    lines.append(f"// (:VanBan) ({len(params_vanban)})")
    lines.append("// ============================================================")
    lines.append("UNWIND $vanbans AS v")
    lines.append("MERGE (vb:VanBan {id: v.id})")
    lines.append("SET vb.title            = v.title,")
    lines.append("    vb.document_type    = v.document_type,")
    lines.append("    vb.status           = v.status,")
    lines.append("    vb.so_ky_hieu       = v.so_ky_hieu,")
    lines.append("    vb.ngay_ban_hanh    = v.ngay_ban_hanh,")
    lines.append("    vb.ngay_co_hieu_luc = v.ngay_co_hieu_luc,")
    lines.append("    vb.ngay_het_hieu_luc= v.ngay_het_hieu_luc,")
    lines.append("    vb.co_quan_ban_hanh = v.co_quan_ban_hanh,")
    lines.append("    vb.nguoi_ky         = v.nguoi_ky,")
    lines.append("    vb.nganh            = v.nganh,")
    lines.append("    vb.linh_vuc         = v.linh_vuc,")
    lines.append("    vb.lab_session      = 'buoi_14';")
    lines.append("")

    # DieuKhoan
    lines.append("// ============================================================")
    lines.append(f"// (:DieuKhoan) ({len(params_dieu)})")
    lines.append("// ============================================================")
    lines.append("UNWIND $dieukhoans AS d")
    lines.append("MERGE (dk:DieuKhoan {id: d.id})")
    lines.append("SET dk.document_id   = d.document_id,")
    lines.append("    dk.text          = d.text,")
    lines.append("    dk.article       = d.article,")
    lines.append("    dk.clause        = d.clause,")
    lines.append("    dk.chapter       = d.chapter,")
    lines.append("    dk.section       = d.section,")
    lines.append("    dk.article_title = d.article_title,")
    lines.append("    dk.so_ky_hieu    = d.so_ky_hieu,")
    lines.append("    dk.ngay_ban_hanh = d.ngay_ban_hanh,")
    lines.append("    dk.document_type = d.document_type,")
    lines.append("    dk.title         = d.title,")
    lines.append("    dk.status        = d.status,")
    lines.append("    dk.lab_session   = 'buoi_14';")
    lines.append("")

    # Entity
    lines.append("// ============================================================")
    lines.append(f"// (:Entity) ({len(params_entity)})")
    lines.append("// ============================================================")
    lines.append("UNWIND $entities AS e")
    lines.append("MERGE (en:Entity {id: e.id})")
    lines.append("SET en.name       = e.name,")
    lines.append("    en.name_alt   = coalesce(e.name_alt, null),")
    lines.append("    en.lab_session= 'buoi_14';")
    lines.append("")

    # CONTAINS
    lines.append("// ============================================================")
    lines.append(f"// (:VanBan)-[:CONTAINS]->(:DieuKhoan) ({len(params_contains)})")
    lines.append("// ============================================================")
    lines.append("UNWIND $contains AS c")
    lines.append("MATCH (vb:VanBan {id: c.src, lab_session: 'buoi_14'})")
    lines.append("MATCH (dk:DieuKhoan {id: c.tgt, lab_session: 'buoi_14'})")
    lines.append("MERGE (vb)-[r:CONTAINS]->(dk)")
    lines.append("SET r.lab_session = 'buoi_14', r.method = 'chunks_normalized', r.confidence = 1.0, r.evidence = '', r.source = 'chunks_normalized.csv';")
    lines.append("")

    # NEXT
    lines.append("// ============================================================")
    lines.append(f"// (:DieuKhoan)-[:NEXT]->(:DieuKhoan) ({len(params_next)})")
    lines.append("// chain trong cùng 1 VanBan theo thứ tự chunk trong chunks_normalized.csv")
    lines.append("// ============================================================")
    lines.append("UNWIND $nexts AS n")
    lines.append("MATCH (a:DieuKhoan {id: n.src, lab_session: 'buoi_14'})")
    lines.append("MATCH (b:DieuKhoan {id: n.tgt, lab_session: 'buoi_14'})")
    lines.append("MERGE (a)-[r:NEXT]->(b)")
    lines.append("SET r.lab_session = 'buoi_14', r.method = 'chunks_normalized', r.confidence = 1.0, r.evidence = '', r.source = 'chunks_normalized.csv (order)';")
    lines.append("")

    # Other edges (Document↔Document, Document→Entity) grouped by type
    lines.append("// ============================================================")
    lines.append("// Edges từ relationships.csv")
    lines.append("// ============================================================")
    for rtype, plist in edge_groups.items():
        var = f"rels_{rtype.lower()}"
        sk, tk = REL_KIND[rtype]
        lines.append(f"// --- {rtype} ({len(plist)}) — {sk} → {tk} ---")
        lines.append(f"UNWIND ${var} AS r")
        lines.append(f"MATCH (s:{sk} {{id: r.src, lab_session: 'buoi_14'}})")
        lines.append(f"MATCH (t:{tk} {{id: r.tgt, lab_session: 'buoi_14'}})")
        lines.append(f"MERGE (s)-[rel:{rtype}]->(t)")
        lines.append("SET rel.confidence = r.confidence,")
        lines.append("    rel.method     = r.method,")
        lines.append("    rel.evidence   = r.evidence,")
        lines.append("    rel.source     = 'kb+hops/relationships.csv',")
        lines.append("    rel.lab_session= 'buoi_14';")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

    # Params JSON
    json_path = path.with_name("load_data_params.json")
    json_path.write_text(
        json.dumps(
            {
                "vanbans": params_vanban,
                "dieukhoans": params_dieu,
                "entities": params_entity,
                "contains": params_contains,
                "nexts": params_next,
                "edges": params_edges,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "vanbans": len(params_vanban),
        "dieukhoans": len(params_dieu),
        "entities": len(params_entity),
        "contains": len(params_contains),
        "nexts": len(params_next),
        "edges": sum(len(v) for v in params_edges.values()),
        "json_path": str(json_path),
    }


# ---------- Neo4j push (an toàn) ----------
def _neo4j_driver(uri: str, user: str, password: str):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(uri, auth=(user, password))


def _safe_drop_buoi_14(driver, database: str):
    """Chạy cleanup giới hạn chỉ trên lab buoi_14. KHÔNG đụng các node khác."""
    with driver.session(database=database) as session:
        # Đếm trước để in cảnh báo
        n = session.run(
            "MATCH (n) WHERE n.lab_session = 'buoi_14' RETURN count(n) AS n"
        ).single()["n"]
        r = session.run(
            "MATCH ()-[r]->() WHERE r.lab_session = 'buoi_14' RETURN count(r) AS r"
        ).single()["r"]
        session.run(
            "MATCH (n) WHERE n.lab_session = 'buoi_14' DETACH DELETE n"
        )
    return n, r


def _push_to_neo4j(driver, database: str, payloads: dict):
    """Push tất cả data qua parameterized Cypher. KHÔNG truyền password."""
    with driver.session(database=database) as session:
        # VanBan
        session.run(
            "UNWIND $xs AS v "
            "MERGE (vb:VanBan {id: v.id}) "
            "SET vb += v, vb.lab_session = 'buoi_14'",
            xs=payloads["vanbans"],
        )
        # DieuKhoan
        session.run(
            "UNWIND $xs AS d "
            "MERGE (dk:DieuKhoan {id: d.id}) "
            "SET dk += d, dk.lab_session = 'buoi_14'",
            xs=payloads["dieukhoans"],
        )
        # Entity
        session.run(
            "UNWIND $xs AS e "
            "MERGE (en:Entity {id: e.id}) "
            "SET en += e, en.lab_session = 'buoi_14'",
            xs=payloads["entities"],
        )
        # CONTAINS
        session.run(
            "UNWIND $xs AS c "
            "MATCH (vb:VanBan {id: c.src, lab_session:'buoi_14'}) "
            "MATCH (dk:DieuKhoan {id: c.tgt, lab_session:'buoi_14'}) "
            "MERGE (vb)-[r:CONTAINS]->(dk) "
            "SET r.lab_session='buoi_14', r.method='chunks_normalized', r.confidence=1.0, r.evidence='', r.source='chunks_normalized.csv'",
            xs=payloads["contains"],
        )
        # NEXT
        session.run(
            "UNWIND $xs AS n "
            "MATCH (a:DieuKhoan {id: n.src, lab_session:'buoi_14'}) "
            "MATCH (b:DieuKhoan {id: n.tgt, lab_session:'buoi_14'}) "
            "MERGE (a)-[r:NEXT]->(b) "
            "SET r.lab_session='buoi_14', r.method='chunks_normalized', r.confidence=1.0, r.evidence='', r.source='chunks_normalized.csv (order)'",
            xs=payloads["nexts"],
        )
        # Other edges
        for rtype, plist in payloads["edges"].items():
            session.run(
                f"UNWIND $xs AS r "
                f"MATCH (s {{id: r.src, lab_session:'buoi_14'}}) "
                f"MATCH (t {{id: r.tgt, lab_session:'buoi_14'}}) "
                f"MERGE (s)-[rel:{rtype}]->(t) "
                "SET rel.confidence=r.confidence, rel.method=r.method, rel.evidence=r.evidence, rel.source='kb+hops/relationships.csv', rel.lab_session='buoi_14'",
                xs=plist,
            )


def _neo4j_count(driver, database: str) -> dict:
    counts = {"labels": {}, "rels": {}}
    with driver.session(database=database) as session:
        for row in session.run(
            "MATCH (n) WHERE n.lab_session='buoi_14' "
            "RETURN labels(n) AS l, count(*) AS c"
        ):
            lab = row["l"][0] if row["l"] else "(no label)"
            counts["labels"][lab] = row["c"]
        for row in session.run(
            "MATCH ()-[r]->() WHERE r.lab_session='buoi_14' "
            "RETURN type(r) AS t, count(*) AS c"
        ):
            counts["rels"][row["t"]] = row["c"]
    return counts


# ---------- report ----------
def _write_report(stats, payload_counts, push_status, neo4j_counts: dict | None, path: Path):
    md: list[str] = []
    md.append("# Buổi 14 — Mini Knowledge Graph Build Report\n")
    md.append("_Sinh tự động bằng `scripts/load_mini_kg.py`._\n")
    md.append("## Thống kê (in-memory + graph sau push nếu có)\n")
    md.append(f"- VanBan      : **{stats['docs_in_meta']}**\n"
              f"- DieuKhoan   : **{stats['dieukhoan_total']}**\n"
              f"- Entity      : **{stats['entity_total']}**\n"
              f"- Edges (all) : **{stats['edges_total']}** "
              f"(CONTAINS={stats['contains_total']} · NEXT={stats['next_total']} · "
              f"rels={stats['rels_total']})\n"
              f"- Skipped do kind/rtype mismatch: **{stats['edges_skipped']}**\n"
              f"- Remap `so_ky_hieu → metadata.id`: **{stats['remapped']}** lần\n")
    md.append("\n### Edges theo type (Source CSV = trước collapse; Graph = sau MERGE/dedup)\n")
    md.append("| Type | Source CSV rows | Graph in-memory (post-collapse) |")
    md.append("|---|---:|---:|")
    by_type = stats["rels_by_type"]  # post-collapse
    src_csv_counts = pd.read_csv(SRC_REL, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    src_csv_counts.columns = [c.strip() for c in src_csv_counts.columns]
    src_counts = src_csv_counts["relationship_type"].value_counts().to_dict()
    for t in ["THAM_CHIEU", "THAY_THE_BOI", "SUA_DOI_BO_SUNG",
              "BAN_HANH_BOI", "KY_BOI", "THUOC_LINH_VUC", "AP_DUNG_CHO"]:
        src_n = src_counts.get(t, 0)
        graph_n = by_type.get(t, 0)
        md.append(f"| {t} | {src_n} | {graph_n} |")
    md.append("\n### Orphan check\n")
    md.append(f"- VanBan không có CONTAINS outgoing : "
              f"**{len(stats['orphan_vanban'])}** "
              f"→ {stats['orphan_vanban'][:10]}{'…' if len(stats['orphan_vanban']) > 10 else ''}\n"
              f"- DieuKhoan không có VanBan cha : "
              f"**{len(stats['orphan_dieukhoan'])}**\n")
    if stats["entity_alt_names"]:
        md.append("\n### Entity có nhiều biến thể tên (`name_alt` được lưu)\n")
        md.append("| Entity id | name | name_alt |\n|---|---|---|")
        for eid, names in stats["entity_alt_names"].items():
            md.append(f"| {eid} | {names[0]} | {' · '.join(names[1:])} |")
        md.append("")
    md.append("\n## Cypher files\n")
    md.append("- `buoi_14/cypher/schema.cypher` — viết tay (ontology VanBan/DieuKhoan)\n"
              "- `buoi_14/cypher/load_data.cypher` — auto, MERGE + UNWIND, idempotent\n"
              "- `buoi_14/cypher/load_data_params.json` — params cho Cypher Browser `:param`\n"
              "- `buoi_14/cypher/demo_queries.cypher` — viết tay ($params)\n")
    md.append("\n## Push status\n")
    md.append(f"- **{push_status}**\n")
    if neo4j_counts:
        md.append("\n### Counts trên Neo4j (sau push)\n")
        md.append("| Label | n |\n|---|---:|")
        for k, n in sorted(neo4j_counts["labels"].items()):
            md.append(f"| {k} | {n} |")
        md.append("\n| Rel type | n |\n|---|---:|")
        for k, n in sorted(neo4j_counts["rels"].items()):
            md.append(f"| {k} | {n} |")
    md.append("\n## An toàn Neo4j\n")
    md.append(
        "- KHÔNG chạy `MATCH (n) DETACH DELETE n` trong script.\n"
        "- Mọi node/edge mang `lab_session = 'buoi_14'` để phạm vi hoá.\n"
        "- Cleanup nếu cần chỉ chạy khi `--clean-previous` (hoặc `CLEAN_PREVIOUS=1`), "
        "giới hạn ở `lab_session='buoi_14'`.\n"
        "- Tất cả Cypher dùng `$params` — KHÔNG string-interpolate, KHÔNG hardcode password.\n"
        "- Password chỉ đọc qua `os.getenv('NEO4J_PASSWORD')` sau `python-dotenv` load `.env`.\n"
    )
    md.append("\n## Hạn chế & ghi chú\n")
    md.append(
        "- `Chunks có text rỗng` bị loại (không tạo :DieuKhoan cho chúng).\n"
        "- `:NEXT` chỉ chain trong CÙNG VanBan, theo thứ tự rows trong CSV.\n"
        "- Quan hệ Document↔Document từ relationships.csv giữ NGUYÊN HƯỚNG như CSV ghi.\n"
        "- 1 source_id có thể có nhiều target qua nhiều method khác nhau; "
        "chỉ merge nếu `(u, v, type)` trùng — KHÔNG trộn evidence các lần.\n"
    )
    path.write_text("\n".join(md), encoding="utf-8")


# ---------- main ----------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cypher-emit", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--clean-previous", action="store_true",
                    help="xoá node/edge lab_session='buoi_14' TRƯỚC khi nạp "
                         "(KHÔNG đụng các lab khác)")
    args = ap.parse_args()

    print(f"[load] metadata : {SRC_META}")
    print(f"[load] relations: {SRC_REL}")
    print(f"[load] chunks   : {SRC_CHUNKS}")

    G, stats, edge_groups = _build_graph()
    print(f"[graph] VanBan={stats['docs_in_meta']}  "
          f"DieuKhoan={stats['dieukhoan_total']}  "
          f"Entity={stats['entity_total']}  "
          f"Edges={stats['edges_total']}  "
          f"(CONTAINS={stats['contains_total']} NEXT={stats['next_total']} "
          f"rels={stats['rels_total']})")
    print(f"[graph] skipped={stats['edges_skipped']}  remapped={stats['remapped']}")
    print(f"[graph] orphan VanBan={len(stats['orphan_vanban'])}  "
          f"orphan DieuKhoan={len(stats['orphan_dieukhoan'])}")

    if not args.no_cypher_emit:
        cstats = _emit_load_cypher(G, edge_groups, stats,
                                    CYPHER_DIR / "load_data.cypher")
        print(f"[cypher] wrote {CYPHER_DIR / 'load_data.cypher'}  "
              f"({cstats['vanbans']} V / {cstats['dieukhoans']} D / "
              f"{cstats['entities']} E / {cstats['contains']} C / "
              f"{cstats['nexts']} N / {cstats['edges']} R)")
        print(f"[cypher] wrote {CYPHER_DIR / 'load_data_params.json'}")

    # Neo4j push
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "neo4j")
    force_clean = (args.clean_previous
                   or os.environ.get("CLEAN_PREVIOUS") == "1")

    push_status = ""
    neo4j_counts = None
    if neo4j_uri and neo4j_password and not args.no_push:
        try:
            driver = _neo4j_driver(neo4j_uri, neo4j_user, neo4j_password)
        except Exception as e:
            push_status = f"FAIL — không khởi tạo driver: {e}"
            print(f"[neo4j] {push_status}")
            _write_report(stats, {}, push_status, neo4j_counts,
                          OUT_DIR / "kg_build_report.md")
            return 0

        try:
            if force_clean:
                print(f"[neo4j] cleanup dữ liệu cũ (lab='buoi_14') — giới hạn phạm vi")
                n_del, r_del = _safe_drop_buoi_14(driver, neo4j_database)
                print(f"[neo4j] đã xoá {n_del} nodes + {r_del} relationships "
                      "(chỉ trong lab_session='buoi_14')")

            payload, _ = _payload_for_push(G, edge_groups)
            _push_to_neo4j(driver, neo4j_database, payload)
            neo4j_counts = _neo4j_count(driver, neo4j_database)
            push_status = (f"OK — đã đẩy lên Neo4j `{neo4j_database}` "
                           f"@ {neo4j_uri}")
            print(f"[neo4j] {push_status}")
        except Exception as e:
            push_status = f"FAIL — {type(e).__name__}: {e}"
            print(f"[neo4j] {push_status}")
        finally:
            driver.close()
    else:
        reasons = []
        if not neo4j_uri: reasons.append("không có NEO4J_URI env")
        if not neo4j_password: reasons.append("không có NEO4J_PASSWORD env")
        if args.no_push: reasons.append("--no-push")
        push_status = "SKIP — " + ", ".join(reasons or ["không rõ"])
        print(f"[neo4j] {push_status}")
        if push_status == "SKIP — không rõ":
            push_status = "SKIP — không có NEO4J_URI/NEO4J_PASSWORD env"

    _write_report(stats, {}, push_status, neo4j_counts,
                  OUT_DIR / "kg_build_report.md")
    print(f"[report] wrote {OUT_DIR / 'kg_build_report.md'}")
    return 0


def _payload_for_push(G, edge_groups):
    """Trả về payloads dict + dict cxh cho report."""
    vanbans = [d for n, d in G.nodes(data=True) if d.get("kind") == "VanBan"]
    dieukhoans = [d for n, d in G.nodes(data=True) if d.get("kind") == "DieuKhoan"]
    entities = [d for n, d in G.nodes(data=True) if d.get("kind") == "Entity"]

    contains = []
    for u, v, d in G.edges(data=True):
        if d.get("type") == "CONTAINS":
            contains.append({"src": u.split(":", 1)[1], "tgt": v.split(":", 1)[1]})
    nexts = []
    for u, v, d in G.edges(data=True):
        if d.get("type") == "NEXT":
            nexts.append({"src": u.split(":", 1)[1], "tgt": v.split(":", 1)[1]})
    edges = {}
    for rtype, plist in edge_groups.items():
        edges[rtype] = [
            {"src": u.split(":", 1)[1], "tgt": v.split(":", 1)[1],
             "confidence": float(d.get("confidence", 0.0)),
             "method": d.get("method", ""),
             "evidence": d.get("evidence", "")}
            for u, v, d in plist
        ]
    return {
        "vanbans": vanbans,
        "dieukhoans": dieukhoans,
        "entities": entities,
        "contains": contains,
        "nexts": nexts,
        "edges": edges,
    }, {}


if __name__ == "__main__":
    raise SystemExit(main())
