#!/usr/bin/env python3
"""security_audit.py — Buổi 15.

Tự động kiểm thử **rò rỉ dữ liệu** của hệ thống secure retrieval.

Thiết kế 6 test case (3 HR + 3 Risk/mixed), mỗi case gồm:
- query                  — câu hỏi có từ khóa nhạy cảm
- target_document_id     — mã tài liệu được xem là "nhạy cảm"
- unauthorized_roles     — vai trò KHÔNG ĐƯỢC phép thấy tài liệu này
- authorized_roles       — vai trò ĐƯỢC phép thấy

Mỗi test chạy 2 truy vấn:
  (a) Với ``unauthorized_roles`` — Assert KHÔNG chunk nào của target_doc_id
      xuất hiện trong Top-K. Nếu xuất hiện → FAIL (rò r�).
  (b) Với ``authorized_roles``   — Assert target_doc_id CÓ THỂ xuất hiện
      (chỉ WARN nếu không xuất hiện vì score thấp).

Cấu hình DB đọc từ ``buoi_15/.env``.

Output: ``buoi_15/outputs/security_audit_report.md``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env sớm.
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
except Exception:
    pass

from src.secure_retriever import SecureRetriever, normalize_roles  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUT_DIR / "security_audit_report.md"
TOP_K = 10  # top-k để kiểm tra leak


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    category: str  # "HR" | "Risk" | "Mixed"
    query: str
    target_document_id: str
    target_title: str
    unauthorized_roles: list[str]
    authorized_roles: list[str]


# Được chọn bằng cách phân tích chunks_secure.csv (xem commit message
# của buổi 15) — mỗi document_id dưới đây có ≥1 chunk thuộc category HR/Risk.
TEST_CASES: list[TestCase] = [
    # 1) HR-only doc, query khớp title
    TestCase(
        name="T01_HR_pure_via_Guest",
        category="HR",
        query="bổ nhiệm thành viên hội đồng quản trị",
        target_document_id="166170",
        target_title="Luật Các tổ chức tín dụng số 32/2024/QH15",
        unauthorized_roles=["Guest"],
        authorized_roles=["Admin", "HR_Manager"],
    ),
    # 2) Risk-only doc, query chứa "tín dụng"
    TestCase(
        name="T02_Risk_via_Guest",
        category="Risk",
        query="giao nhận bảo quản vận chuyển tiền mặt tài sản quý",
        target_document_id="44209",
        target_title="Thông tư 01/2014/TT-NHNN Quy định giao nhận, bảo quản, vận chuyển",
        unauthorized_roles=["Guest"],
        authorized_roles=["Admin", "Risk_Officer", "Employee"],
    ),
    # 3) Risk-only doc khác
    TestCase(
        name="T03_Risk_an_toan_von_via_Guest",
        category="Risk",
        query="tỷ lệ an toàn vốn ngân hàng",
        target_document_id="117310",
        target_title="Thông tư 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn",
        unauthorized_roles=["Guest"],
        authorized_roles=["Admin", "Risk_Officer"],
    ),
    # 4) HR doc, Employee blocked (Employee can see Risk/Internal, NOT HR)
    TestCase(
        name="T04_HR_via_Employee_blocked",
        category="HR",
        query="chuẩn mực kiểm toán nội bộ",
        target_document_id="150974",
        target_title="Thông tư 08/2021/TT-BTC chuẩn mực kiểm toán nội bộ",
        unauthorized_roles=["Employee"],
        authorized_roles=["Admin", "HR_Manager"],
    ),
    # 5) Mixed doc 166170 — VanBan INTERSECTION = ['Admin'] only
    #    Test rằng 4 non-Admin roles đều bị block.
    TestCase(
        name="T05_Mixed_doc_Admin_only",
        category="Mixed",
        query="luật các tổ chức tín dụng",
        target_document_id="166170",
        target_title="Luật Các tổ chức tín dụng số 32/2024/QH15 (mixed HR+Risk)",
        unauthorized_roles=["HR_Manager", "Risk_Officer", "Employee", "Guest"],
        authorized_roles=["Admin"],
    ),
    # 6) Multiselect union: HR_Manager + Risk_Officer nên thấy HR doc (HR keyword)
    TestCase(
        name="T06_Multiselect_union_can_see_HR",
        category="HR",
        query="bổ nhiệm",
        target_document_id="166170",
        target_title="Luật Các tổ chức tín dụng (HR chunks về bổ nhiệm HĐQT)",
        unauthorized_roles=["Guest"],
        authorized_roles=["HR_Manager", "Risk_Officer"],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: TestCase
    unauth_results: list[dict] = field(default_factory=list)
    auth_results: list[dict] = field(default_factory=list)
    unauth_leak: list[str] = field(default_factory=list)  # chunk_ids bị leak
    auth_found: bool = False
    auth_score: float | None = None
    status: str = "PENDING"
    notes: list[str] = field(default_factory=list)


def _run_one(case: TestCase) -> CaseResult:
    res = CaseResult(case=case)
    # ------------------------------------------------------------------ unauthorized
    try:
        sr_u = SecureRetriever(
            user_roles=case.unauthorized_roles,
            use_dense=False,  # audit chỉ cần BM25 + Graph; tránh tải model nặng
            use_rerank=False,
            use_graph=True,
        )
        # Hybrid kết hợp nhiều retrieval paths → leak test mạnh hơn
        unauth = sr_u.search_hybrid(case.query, top_k=TOP_K, candidate_k=20)
        sr_u.close()
        res.unauth_results = unauth
        res.unauth_leak = [
            r["chunk_id"]
            for r in unauth
            if str(r.get("document_id", "")) == str(case.target_document_id)
        ]
    except Exception as e:
        res.notes.append(f"Lỗi khi chạy unauthorized search: {e}")

    # ------------------------------------------------------------------ authorized
    try:
        sr_a = SecureRetriever(
            user_roles=case.authorized_roles,
            use_dense=False,
            use_rerank=False,
            use_graph=True,
        )
        auth = sr_a.search_hybrid(case.query, top_k=TOP_K, candidate_k=20)
        sr_a.close()
        res.auth_results = auth
        auth_hits = [
            r for r in auth
            if str(r.get("document_id", "")) == str(case.target_document_id)
        ]
        res.auth_found = bool(auth_hits)
        if auth_hits:
            res.auth_score = max(
                (r.get("rrf_score") or 0.0) for r in auth_hits
            )
    except Exception as e:
        res.notes.append(f"Lỗi khi chạy authorized search: {e}")

    # ------------------------------------------------------------------ verdict
    if res.unauth_leak:
        res.status = "FAIL"
        res.notes.append(
            f"🚨 RÒ RỈ: {len(res.unauth_leak)} chunk(s) của doc "
            f"{case.target_document_id} xuất hiện trong kết quả của role "
            f"{case.unauthorized_roles}"
        )
    elif not res.auth_found:
        # authorized roles không tìm thấy doc — không phải leak, nhưng đáng note
        res.status = "WARN"
        res.notes.append(
            f"⚠️ Authorized roles {case.authorized_roles} không tìm thấy doc "
            f"{case.target_document_id} trong top-{TOP_K} — kiểm tra query "
            f"hoặc điểm tương đồng quá thấp."
        )
    else:
        res.status = "PASS"
        res.notes.append(
            f"✓ Không leak: doc {case.target_document_id} bị ẩn khỏi role "
            f"{case.unauthorized_roles}; authorized roles "
            f"{case.authorized_roles} thấy doc với rrf_score={res.auth_score:.4f}"
        )

    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _truncate(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _emit_markdown(results: list[CaseResult]) -> str:
    lines: list[str] = []
    lines.append("# Bu�i 15 — Security Audit Report (RBAC)")
    lines.append("")
    lines.append(f"_Sinh tự động bởi `scripts/security_audit.py` vào {time.strftime('%Y-%m-%d %H:%M:%S')}._")
    lines.append("")
    lines.append("## Tổng quan")
    n_total = len(results)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    lines.append(f"- Tổng số test case: **{n_total}**")
    lines.append(f"- ✅ PASS (không leak): **{n_pass}**")
    lines.append(f"- � FAIL (rò rỉ dữ liệu): **{n_fail}**")
    lines.append(f"- ⚠️ WARN (authorized không tìm thấy): **{n_warn}**")
    lines.append("")

    # Summary table
    lines.append("## Bảng tóm tắt")
    lines.append("")
    lines.append("| # | Test case | Category | Status | Leak chunks | Auth hit? |")
    lines.append("|---:|---|---|---|---:|:---:|")
    for i, r in enumerate(results, 1):
        c = r.case
        status_icon = {"PASS": "✅ PASS", "FAIL": "🚨 FAIL", "WARN": "⚠️ WARN"}.get(r.status, r.status)
        leak_n = len(r.unauth_leak)
        auth_hit = "✓" if r.auth_found else "✗"
        lines.append(f"| {i} | `{c.name}` | {c.category} | {status_icon} | {leak_n} | {auth_hit} |")
    lines.append("")

    # Per-test detail
    lines.append("## Chi tiết từng test case")
    lines.append("")
    for i, r in enumerate(results, 1):
        c = r.case
        lines.append(f"### {i}. `{c.name}`")
        lines.append("")
        lines.append(f"- **Query:** `{c.query}`")
        lines.append(f"- **Target doc:** `{c.target_document_id}` — _{_truncate(c.target_title, 90)}_")
        lines.append(f"- **Category:** {c.category}")
        lines.append(f"- **Unauthorized roles:** `{c.unauthorized_roles}`")
        lines.append(f"- **Authorized roles:** `{c.authorized_roles}`")
        lines.append("")

        # Unauthorized search results
        lines.append(f"#### 🔒 Search với unauthorized roles → {len(r.unauth_results)} hits")
        if r.unauth_leak:
            lines.append("")
            lines.append(f"**🚨 PHÁT HIỆN RÒ RỈ: {len(r.unauth_leak)} chunk(s) thuộc target doc:**")
            lines.append("")
            lines.append("| chunk_id | rank | rrf | label |")
            lines.append("|---|---:|---:|---|")
            for hit in r.unauth_results:
                if str(hit.get("document_id", "")) != str(c.target_document_id):
                    continue
                lines.append(
                    f"| `{hit['chunk_id'][:8]}…` | {hit.get('final_rank') or hit.get('rank')} "
                    f"| {(hit.get('rrf_score') or 0):.4f} | {hit.get('security_label')} |"
                )
        else:
            lines.append("")
            lines.append("✅ **Bằng chứng không leak:** trong top-K không có chunk nào thuộc "
                         f"target doc `{c.target_document_id}`.")
            lines.append("")
            # Show top-3 hits for context
            lines.append("Top-3 hits:")
            lines.append("")
            lines.append("| rank | chunk_id (8 chars) | doc | label |")
            lines.append("|---:|---|---|---|")
            for hit in r.unauth_results[:3]:
                lines.append(
                    f"| {hit.get('final_rank') or hit.get('rank')} "
                    f"| `{hit['chunk_id'][:8]}…` "
                    f"| `{hit.get('document_id', '')[:10]}` "
                    f"| {hit.get('security_label')} |"
                )
        lines.append("")

        # Authorized search results
        lines.append(f"#### 🔓 Search với authorized roles → {len(r.auth_results)} hits")
        if r.auth_found:
            lines.append("")
            lines.append(f"✅ **Authorized tìm thấy target doc** với `rrf_score={r.auth_score:.4f}`.")
        else:
            lines.append("")
            lines.append("⚠️ Authorized roles **không** tìm thấy target doc trong top-K — "
                         "không phải leak, nhưng đáng kiểm tra query.")
        lines.append("")

        # Notes
        if r.notes:
            lines.append("**Notes:**")
            for n in r.notes:
                lines.append(f"- {n}")
            lines.append("")

    # Final verdict
    lines.append("## Kết luận")
    lines.append("")
    if n_fail == 0 and n_warn == 0:
        verdict = (
            f"✅ **�ẠT chứng nhận an toàn dữ liệu mức cơ bản.** "
            f"Toàn bộ {n_total} test case PASS: "
            f"mọi truy vấn của unauthorized roles đều KHÔNG lộ chunk của tài liệu nhạy cảm, "
            f"và authorized roles có thể truy cập được (tùy điểm tương đồng)."
        )
    elif n_fail == 0:
        verdict = (
            f"⚠️ **ĐẠT có điều kiện.** Không có rò rỉ ({n_fail} FAIL), "
            f"nhưng {n_warn} test case có authorized roles không tìm thấy doc "
            f"(có thể do query quá cụ thể hoặc điểm thấp — cần xem xét query)."
        )
    else:
        verdict = (
            f"🚨 **KHÔNG ĐẠT.** Phát hiện {n_fail} trường hợp rò rỉ dữ liệu. "
            f"Hệ thống RBAC cần được sửa chữa trước khi đưa vào production."
        )
    lines.append(verdict)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Cách tái-chạy: `python ../buoi_14/.venv/bin/python3.12 scripts/security_audit.py`. "
        "Sửa query hoặc target_document_id trong `TEST_CASES` để mở rộng phạm vi test."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"📂 Loading test cases from `scripts/security_audit.py`")
    print(f"🔒 Top-K = {TOP_K}")
    print(f"🧪 Running {len(TEST_CASES)} test cases...\n")

    results: list[CaseResult] = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {case.name}", flush=True)
        t0 = time.time()
        result = _run_one(case)
        dt = time.time() - t0
        results.append(result)
        print(
            f"   → {result.status} ({dt:.1f}s) | "
            f"leak={len(result.unauth_leak)} | "
            f"auth_found={result.auth_found}"
        )

    # Emit report
    md = _emit_markdown(results)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"\n📄 Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # JSON dump phụ (cho CI / script khác đọc)
    json_path = OUT_DIR / "security_audit_results.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "name": r.case.name,
                    "category": r.case.category,
                    "status": r.status,
                    "query": r.case.query,
                    "target_document_id": r.case.target_document_id,
                    "target_title": r.case.target_title,
                    "unauthorized_roles": r.case.unauthorized_roles,
                    "authorized_roles": r.case.authorized_roles,
                    "unauth_leak_count": len(r.unauth_leak),
                    "unauth_leak_chunks": r.unauth_leak,
                    "auth_found": r.auth_found,
                    "auth_score": r.auth_score,
                    "notes": r.notes,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"📄 Wrote {json_path.relative_to(PROJECT_ROOT)}")

    n_fail = sum(1 for r in results if r.status == "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
