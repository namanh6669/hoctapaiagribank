#!/usr/bin/env python3
"""Kiểm tra môi trường OCR/RAG cho Buổi 5.

Script này chỉ đọc thông tin môi trường và thử import thư viện.
Không đọc/in biến môi trường chứa secret, không sửa PDF gốc.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CheckItem:
    name: str
    import_name: str | None
    package_name: str | None
    purpose: str
    fix_hint: str
    checker: Callable[[], tuple[bool, str]] | None = None


@dataclass(frozen=True)
class CheckResult:
    tool: str
    status: str
    detail: str
    fix_hint: str


def _version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "không rõ phiên bản"


def check_python() -> tuple[bool, str]:
    version = sys.version_info
    detail = f"Python {version.major}.{version.minor}.{version.micro} tại {sys.executable}"
    # Python 3.10+ phù hợp cho hầu hết thư viện RAG hiện đại; 3.11/3.12 càng tốt.
    return version >= (3, 10), detail


def check_import(import_name: str, package_name: str | None) -> tuple[bool, str]:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - cần báo lỗi import rõ ràng cho người mới học
        return False, f"Không import được `{import_name}`: {exc.__class__.__name__}: {exc}"

    version = None
    if package_name:
        version = _version(package_name)
    elif hasattr(module, "__version__"):
        version = str(module.__version__)

    if version:
        return True, f"Import `{import_name}` OK, phiên bản {version}"
    return True, f"Import `{import_name}` OK"


def run_checks() -> list[CheckResult]:
    checks = [
        CheckItem(
            name="Python",
            import_name=None,
            package_name=None,
            purpose="Chạy script kiểm tra và pipeline OCR/RAG.",
            fix_hint="Cài Python 3.10+ rồi tạo lại môi trường ảo bằng `python3 -m venv .venv`.",
            checker=check_python,
        ),
        CheckItem(
            name="PyMuPDF",
            import_name="pymupdf",
            package_name="PyMuPDF",
            purpose="Đọc PDF, trích xuất trang/ảnh phục vụ OCR.",
            fix_hint="Cài bằng `python -m pip install PyMuPDF`.",
        ),
        CheckItem(
            name="Pillow",
            import_name="PIL",
            package_name="Pillow",
            purpose="Xử lý ảnh trang PDF trước/sau OCR.",
            fix_hint="Cài bằng `python -m pip install Pillow`.",
        ),
        CheckItem(
            name="llama_cloud",
            import_name="llama_cloud",
            package_name="llama-cloud",
            purpose="Kết nối Llama Cloud khi bài học cần parser/cloud pipeline.",
            fix_hint="Cài bằng `python -m pip install llama-cloud`.",
        ),
        CheckItem(
            name="Pydantic",
            import_name="pydantic",
            package_name="pydantic",
            purpose="Định nghĩa schema dữ liệu và validate metadata.",
            fix_hint="Cài bằng `python -m pip install pydantic`.",
        ),
        CheckItem(
            name="Streamlit",
            import_name="streamlit",
            package_name="streamlit",
            purpose="Tạo giao diện demo kiểm tra OCR/RAG.",
            fix_hint="Cài bằng `python -m pip install streamlit`.",
        ),
        CheckItem(
            name="python-dotenv",
            import_name="dotenv",
            package_name="python-dotenv",
            purpose="Đọc file .env cục bộ mà không in secret ra màn hình.",
            fix_hint="Cài bằng `python -m pip install python-dotenv`.",
        ),
    ]

    results: list[CheckResult] = []
    for item in checks:
        if item.checker is not None:
            ok, detail = item.checker()
        else:
            assert item.import_name is not None
            ok, detail = check_import(item.import_name, item.package_name)
        results.append(
            CheckResult(
                tool=item.name,
                status="PASS" if ok else "FAIL",
                detail=detail,
                fix_hint="-" if ok else item.fix_hint,
            )
        )
    return results


def print_table(results: list[CheckResult]) -> None:
    headers = ["Công cụ", "Trạng thái", "Chi tiết"]
    rows = [[r.tool, r.status, r.detail] for r in results]
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(sep: str = "-") -> str:
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    print(line())
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(line("="))
    for row in rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line())


def print_fix_hints(results: list[CheckResult]) -> None:
    failed = [r for r in results if r.status == "FAIL"]
    if not failed:
        print("\nTất cả kiểm tra đều PASS. Không cần tự khắc phục thêm.")
        return

    print("\nHướng dẫn tự khắc phục từng trạng thái FAIL:")
    for result in failed:
        print(f"- {result.tool}: {result.fix_hint}")
    print("\nGợi ý cài toàn bộ Python packages còn thiếu trong môi trường hiện tại:")
    print("python -m pip install PyMuPDF Pillow llama-cloud pydantic streamlit python-dotenv")
    print("\nLưu ý: nếu thiếu phần mềm hệ thống như Tesseract/Poppler, hãy hỏi người hướng dẫn trước khi cài.")


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    print("Kiểm tra môi trường OCR/RAG Buổi 5")
    print(f"Thư mục bài học: {project_dir}")
    print(f"Hệ điều hành: {platform.system()} {platform.release()}")
    print("Không in secret, không sửa PDF gốc.\n")

    results = run_checks()
    print_table(results)
    print_fix_hints(results)
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
