"""Suppress transformers' docstring-validation noise.

`transformers` v5+ chạy `@auto_docstring` decorator lúc khai báo class, gọi
`_process_parameters_section` (transformers/utils/auto_docstring.py:3897).
Hàm này có một đoạn:

    if len(undocumented_parameters) > 0:
        print("\\n".join(undocumented_parameters))

→ in `[ERROR] 'foo' is part of XKwargs, but not documented...` cho mỗi image
processor (vision) class khi transformers' lazy-scan submodules.

Đây KHÔNG phải logging → `logging.getLogger("transformers").setLevel(CRITICAL)`
KHÔNG triệt được. Phải chặn từ `builtins.print`.

Helper này patch `print` để bỏ qua các message:
- bắt đầu bằng "[ERROR]" HOẶC chứa "is part of ... but not documented"
- VÀ đến từ module bắt đầu với "transformers."

Được gọi 1 lần trước khi load transformers.
"""

from __future__ import annotations

import builtins
import sys


_NOISE_FRAGMENTS = (
    "is part of",
    "but not documented",
    "Make sure to add it to the docstring",
)


def install_quiet_print_once() -> None:
    """Monkey-patch builtins.print để lọc transformers' docstring-validation noise.

    Idempotent: nhiều lần gọi cũng chỉ patch 1 lần (kiểm tra marker trên hàm).
    """
    cur = builtins.print
    if getattr(cur, "_buoi14_quiet_print_patched", False):
        return

    _orig_print = cur

    def _quiet_print(*args, **kwargs):
        try:
            msg = " ".join(str(a) for a in args)
        except Exception:
            msg = ""
        if msg and (
            msg.lstrip().startswith("[ERROR]")
            or all(f in msg for f in _NOISE_FRAGMENTS)
        ):
            # Lấy caller frame để xác định module
            try:
                f = sys._getframe(1)
                mod = (f.f_globals.get("__name__") or "")
                if mod.startswith("transformers."):
                    return
            except Exception:
                # Không lấy được frame → chặn luôn để an toàn
                return
        return _orig_print(*args, **kwargs)

    _quiet_print._buoi14_quiet_print_patched = True
    builtins.print = _quiet_print