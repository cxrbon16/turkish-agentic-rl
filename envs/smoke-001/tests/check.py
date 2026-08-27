"""Checker for smoke-001.

Runs on the HOST against a snapshot of the container's workdir -- never
inside the sandbox -- so nothing the agent did can influence grading.

Usage:  python check.py <snapshot_root>
Emits a JSON report on stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_CONTENT = "yavuz123"


def run_checks(root: Path) -> list[dict]:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    cosmos = root / "cosmos"
    record("cosmos_is_dir", cosmos.is_dir(), f"{cosmos} yok veya dizin degil")

    yavuz_dir = root / "yavuz"
    record("yavuz_is_dir", yavuz_dir.is_dir(), f"{yavuz_dir} yok veya dizin degil")

    target = cosmos / "yavuz.txt"
    if not target.is_file():
        record("yavuz_txt_is_file", False, f"{target} yok veya dosya degil")
        record("yavuz_txt_content", False, "dosya olmadigi icin icerik kontrol edilemedi")
        return checks

    record("yavuz_txt_is_file", True)
    raw = target.read_text(encoding="utf-8", errors="replace")
    # Trailing newline'a duyarsiz: `echo` da `printf` de kabul edilir.
    actual = raw.strip()
    record(
        "yavuz_txt_content",
        actual == EXPECTED_CONTENT,
        f"beklenen {EXPECTED_CONTENT!r}, bulunan {actual!r}",
    )
    return checks


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: check.py <snapshot_root>"}))
        return 2

    root = Path(sys.argv[1])
    checks = run_checks(root)
    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    report = {
        "checks": checks,
        "passed": passed,
        "total": total,
        "reward": passed / total if total else 0.0,
        "solved": passed == total,
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
