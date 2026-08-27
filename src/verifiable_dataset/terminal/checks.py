"""Declarative checks evaluated against a host-side snapshot.

Tasks declare what must be true in ``task.yaml`` instead of shipping a
checker script, so checks can be generated automatically from the state
diff between the initial world and the reference solution's result.

Every op reads the snapshot on the host. The one exception is ``run_*``,
which needs to execute something: it does so in a *fresh* container with
the snapshot copied in, never in the episode's own container, so nothing
the agent left running can influence the result.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DOCKER = shutil.which("docker") or "docker"


class CheckError(Exception):
    """Raised when a check is malformed (not when it merely fails)."""


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _resolve(root: Path, path: str) -> Path:
    """Resolve a task-relative path, refusing to escape the snapshot."""
    target = (root / path.lstrip("/")).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise CheckError(f"snapshot disina cikan yol: {path!r}")
    return target


def _read(root: Path, path: str) -> str:
    return _resolve(root, path).read_text(encoding="utf-8", errors="replace")


# -- filesystem ops ---------------------------------------------------

def op_is_dir(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    return CheckResult("is_dir", p.is_dir(), f"{spec['path']} yok veya dizin degil")


def op_is_file(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    return CheckResult("is_file", p.is_file(), f"{spec['path']} yok veya dosya degil")


def op_not_exists(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    return CheckResult("not_exists", not p.exists(), f"{spec['path']} hala duruyor")


def op_file_content_eq(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    if not p.is_file():
        return CheckResult("file_content_eq", False, f"{spec['path']} yok")
    actual = _read(root, spec["path"])
    expected = str(spec["value"])
    if spec.get("strip", True):
        actual, expected = actual.strip(), expected.strip()
    return CheckResult(
        "file_content_eq", actual == expected,
        f"{spec['path']}: beklenen {expected!r}, bulunan {actual!r}",
    )


def op_file_contains(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    if not p.is_file():
        return CheckResult("file_contains", False, f"{spec['path']} yok")
    return CheckResult(
        "file_contains", str(spec["value"]) in _read(root, spec["path"]),
        f"{spec['path']} icinde {spec['value']!r} yok",
    )


def op_file_matches(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    if not p.is_file():
        return CheckResult("file_matches", False, f"{spec['path']} yok")
    ok = re.search(spec["value"], _read(root, spec["path"]), re.MULTILINE) is not None
    return CheckResult("file_matches", ok, f"{spec['path']} {spec['value']!r} ile eslesmiyor")


def op_file_line_count_eq(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    if not p.is_file():
        return CheckResult("file_line_count_eq", False, f"{spec['path']} yok")
    lines = [ln for ln in _read(root, spec["path"]).splitlines() if ln.strip()]
    return CheckResult(
        "file_line_count_eq", len(lines) == int(spec["value"]),
        f"{spec['path']}: beklenen {spec['value']} satir, bulunan {len(lines)}",
    )


def op_dir_entries_eq(root: Path, spec: dict) -> CheckResult:
    p = _resolve(root, spec["path"])
    if not p.is_dir():
        return CheckResult("dir_entries_eq", False, f"{spec['path']} dizin degil")
    actual = sorted(e.name for e in p.iterdir())
    expected = sorted(str(v) for v in spec["value"])
    return CheckResult(
        "dir_entries_eq", actual == expected,
        f"{spec['path']}: beklenen {expected}, bulunan {actual}",
    )


# -- execution op -----------------------------------------------------

def op_run_stdout_eq(root: Path, spec: dict) -> CheckResult:
    """Run a command over the snapshot in a throwaway container."""
    image = spec["_image"]
    workdir = spec.get("workdir", "/workspace")
    proc = subprocess.run(
        [DOCKER, "run", "-d", "--rm", "--network", "none", "--cpus", "1",
         "--memory", "512m", "--pids-limit", "256", "--workdir", workdir,
         image, "sleep", "300"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if proc.returncode != 0:
        raise CheckError(f"grade konteyneri baslatilamadi: {proc.stderr.strip()}")
    cid = proc.stdout.strip()
    try:
        subprocess.run([DOCKER, "cp", f"{root}/.", f"{cid}:{workdir}"],
                       capture_output=True, timeout=120, check=True)
        run = subprocess.run(
            [DOCKER, "exec", "--workdir", workdir, cid, "bash", "-lc", spec["cmd"]],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=spec.get("timeout", 30),
        )
    except subprocess.SubprocessError as e:
        raise CheckError(f"grade komutu calistirilamadi: {e}") from e
    finally:
        subprocess.run([DOCKER, "kill", cid], capture_output=True, timeout=60)

    actual, expected = run.stdout, str(spec["value"])
    if spec.get("strip", True):
        actual, expected = actual.strip(), expected.strip()
    return CheckResult(
        "run_stdout_eq", actual == expected,
        f"`{spec['cmd']}`: beklenen {expected!r}, bulunan {actual!r} "
        f"(exit={run.returncode}, stderr={run.stderr.strip()[:200]!r})",
    )


OPS: dict[str, Callable[[Path, dict], CheckResult]] = {
    "is_dir": op_is_dir,
    "is_file": op_is_file,
    "not_exists": op_not_exists,
    "file_content_eq": op_file_content_eq,
    "file_contains": op_file_contains,
    "file_matches": op_file_matches,
    "file_line_count_eq": op_file_line_count_eq,
    "dir_entries_eq": op_dir_entries_eq,
    "run_stdout_eq": op_run_stdout_eq,
}


def run_checks(root: Path, specs: list[dict], image: str) -> list[dict]:
    """Evaluate every check against the snapshot rooted at ``root``."""
    results: list[dict] = []
    for i, spec in enumerate(specs):
        op_name = spec.get("op")
        fn = OPS.get(op_name)
        if fn is None:
            results.append({"name": f"{i}:{op_name}", "ok": False,
                            "detail": f"bilinmeyen op: {op_name}"})
            continue
        try:
            res = fn(root, {**spec, "_image": image})
            label = f"{i}:{res.name}({spec.get('path') or spec.get('cmd', '')})"
            results.append({"name": label, "ok": res.ok,
                            "detail": "" if res.ok else res.detail})
        except (CheckError, OSError) as e:
            results.append({"name": f"{i}:{op_name}", "ok": False, "detail": str(e)})
    return results


def report(results: list[dict]) -> dict[str, Any]:
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    return {
        "checks": results,
        "passed": passed,
        "total": total,
        "reward": passed / total if total else 0.0,
        "solved": total > 0 and passed == total,
    }
