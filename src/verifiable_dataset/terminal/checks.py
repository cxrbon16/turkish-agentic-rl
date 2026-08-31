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

import json
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


def op_json_field_eq(root: Path, spec: dict) -> CheckResult:
    """Compare one field of a JSON file.

    Numbers are compared with a tolerance and accept a numeric string, so a
    model that writes "60.1" instead of 60.1 is not punished for JSON typing.
    Lists can be compared as sets when order carries no meaning.
    """
    p = _resolve(root, spec["path"])
    if not p.is_file():
        return CheckResult("json_field_eq", False, f"{spec['path']} yok")
    try:
        data = json.loads(_read(root, spec["path"]))
    except json.JSONDecodeError as e:
        return CheckResult("json_field_eq", False, f"{spec['path']} gecerli JSON degil: {e}")

    field = spec["field"]
    if not isinstance(data, dict) or field not in data:
        return CheckResult("json_field_eq", False,
                           f"{spec['path']} icinde '{field}' alani yok "
                           f"(bulunan alanlar: {list(data)[:10] if isinstance(data, dict) else type(data).__name__})")

    actual, expected = data[field], spec["value"]

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return CheckResult("json_field_eq", False, f"'{field}' liste degil: {actual!r}")
        norm = (lambda x: str(x).strip()) if spec.get("strip", True) else str
        a, e = ([norm(x) for x in actual], [norm(x) for x in expected])
        ok = set(a) == set(e) if spec.get("as_set", True) else a == e
        missing, extra = sorted(set(e) - set(a)), sorted(set(a) - set(e))
        return CheckResult("json_field_eq", ok,
                           f"'{field}': eksik={missing[:6]} fazla={extra[:6]}")

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            num = float(actual)
        except (TypeError, ValueError):
            return CheckResult("json_field_eq", False, f"'{field}' sayi degil: {actual!r}")
        tol = spec.get("tol", 0.01)
        return CheckResult("json_field_eq", abs(num - float(expected)) <= tol,
                           f"'{field}': beklenen {expected}, bulunan {actual!r}")

    a_s, e_s = str(actual), str(expected)
    if spec.get("strip", True):
        a_s, e_s = a_s.strip(), e_s.strip()
    return CheckResult("json_field_eq", a_s == e_s,
                       f"'{field}': beklenen {e_s!r}, bulunan {a_s!r}")


def _json_set_key(items: list) -> list[str]:
    return sorted(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in items)


def op_json_eq(root: Path, spec: dict) -> CheckResult:
    """Compare a whole JSON document structurally.

    Needed when the output's top level is not an object -- a list or a
    scalar -- so there are no fields to compare one by one. Parsing before
    comparing means indentation, key order and ensure_ascii do not decide
    the reward, which byte equality would.
    """
    p = _resolve(root, spec["path"])
    if not p.is_file():
        return CheckResult("json_eq", False, f"{spec['path']} yok")
    try:
        actual = json.loads(_read(root, spec["path"]))
    except json.JSONDecodeError as e:
        return CheckResult("json_eq", False, f"{spec['path']} gecerli JSON degil: {e}")

    expected = spec["value"]
    if spec.get("as_set") and isinstance(actual, list) and isinstance(expected, list):
        ok = _json_set_key(actual) == _json_set_key(expected)
    else:
        ok = actual == expected
    return CheckResult(
        "json_eq", ok,
        f"{spec['path']}: beklenen {json.dumps(expected, ensure_ascii=False)[:160]}, "
        f"bulunan {json.dumps(actual, ensure_ascii=False)[:160]}",
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
    "json_field_eq": op_json_field_eq,
    "json_eq": op_json_eq,
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
    """Score a check run.

    ``reward`` is binary on purpose: the whole task or nothing. Partial
    credit pays for work that merely looks like progress -- a solution that
    creates the reference's scratch files, or half-fills the output, earns
    reward without being right, and the policy learns to chase the checks
    instead of the goal. Endless Terminals (arXiv:2601.16443) trains on a
    binary episode reward for the same reason.

    ``partial`` keeps the fraction for diagnostics only: a task nobody
    solves but everybody nearly solves looks very different from one nobody
    gets anywhere on, and that difference is worth seeing in a sweep.
    """
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    solved = total > 0 and passed == total
    return {
        "checks": results,
        "passed": passed,
        "total": total,
        "reward": 1.0 if solved else 0.0,
        "partial": passed / total if total else 0.0,
        "solved": solved,
    }
