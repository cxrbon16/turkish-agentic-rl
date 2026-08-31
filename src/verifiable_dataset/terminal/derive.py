"""Derive declarative checks from the state diff a reference solution causes.

    # tek task
    python -m verifiable_dataset.terminal.derive --task envs/tr-002-hata-say

    # hepsi: turetilen check'leri elle yazilanlarla karsilastir
    python -m verifiable_dataset.terminal.derive --all

Setup calistirilip dunyanin A fotografi cekilir, sonra referans cozum
calistirilip B fotografi cekilir. Aradaki fark check'e cevrilir.

Check'ler cozumun *gercekten yaptigi* isten turedigi icin dataset kendi
verifier'indan sapamaz. Uretilmis task'larda bu sart: bir LLM'in "herhalde
soyle bir dosya olusur" tahminine guvenmek, sessizce bozuk task uretmenin
en kisa yolu.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from verifiable_dataset.terminal.checks import report as checks_report
from verifiable_dataset.terminal.checks import run_checks
from verifiable_dataset.terminal.sandbox import DockerSandbox, docker_available
from verifiable_dataset.terminal.task import Task

# Bu sayidan uzun dosyalarda satir satir file_contains uretmek yerine
# tek bir file_content_eq daha okunakli.
MAX_LINE_CHECKS = 8

PROGRAM_SUFFIXES = {".py", ".sh", ".bash"}


# -- snapshot tarama --------------------------------------------------

def scan(root: Path) -> tuple[dict[str, frozenset], dict[str, bytes]]:
    """Bir snapshot'i (dizin -> cocuk adlari, dosya -> icerik) haline getir."""
    dirs: dict[str, frozenset] = {}
    files: dict[str, bytes] = {}
    for p in root.rglob("*"):
        if p.is_symlink():
            continue  # symlink semantigi ayri bir konu; simdilik kapsam disi
        rel = p.relative_to(root).as_posix()
        if p.is_dir():
            dirs[rel] = frozenset(c.name for c in p.iterdir())
        elif p.is_file():
            files[rel] = p.read_bytes()
    return dirs, files


def _ancestors(path: str) -> list[str]:
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def _topmost(paths: set[str]) -> list[str]:
    """Silinen bir dizinin cocuklari icin ayrica not_exists uretme."""
    return sorted(p for p in paths if not any(a in paths for a in _ancestors(p)))


@dataclass
class TreeDiff:
    created_dirs: list[str] = field(default_factory=list)
    changed_dirs: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.created_dirs or self.changed_dirs or self.created_files
                    or self.modified_files or self.deleted)


def diff_trees(a: tuple[dict, dict], b: tuple[dict, dict]) -> TreeDiff:
    a_dirs, a_files = a
    b_dirs, b_files = b
    d = TreeDiff()

    for rel in sorted(b_dirs):
        if rel not in a_dirs:
            d.created_dirs.append(rel)
        elif a_dirs[rel] != b_dirs[rel]:
            d.changed_dirs.append(rel)

    for rel in sorted(b_files):
        if rel not in a_files:
            d.created_files.append(rel)
        elif a_files[rel] != b_files[rel]:
            d.modified_files.append(rel)

    gone = (set(a_dirs) | set(a_files)) - (set(b_dirs) | set(b_files))
    d.deleted = _topmost(gone)
    return d


# -- check uretimi ----------------------------------------------------

def infer_kind(rel: str, raw: bytes) -> str:
    """Bir dosyanin nasil kontrol edilecegini adindan ve iceriginden tahmin et."""
    suffix = Path(rel).suffix.lower()
    if suffix in PROGRAM_SUFFIXES:
        return "program"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "opaque"
    stripped = text.strip()
    if not stripped:
        return "empty"
    if suffix == ".json" or stripped[0] in "{[":
        try:
            if isinstance(json.loads(stripped), dict):
                return "json"
        except json.JSONDecodeError:
            pass
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if 1 < len(lines) <= MAX_LINE_CHECKS:
        return "lines"
    return "file"


def json_field_checks(rel: str, text: str) -> tuple[list[dict], list[str]]:
    """JSON ciktisini alan alan kontrol et -- butun dosyayi byte byte degil.

    Anahtar sirasi, girinti ve ensure_ascii tercihi dogru cozumler arasinda
    degisir; alan bazli karsilastirma bunlari cezalandirmaz.
    """
    checks: list[dict] = []
    skipped: list[str] = []
    for key, value in json.loads(text).items():
        if isinstance(value, list):
            checks.append({"op": "json_field_eq", "path": rel, "field": key,
                           "value": value, "as_set": True})
        elif isinstance(value, bool) or value is None or isinstance(value, dict):
            # bool/None/ic ice dict bugunku op setinin disinda kaliyor
            skipped.append(f"{key} ({type(value).__name__})")
        elif isinstance(value, float):
            checks.append({"op": "json_field_eq", "path": rel, "field": key,
                           "value": value, "tol": 0.01})
        else:
            checks.append({"op": "json_field_eq", "path": rel, "field": key,
                           "value": value})
    return checks, skipped


def checks_for_file(rel: str, raw: bytes, override: dict | None,
                    sandbox: DockerSandbox | None) -> tuple[list[dict], list[str]]:
    kind = (override or {}).get("kind") or infer_kind(rel, raw)
    notes: list[str] = []

    if kind == "program":
        # Referansin yazdigi her betik hedefin parcasi degil: cogu sadece
        # ise yarayan bir arac. Bildirilmemis betige check koymak "sen de
        # tam boyle bir dosya olustur" demek olur ve tek satirlik dogru
        # cozumu haksiz yere duserdi. Bu yuzden yalnizca outputs: icinde
        # bildirilenler kontrol edilir.
        if override is None:
            notes.append(f"{rel}: ara dosya sayilip atlandi; hedefin parcasiysa "
                         f"outputs: icinde bildir")
            return [], notes
        # Bir programin dogrulugu diskteki metninde degil, calisinca ne
        # urettiginde. Diff bunu goremez, `run` bildirimi sart.
        out = [{"op": "is_file", "path": rel}]
        run = override.get("run")
        if not run:
            notes.append(f"{rel}: outputs: icinde `run` bildirilirse "
                         f"run_stdout_eq de uretilir")
        elif sandbox is not None:
            res = sandbox.exec(run)
            if res.exit_code == 0:
                out.append({"op": "run_stdout_eq", "cmd": run,
                            "value": res.stdout.strip()})
            else:
                notes.append(f"{rel}: `{run}` referans dunyasinda exit={res.exit_code}")
        return out, notes

    if kind == "opaque":
        notes.append(f"{rel}: metin degil, sadece varligi kontrol ediliyor")
        return [{"op": "is_file", "path": rel}], notes

    if kind == "empty":
        return [{"op": "is_file", "path": rel}], notes

    text = raw.decode("utf-8", errors="replace")

    if kind == "json":
        try:
            field_checks, skipped = json_field_checks(rel, text)
        except json.JSONDecodeError as e:
            notes.append(f"{rel}: JSON cozulemedi ({e}), icerik esitligine dusuldu")
            return ([{"op": "is_file", "path": rel},
                     {"op": "file_content_eq", "path": rel, "value": text.strip()}], notes)
        if skipped:
            notes.append(f"{rel}: kontrol edilemeyen alanlar: {', '.join(skipped)}")
        return [{"op": "is_file", "path": rel}] + field_checks, notes

    if kind == "lines":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        out = [{"op": "is_file", "path": rel},
               {"op": "file_line_count_eq", "path": rel, "value": len(lines)}]
        out += [{"op": "file_contains", "path": rel, "value": ln} for ln in lines]
        return out, notes

    return ([{"op": "is_file", "path": rel},
             {"op": "file_content_eq", "path": rel, "value": text.strip()}], notes)


def build_checks(diff: TreeDiff, b_dirs: dict, b_files: dict,
                 overrides: dict[str, dict],
                 sandbox: DockerSandbox | None) -> tuple[list[dict], list[str]]:
    checks: list[dict] = []
    notes: list[str] = []

    for rel in diff.created_dirs:
        checks.append({"op": "is_dir", "path": rel})
        checks.append({"op": "dir_entries_eq", "path": rel, "value": sorted(b_dirs[rel])})
    for rel in diff.changed_dirs:
        checks.append({"op": "dir_entries_eq", "path": rel, "value": sorted(b_dirs[rel])})

    for rel in diff.created_files + diff.modified_files:
        file_checks, file_notes = checks_for_file(
            rel, b_files[rel], overrides.get(rel), sandbox)
        checks.extend(file_checks)
        notes.extend(file_notes)

    for rel in diff.deleted:
        checks.append({"op": "not_exists", "path": rel})

    for rel in overrides:
        if rel not in b_files and rel not in b_dirs:
            notes.append(f"{rel}: outputs: icinde bildirilmis ama referans uretmiyor")

    return checks, notes


# -- surus ------------------------------------------------------------

@dataclass
class DeriveReport:
    task_id: str
    checks: list[dict]
    notes: list[str]
    ref_passed: int      # turetilen check'lerden kaci B fotografinda geciyor
    init_passed: int     # ... A fotografinda (hepsi gecerse task anlamsiz)
    total: int
    error: str | None = None

    @property
    def sound(self) -> bool:
        """Referansta tam gecip baslangicta gecmiyorsa check seti ise yarar."""
        return (self.total > 0 and self.ref_passed == self.total
                and self.init_passed < self.total)


def derive(task: Task) -> DeriveReport:
    if not task.reference_solution.strip():
        return DeriveReport(task.id, [], [], 0, 0, 0, "reference_solution yok")

    overrides = {o["path"]: o for o in task.outputs}

    with tempfile.TemporaryDirectory(prefix="vds-derive-") as tmp:
        with task.make_sandbox() as sandbox:
            task.prepare(sandbox)
            snap_a = sandbox.snapshot(Path(tmp) / "a")
            a = scan(snap_a)

            result = sandbox.exec(task.reference_solution, timeout=60)
            if result.exit_code != 0:
                return DeriveReport(
                    task.id, [], [], 0, 0, 0,
                    f"referans exit={result.exit_code}: {result.stderr.strip()[:200]}")

            snap_b = sandbox.snapshot(Path(tmp) / "b")
            b = scan(snap_b)

            d = diff_trees(a, b)
            if d.is_empty():
                return DeriveReport(task.id, [], [], 0, 0, 0,
                                    "referans cozum dunyayi degistirmiyor")
            checks, notes = build_checks(d, b[0], b[1], overrides, sandbox)

        # Turetilen check'ler iki fotografta da olculur: B'de hepsi gecmeli
        # (yoksa turetici kendi kendiyle celisiyor), A'da hepsi gecmemeli
        # (yoksa task baslangicta zaten cozulmus sayilir).
        ref = checks_report(run_checks(snap_b, checks, task.image))
        init = checks_report(run_checks(snap_a, checks, task.image))

    return DeriveReport(task.id, checks, notes,
                        ref["passed"], init["passed"], len(checks))


# -- elle yazilanla karsilastirma -------------------------------------

def signature(check: dict) -> tuple:
    return (check.get("op"), check.get("path") or check.get("cmd", ""), check.get("field"))


# Ayni yol uzerinde soldaki op'u gereksiz kilan, ondan daha guclu op'lar.
# Elle yazilmis bir check'i turetici daha siki bir op'la kapsiyorsa bu bir
# eksik degil; rapor ikisini karistirmamali.
SUBSUMED_BY: dict[str, set[str]] = {
    "is_file": {"file_content_eq", "file_line_count_eq", "file_contains",
                "file_matches", "json_field_eq", "run_stdout_eq"},
    "is_dir": {"dir_entries_eq"},
    "file_contains": {"file_content_eq"},
    "file_matches": {"file_content_eq"},
    "file_line_count_eq": {"file_content_eq"},
}


def compare(derived: list[dict], handwritten: list[dict]) -> tuple[list, list, list, list]:
    """(ortak, daha-gucluyle-kapsanan, gercekten-kacan, ekstra)."""
    d = {signature(c) for c in derived}
    h = {signature(c) for c in handwritten}

    by_path: dict[str, set[str]] = {}
    for op, path, _ in d:
        by_path.setdefault(path, set()).add(op)

    covered, missed = [], []
    for sig in sorted(h - d):
        op, path, _ = sig
        if by_path.get(path, set()) & SUBSUMED_BY.get(op, set()):
            covered.append(sig)
        else:
            missed.append(sig)
    return sorted(d & h), covered, missed, sorted(d - h)


def dump_checks(checks: list[dict]) -> str:
    lines = []
    for c in checks:
        body = yaml.safe_dump(c, default_flow_style=True, sort_keys=False,
                              allow_unicode=True, width=10 ** 6).strip()
        lines.append(f"  - {body}")
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="", help="tek task dizini")
    parser.add_argument("--all", action="store_true", help="envs/ altindaki her task")
    parser.add_argument("--envs", default="envs")
    parser.add_argument("--yaml", action="store_true", help="turetilen check'leri YAML bas")
    args = parser.parse_args()

    ok, info = docker_available()
    if not ok:
        print(f"Docker daemon'a ulasilamiyor: {info}")
        return 1

    if args.all:
        task_dirs = sorted(p.parent for p in Path(args.envs).glob("*/task.yaml")
                           if not p.parent.name.startswith("_"))
    elif args.task:
        task_dirs = [Path(args.task)]
    else:
        print("--task ya da --all ver")
        return 2

    n_sound = 0
    n_missed = 0
    for task_dir in task_dirs:
        task = Task.load(task_dir)
        try:
            rep = derive(task)
        except Exception as e:  # noqa: BLE001 - tek task cokerse digerleri devam etsin
            print(f"{task.id}\n  HATA {type(e).__name__}: {e}\n")
            continue

        print(task.id)
        if rep.error:
            print(f"  HATA {rep.error}\n")
            continue

        verdict = "saglam" if rep.sound else "SORUNLU"
        print(f"  turetilen      : {rep.total} check   [{verdict}]")
        print(f"  referansta (B) : {rep.ref_passed}/{rep.total} geciyor"
              f"{'' if rep.ref_passed == rep.total else '   <-- turetici kendiyle celisiyor'}")
        print(f"  baslangicta (A): {rep.init_passed}/{rep.total} geciyor"
              f"{'   <-- ayirt etmiyor!' if rep.init_passed == rep.total else ''}")
        n_sound += rep.sound

        if task.checks:
            common, covered, missed, extra = compare(rep.checks, task.checks)
            print(f"  elle yazilanla : {len(common)} ortak, {len(covered)} daha "
                  f"gucluyle kapsanan, {len(missed)} kacan, {len(extra)} ekstra")
            for op, path, fld in covered:
                print(f"     ~ kapsanan: {op}({path}) daha siki bir op'la karsilaniyor")
            for op, path, fld in missed:
                print(f"     - kacan   : {op}({path}{'.' + fld if fld else ''})")
            n_missed += len(missed)
        else:
            print("  elle yazilanla : (bu task eski tarz check.py kullaniyor)")

        for note in rep.notes:
            print(f"  ! {note}")
        if args.yaml:
            print("\nchecks:")
            print(dump_checks(rep.checks))
        print()

    print(f"OZET: {n_sound}/{len(task_dirs)} task'ta turetilen check seti saglam, "
          f"elle yazilanlardan {n_missed} check kacirildi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
