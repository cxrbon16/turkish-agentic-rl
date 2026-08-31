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
from typing import Any

import yaml

from verifiable_dataset.terminal.checks import _json_close
from verifiable_dataset.terminal.checks import report as checks_report
from verifiable_dataset.terminal.checks import run_checks
from verifiable_dataset.terminal.sandbox import DockerSandbox, docker_available
from verifiable_dataset.terminal.task import Task, run_script

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
            json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            # Sozluk olmak sart degil: ust duzeyi liste olan bir cikti da
            # JSON gibi kontrol edilmeli, satir satir metin gibi degil.
            return "json"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if 1 < len(lines) <= MAX_LINE_CHECKS:
        return "lines"
    return "file"


def beklenen_uyusmazligi(rel: str, kind: str, actual: Any, beklenen: Any) -> str:
    """Modelin bildirdigi degeri referansin gercekten urettigiyle karsilastir.

    Turetme, referansin DAVRANISINI hakikat kabul eder; bu yuzden buggy bir
    referans yanlis cevabi ground truth yapar ve her kapi memnun kalir.
    ``beklenen`` modelin NIYETI: ikisi celisiyorsa biri yanlistir ve task
    hazir degildir. Ayni yanitin icinde geldigi icin ek maliyeti yok.
    """
    if kind == "json":
        hedef = beklenen
        if isinstance(beklenen, str):
            try:
                hedef = json.loads(beklenen)
            except json.JSONDecodeError:
                return f"{rel}: beklenen gecerli JSON degil"
        if _json_close(actual, hedef, 0.01, True):
            return ""
    elif kind in {"lines", "dir"}:
        istenen = (beklenen if isinstance(beklenen, list)
                   else [ln.strip() for ln in str(beklenen).splitlines() if ln.strip()])
        if sorted(str(x).strip() for x in actual) == sorted(str(x).strip() for x in istenen):
            return ""
    elif str(actual).strip() == str(beklenen).strip():
        return ""
    return (f"{rel}: referansin urettigi deger, outputs'ta bildirilen beklenen "
            f"degerle uyusmuyor -- beklenen {str(beklenen)[:80]!r}, "
            f"uretilen {str(actual)[:80]!r}")


def checks_for_file(rel: str, raw: bytes, override: dict | None,
                    sandbox: DockerSandbox | None,
                    ) -> tuple[list[dict], list[str], list[str]]:
    """Bir artefakt icin TEK check uret.

    Odul ikili oldugu icin bir dosyaya uc check koymak ikisinden fazlasini
    getirmiyor: hepsi gecmek zorunda, yani konjonksiyondan baska bir sey
    degiller. Gevseklik ise op'un kendi semantiginde tasiniyor --
    json_eq sayilara tolerans, file_lines_eq siraya duyarsizlik verir.
    """
    kind = (override or {}).get("kind") or infer_kind(rel, raw)
    beklenen = (override or {}).get("beklenen")
    notes: list[str] = []
    sorunlar: list[str] = []

    def bitir(check: dict | None, deger: Any) -> tuple[list[dict], list[str], list[str]]:
        if beklenen is not None:
            fark = beklenen_uyusmazligi(rel, kind, deger, beklenen)
            if fark:
                sorunlar.append(fark)
        return ([check] if check else []), notes, sorunlar

    if kind == "program":
        # Referansin yazdigi her betik hedefin parcasi degil: cogu sadece
        # ise yarayan bir arac. Bildirilmemis betige check koymak "sen de
        # tam boyle bir dosya olustur" demek olur ve tek satirlik dogru
        # cozumu haksiz yere duserdi.
        if override is None:
            notes.append(f"{rel}: ara dosya sayilip atlandi; hedefin parcasiysa "
                         f"outputs: icinde bildir")
            return [], notes, sorunlar
        # Bir programin dogrulugu diskteki metninde degil, calisinca ne
        # urettiginde. Diff bunu goremez, `run` bildirimi sart.
        run = override.get("run")
        if not run:
            notes.append(f"{rel}: outputs: icinde `run` bildirilirse betigin "
                         f"metni yerine ciktisi kontrol edilir")
            return bitir({"op": "is_file", "path": rel}, "")
        if sandbox is None:
            return bitir({"op": "is_file", "path": rel}, "")
        res = sandbox.exec(run)
        if res.exit_code != 0:
            sorunlar.append(f"{rel}: `{run}` referans dunyasinda exit={res.exit_code}")
            return bitir({"op": "is_file", "path": rel}, "")
        return bitir({"op": "run_stdout_eq", "cmd": run, "value": res.stdout.strip()},
                     res.stdout.strip())

    if kind == "opaque":
        notes.append(f"{rel}: metin degil, sadece varligi kontrol ediliyor")
        return [{"op": "is_file", "path": rel}], notes, sorunlar

    if kind == "empty":
        return bitir({"op": "is_file", "path": rel}, "")

    text = raw.decode("utf-8", errors="replace")

    if kind == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            sorunlar.append(f"{rel}: kind=json bildirildi ama referansin urettigi "
                            f"dosya gecerli JSON degil ({e})")
            return [{"op": "file_content_eq", "path": rel, "value": text.strip()}], \
                notes, sorunlar
        return bitir({"op": "json_eq", "path": rel, "value": data}, data)

    if kind == "lines":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return bitir({"op": "file_lines_eq", "path": rel, "value": lines}, lines)

    return bitir({"op": "file_content_eq", "path": rel, "value": text.strip()},
                 text.strip())


def build_checks(diff: TreeDiff, b_dirs: dict, b_files: dict,
                 overrides: dict[str, dict],
                 sandbox: DockerSandbox | None,
                 ) -> tuple[list[dict], list[str], list[str]]:
    checks: list[dict] = []
    notes: list[str] = []
    sorunlar: list[str] = []

    # outputs: bildirildiyse hedef odur ve gerisi aractir. Referansin yol
    # boyunca biraktigi ara dosyalari check'e cevirmek "sen de tam bu gecici
    # dosyalari ayni adlarla uret" demek olur; uretilmis bir task'ta bu
    # sessizce cozumu referansin bicimine hapsediyordu.
    kapsam = set(overrides)

    def kapsamda(rel: str) -> bool:
        return not kapsam or rel in kapsam

    for rel in diff.created_dirs + diff.changed_dirs:
        if not kapsamda(rel):
            continue
        # is_dir gereksiz: dir_entries_eq dizin degilse zaten duser.
        checks.append({"op": "dir_entries_eq", "path": rel, "value": sorted(b_dirs[rel])})

    atlanan = 0
    for rel in diff.created_files + diff.modified_files:
        if not kapsamda(rel):
            atlanan += 1
            continue
        file_checks, file_notes, file_sorunlar = checks_for_file(
            rel, b_files[rel], overrides.get(rel), sandbox)
        checks.extend(file_checks)
        notes.extend(file_notes)
        sorunlar.extend(file_sorunlar)
    if atlanan:
        notes.append(f"{atlanan} ara dosya kapsam disi birakildi (outputs bildirilmis)")

    # Silmeler her zaman kapsamda: baslangic dunyasindan kaybolan bir dosya
    # ara dosya olamaz, gercek bir durum degisikligidir.
    for rel in diff.deleted:
        checks.append({"op": "not_exists", "path": rel})

    for rel in overrides:
        if rel not in b_files and rel not in b_dirs:
            sorunlar.append(f"{rel}: outputs: icinde bildirilmis ama referans uretmiyor")
        elif overrides[rel].get("kind") == "dir" and rel not in b_dirs:
            sorunlar.append(f"{rel}: kind=dir bildirildi ama dizin degil")

    return checks, notes, sorunlar


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
    # Bildirilen cikti turu ile gercegin uyusmadigi yerler: not degil,
    # task'i reddettiren sert uyusmazliklar.
    sorunlar: list[str] = field(default_factory=list)

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

            result = run_script(sandbox, task.reference_solution, timeout=60)
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
            checks, notes, sorunlar = build_checks(d, b[0], b[1], overrides, sandbox)

        # Turetilen check'ler iki fotografta da olculur: B'de hepsi gecmeli
        # (yoksa turetici kendi kendiyle celisiyor), A'da hepsi gecmemeli
        # (yoksa task baslangicta zaten cozulmus sayilir).
        ref = checks_report(run_checks(snap_b, checks, task.image))
        init = checks_report(run_checks(snap_a, checks, task.image))

    return DeriveReport(task.id, checks, notes,
                        ref["passed"], init["passed"], len(checks),
                        sorunlar=sorunlar)


# -- elle yazilanla karsilastirma -------------------------------------

def signature(check: dict) -> tuple:
    return (check.get("op"), check.get("path") or check.get("cmd", ""), check.get("field"))


# Ayni yol uzerinde soldaki op'u gereksiz kilan, ondan daha guclu op'lar.
# Elle yazilmis bir check'i turetici daha siki bir op'la kapsiyorsa bu bir
# eksik degil; rapor ikisini karistirmamali.
SUBSUMED_BY: dict[str, set[str]] = {
    "is_file": {"file_content_eq", "file_lines_eq", "file_line_count_eq",
                "file_contains", "file_matches", "json_field_eq", "json_eq",
                "run_stdout_eq"},
    "is_dir": {"dir_entries_eq"},
    "file_contains": {"file_content_eq", "file_lines_eq"},
    "file_matches": {"file_content_eq", "file_lines_eq"},
    "file_line_count_eq": {"file_content_eq", "file_lines_eq"},
    "json_field_eq": {"json_eq"},
}


def compare(derived: list[dict], handwritten: list[dict]) -> tuple[list, list, list, list]:
    """(ortak, daha-gucluyle-kapsanan, gercekten-kacan, ekstra)."""
    d = {signature(c) for c in derived}
    h = {signature(c) for c in handwritten}

    by_path: dict[str, set[str]] = {}
    for op, path, _ in d:
        by_path.setdefault(path, set()).add(op)
        if op == "run_stdout_eq":
            # Bir komut, adi gecen dosyalari da kapsar: `python3 topla.py`
            # topla.py yoksa zaten duser, ayrica is_file aramaya gerek yok.
            for token in path.split():
                by_path.setdefault(token, set()).add(op)

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

        for sorun in rep.sorunlar:
            print(f"  X {sorun}")
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
