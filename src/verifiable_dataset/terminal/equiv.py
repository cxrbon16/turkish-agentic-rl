"""Equivalence gate: correct-but-different solutions must still pass.

    python -m verifiable_dataset.terminal.equiv --all

Bir check seti iki yonden bozulabilir. Gevsek olani yanlis isi gecirir --
anti-hack tarafi, ayri bir kapi. Siki olani ise DOGRU isi duserer: model
`echo` yerine `printf` kullandigi, JSON anahtarlarini baska sirayla yazdigi
ya da ara betik olusturmadan cozdugu icin sifir alir. Bu sessiz bir hatadir;
task calisiyor gorunur, sadece ogrenme sinyali gurultuye doner.

Bu modul task.yaml'daki ``alt_solutions`` listesini calistirir ve turetilen
check'lerin hepsinde gecmesini bekler. Gecmeyen bir alternatif, check'in
bicime takildigi yeri gosterir.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from verifiable_dataset.terminal.checks import report as checks_report
from verifiable_dataset.terminal.checks import run_checks
from verifiable_dataset.terminal.derive import derive
from verifiable_dataset.terminal.sandbox import docker_available
from verifiable_dataset.terminal.task import Task


@dataclass
class AltResult:
    name: str
    exit_code: int = 0
    stderr: str = ""
    derived_passed: int = 0
    derived_total: int = 0
    hand_passed: int = 0
    hand_total: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.exit_code == 0 and self.derived_total > 0
                and self.derived_passed == self.derived_total)


def run_alt(task: Task, alt: dict, derived: list[dict]) -> AltResult:
    """Bir alternatif cozumu temiz bir dunyada calistirip check'lerle olc."""
    res = AltResult(name=alt.get("name", "(isimsiz)"))
    with tempfile.TemporaryDirectory(prefix="vds-equiv-") as tmp:
        with task.make_sandbox() as sandbox:
            task.prepare(sandbox)
            run = sandbox.exec(alt["script"], timeout=60)
            res.exit_code = run.exit_code
            res.stderr = run.stderr.strip()[:200]
            snapshot = sandbox.snapshot(Path(tmp) / "ws")

        rep = checks_report(run_checks(snapshot, derived, task.image))
        res.derived_passed, res.derived_total = rep["passed"], rep["total"]
        res.failures = [f"{c['name']}: {c['detail'][:110]}"
                        for c in rep["checks"] if not c["ok"]]

        if task.checks:
            hand = checks_report(run_checks(snapshot, task.checks, task.image))
            res.hand_passed, res.hand_total = hand["passed"], hand["total"]
    return res


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="", help="tek task dizini")
    parser.add_argument("--all", action="store_true", help="envs/ altindaki her task")
    parser.add_argument("--envs", default="envs")
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

    n_alts = n_ok = n_hand_fail = 0
    tasks_without_alts: list[str] = []

    for task_dir in task_dirs:
        task = Task.load(task_dir)
        print(task.id)

        if not task.alt_solutions:
            print("  (alt_solutions tanimli degil -- denklik olculemiyor)\n")
            tasks_without_alts.append(task.id)
            continue

        rep = derive(task)
        if rep.error:
            print(f"  HATA turetme basarisiz: {rep.error}\n")
            continue
        print(f"  referanstan turetilen: {rep.total} check")

        for alt in task.alt_solutions:
            r = run_alt(task, alt, rep.checks)
            n_alts += 1
            n_ok += r.ok
            mark = "GECTI " if r.ok else "DUSTU "
            hand = (f"   elle {r.hand_passed}/{r.hand_total}"
                    if r.hand_total else "")
            print(f"  {mark} {r.derived_passed}/{r.derived_total}{hand}   {r.name}")
            if r.exit_code != 0:
                print(f"         ! cozum exit={r.exit_code}: {r.stderr}")
            for f in r.failures:
                print(f"         x {f}")
            if r.hand_total and r.hand_passed < r.hand_total:
                # Turetilen gecip elle yazilan duserse siki olan elle yazilandir.
                n_hand_fail += 1
                print(f"         ~ elle yazilan check seti bu cozumu gecirmiyor")
        print()

    print(f"DENKLIK: {n_ok}/{n_alts} alternatif cozum turetilen check'leri geciyor")
    if n_hand_fail:
        print(f"Elle yazilan check setinin duserdigi alternatif sayisi: {n_hand_fail}")
    if tasks_without_alts:
        print(f"alt_solutions yok: {', '.join(tasks_without_alts)}")
    return 0 if n_alts and n_ok == n_alts else 1


if __name__ == "__main__":
    raise SystemExit(main())
