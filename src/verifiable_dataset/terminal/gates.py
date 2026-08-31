"""The LLM-free gauntlet a task must survive before it enters the dataset.

    python -m verifiable_dataset.terminal.gates --all

Kapilarin hepsi Docker'la calisir, hicbiri model cagirmaz. Uretimde once
bunlar kosar: model cagiran pahali kapilara (cozulebilirlik, bant
kalibrasyonu) yalnizca buradan sag cikan adaylar ulasir.

Elle yazilmis task'lar da birer spec oldugu icin gauntlet onlara karsi
dogrulanabiliyor: 8'i de gecmeliyse, gecmeyen bir sey varsa hata kapida,
uretici modelde degil.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from verifiable_dataset.terminal.derive import derive
from verifiable_dataset.terminal.equiv import run_alt
from verifiable_dataset.terminal.sandbox import docker_available
from verifiable_dataset.terminal.seeds import tools_used, uses_tool
from verifiable_dataset.terminal.task import Task

TURKCE_HARFLER = set("çğıöşüÇĞİÖŞÜ")


@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False

    @property
    def mark(self) -> str:
        if self.skipped:
            return "ATLA"
        return "OK  " if self.ok else "DUSTU"


def _norm(checks: list[dict]) -> str:
    return json.dumps(sorted((json.dumps(c, sort_keys=True, ensure_ascii=False)
                              for c in checks)), ensure_ascii=False)


# -- kapilar ----------------------------------------------------------

def gate_spec_complete(task: Task, spec_only: bool = False) -> GateResult:
    eksik = []
    if not task.reference_solution.strip():
        eksik.append("reference_solution")
    if not task.goal_tr.strip():
        eksik.append("goal_tr")
    # spec asamasinda prompt heniz yazilmadi; creative writing ayri asama.
    if not spec_only and not task.prompt.strip():
        eksik.append("prompt")
    return GateResult("spec butunlugu", not eksik,
                      f"eksik alanlar: {', '.join(eksik)}" if eksik else "")


def gate_data_ascii(task: Task) -> GateResult:
    """Veri ASCII, prompt tam Turkce -- proje karari.

    Dunyanin icerigi ASCII kalirsa siralama ve kodlama surprizleri
    task'a degil, bilincli olarak secilmis bir aileye ait olur.
    """
    kirli = [ch for ch in task.setup if ord(ch) > 127]
    return GateResult("veri ASCII", not kirli,
                      f"setup icinde ASCII disi karakter: {sorted(set(kirli))[:8]}")


def gate_prompt_turkce(task: Task) -> GateResult:
    if task.metadata.get("language") != "tr":
        return GateResult("prompt turkce", True, skipped=True, detail="tr degil")
    var = TURKCE_HARFLER & set(task.prompt)
    return GateResult("prompt turkce", bool(var),
                      "prompt'ta hic Turkce karakter yok -- ASCII'lestirilmis olabilir")


def gate_tool_conformance(task: Task) -> GateResult:
    """Seed'deki araclar referansta gercekten kullanilmis mi.

    Bu kapi olmazsa model her task'i tek satir python3 ile cozer: grid dolu
    gorunur, cesitlilik sahte olur.
    """
    seed = task.metadata.get("seed")
    if not seed:
        return GateResult("arac uygunlugu", True, skipped=True, detail="seed yok")
    istenen = seed.get("tools", [])
    eksik = [t for t in istenen if not uses_tool(task.reference_solution, t)]
    kacak = (uses_tool(task.reference_solution, "python3")
             and "python3" not in istenen)
    sorun = []
    if eksik:
        sorun.append(f"kullanilmayan seed araclari: {eksik}")
    if kacak:
        sorun.append("seed'de yokken python3'e kacilmis")
    return GateResult("arac uygunlugu", not sorun, "; ".join(sorun))


def gate_determinism(task: Task) -> tuple[GateResult, "object | None"]:
    """Ayni referans iki temiz dunyada ayni check'leri uretmeli.

    Timestamp, $RANDOM, hash sirasi ve locale farklari burada yakalanir --
    uretilmis task'larda elle yazilanlardan cok daha sik sizarlar.
    """
    first = derive(task)
    if first.error:
        return GateResult("determinizm", False, f"turetme basarisiz: {first.error}"), None
    second = derive(task)
    if second.error:
        return GateResult("determinizm", False, f"ikinci kosu: {second.error}"), None
    if _norm(first.checks) != _norm(second.checks):
        a = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in first.checks}
        b = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in second.checks}
        fark = sorted(a ^ b)[:3]
        return GateResult("determinizm", False,
                          f"iki kosu farkli check uretti: {fark}"), None
    return GateResult("determinizm", True,
                      f"iki kosuda ayni {first.total} check"), first


def gate_discriminates(task: Task, rep) -> GateResult:
    if rep is None:
        return GateResult("ayirt etme", False, "turetme yapilamadi")
    ok = rep.ref_passed == rep.total and rep.init_passed < rep.total
    return GateResult("ayirt etme", ok,
                      f"B {rep.ref_passed}/{rep.total}, A {rep.init_passed}/{rep.total}")


def gate_output_kinds(task: Task, rep) -> GateResult:
    """Bildirilen cikti turu, referansin gercekten urettigiyle uyusmali.

    Spec `kind: json` deyip gecersiz JSON uretebiliyor. Bu sessizce byte
    esitligine dusuyor ve task tersine doner: gecerli JSON yazan DOGRU
    cozum sifir alir, cunku beklenen deger bozuk metnin kendisidir.
    """
    if rep is None:
        return GateResult("cikti turu", False, "turetme yapilamadi")
    if not task.outputs:
        return GateResult("cikti turu", True, skipped=True,
                          detail="outputs bildirilmemis")
    return GateResult("cikti turu", not rep.sorunlar, "; ".join(rep.sorunlar[:3]))


def ucuz_hile_betigi(checks: list[dict]) -> str:
    """Check'lerde adi gecen her yolu bos olarak yaratan tembel bir betik.

    Hicbir is yapmiyor: dizinleri aciyor, dosyalari bos olarak dokunuyor,
    silinmesi gerekenleri siliyor. Bu betik check'leri geciyorsa task'in
    olcecek bir seyi yok demektir.
    """
    satirlar: list[str] = []
    for c in checks:
        op, path = c.get("op"), c.get("path")
        if not path:
            continue
        q = shlex.quote(path)
        if op in {"is_dir", "dir_entries_eq"}:
            satirlar.append(f"mkdir -p {q}")
        elif op == "not_exists":
            satirlar.append(f"rm -rf {q}")
        else:
            satirlar.append(f'mkdir -p "$(dirname {q})" && : > {q}')
    return "\n".join(dict.fromkeys(satirlar)) or "true"


def gate_cheap_hack(task: Task, rep) -> GateResult:
    """Hicbir is yapmayan bir cozum gecmemeli.

    Uretilen bir task'in referansi bos bir dosya birakmisti; turetilen check
    de "icerik bos olsun" oldu ve `touch sonuc.txt` tam puan aldi. Bant
    filtresi bunu ancak 16 ajan rollout'u harcayarak fark ederdi. Burada
    tek konteynerde, saniyeler icinde anlasiliyor.
    """
    if rep is None or not rep.checks:
        return GateResult("ucuz hile", False, "turetilen check yok")
    r = run_alt(task, {"name": "ucuz hile", "script": ucuz_hile_betigi(rep.checks)},
                rep.checks)
    return GateResult(
        "ucuz hile", not r.ok,
        f"is yapmayan betik {r.derived_passed}/{r.derived_total} check geciyor -- "
        f"check'lerin olctugu bir sey yok",
    )


def gate_equivalence(task: Task, checks: list[dict] | None) -> GateResult:
    if not task.alt_solutions:
        return GateResult("denklik", True, skipped=True, detail="alt_solutions yok")
    if checks is None:
        return GateResult("denklik", False, "turetilen check yok")
    dusen = []
    for alt in task.alt_solutions:
        r = run_alt(task, alt, checks)
        if not r.ok:
            dusen.append(f"{r.name} ({r.derived_passed}/{r.derived_total})")
    return GateResult("denklik", not dusen,
                      f"{len(task.alt_solutions) - len(dusen)}/{len(task.alt_solutions)} "
                      f"alternatif geciyor" + (f"; dusen: {dusen}" if dusen else ""))


def run_gauntlet(task: Task, spec_only: bool = False) -> list[GateResult]:
    """spec_only: prompt heniz yazilmamis adaylar icin prompt kapilarini atla."""
    results = [
        gate_spec_complete(task, spec_only),
        gate_data_ascii(task),
        gate_tool_conformance(task),
    ]
    if not spec_only:
        results.insert(2, gate_prompt_turkce(task))
    det, rep = gate_determinism(task)
    results.append(det)
    results.append(gate_discriminates(task, rep))
    results.append(gate_output_kinds(task, rep))
    results.append(gate_cheap_hack(task, rep))
    results.append(gate_equivalence(task, rep.checks if rep else None))
    return results


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--envs", default="envs")
    parser.add_argument("--spec-only", action="store_true",
                        help="prompt heniz yazilmamis adaylar icin prompt kapilarini atla")
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

    gecen = 0
    for task_dir in task_dirs:
        task = Task.load(task_dir)
        print(task.id)
        results = run_gauntlet(task, spec_only=args.spec_only)
        for r in results:
            # Detay yalnizca aciklayici oldugu yerde: gecen bir kapinin
            # hata mesajini basmak raporu okunmaz hale getiriyordu.
            detail = f"   {r.detail}" if r.detail and (r.skipped or not r.ok) else ""
            print(f"  [{r.mark}] {r.name:18}{detail}")
        if all(r.ok for r in results):
            gecen += 1
        else:
            dusen = [r.name for r in results if not r.ok]
            print(f"  --> REDDEDILDI: {', '.join(dusen)}")
        print()

    print(f"GAUNTLET: {gecen}/{len(task_dirs)} task butun kapilardan geciyor")
    return 0 if gecen == len(task_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
