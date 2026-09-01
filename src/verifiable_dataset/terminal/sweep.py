"""Run every task in a directory and report aggregate statistics.

    # saglik kontrolu: her task'in referansi hala geciyor mu (model gerekmez)
    python -m verifiable_dataset.terminal.sweep --reference

    # zorluk kalibrasyonu: her task'a N rollout
    python -m verifiable_dataset.terminal.sweep --model Qwen/Qwen3.5-9B \
        --base-url http://localhost:8000/v1 --rollouts 8 --out data/sweep.jsonl

Kalibrasyon modunda her task bir banda yerlestirilir. GRPO'da bir grubun
butun rollout'lari ayni reward'i alirsa advantage sifir olur, yani 0/N ve
N/N task'lar egitimde ayni sekilde olu agirliktir -- ikisi de isaretlenir.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from verifiable_dataset.terminal.llm import make_client, preflight, resolve_model
from verifiable_dataset.terminal.runner import Episode, run_model, run_reference
from verifiable_dataset.terminal.sandbox import docker_available
from verifiable_dataset.terminal.task import Task

# Ogrenme sinyali bu bandin disinda kaybolur.
BAND_LOW, BAND_HIGH = 0.125, 0.875


def discover(envs_dir: Path) -> list[Path]:
    """Find task directories, skipping private ones like _base."""
    return sorted(
        p.parent for p in envs_dir.glob("*/task.yaml")
        if not p.parent.name.startswith("_")
    )


def band_of(pass_rate: float, rollouts: int) -> str:
    if rollouts == 1:
        return "-"
    if pass_rate <= BAND_LOW:
        return "OLU-zor"
    if pass_rate >= BAND_HIGH:
        return "OLU-kolay"
    return "BANT"


def summarize(task_id: str, episodes: list[Episode], rollouts: int) -> dict:
    solved = sum(1 for e in episodes if e.solved)
    # Ikili odulun ortalamasi pass rate ile ayni sey oldugu icin ayrica
    # tutulmuyor; kismi oran ise teshis degeri tasiyor: kimsenin
    # cozemedigi ama herkesin yaklastigi bir task, kimsenin hicbir yere
    # varamadigindan cok farkli bir seydir.
    partials = [e.partial for e in episodes]
    turns = [e.turns for e in episodes if e.turns]
    errors = [e.error for e in episodes if e.error]
    pass_rate = solved / len(episodes) if episodes else 0.0
    return {
        "task_id": task_id,
        "rollouts": len(episodes),
        "solved": solved,
        "pass_rate": pass_rate,
        "mean_partial": statistics.mean(partials) if partials else 0.0,
        "mean_turns": statistics.mean(turns) if turns else 0.0,
        "band": band_of(pass_rate, rollouts),
        "errors": errors,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", default="envs", help="task dizinlerini iceren klasor")
    parser.add_argument("--reference", action="store_true",
                        help="model yerine referans cozumleri calistir (saglik kontrolu)")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="",
                        help="bos birakilirsa .env okunur")
    parser.add_argument("--protocol", choices=["auto", "native", "text"], default="auto")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--only", default="", help="sadece adinda bu gecen task'lari calistir")
    parser.add_argument("--out", default="", help="episode kayitlarini bu JSONL'e yaz")
    parser.add_argument("--istek-araligi", type=float, default=0.0,
                        help="iki API istegi arasinda en az bu kadar saniye bekle")
    parser.add_argument("--ayrintili", action="store_true",
                        help="her turu, modelin yorumunu ve komut ciktisini bas")
    args = parser.parse_args()

    ok, info = docker_available()
    if not ok:
        print(f"Docker daemon'a ulasilamiyor: {info}")
        return 1

    task_dirs = discover(Path(args.envs))
    if args.only:
        task_dirs = [d for d in task_dirs if args.only in d.name]
    if not task_dirs:
        print(f"{args.envs} altinda task bulunamadi")
        return 1

    client = None
    if not args.reference:
        args.model = resolve_model(args.model)
        if not args.model:
            print("--model verilmedi ve .env'de OPENAI_MODEL_NAME yok")
            return 2
        ok_uc, bilgi_uc = preflight(args.base_url, args.api_key, args.model)
        if not ok_uc:
            print(f"UC KONTROLU BASARISIZ: {bilgi_uc}")
            return 2
        print(f"uc dogrulandi: {bilgi_uc}")
        client = make_client(args.base_url, args.api_key, args.istek_araligi)

    mode = "referans" if args.reference else f"{args.model} x{args.rollouts}"
    print(f"{len(task_dirs)} task | mod: {mode} | docker {info}\n")

    rows: list[dict] = []
    all_episodes: list[Episode] = []
    for sira, task_dir in enumerate(task_dirs, start=1):
        try:
            task = Task.load(task_dir)
        except Exception as e:  # noqa: BLE001 - a broken yaml must not stop the sweep
            print(f"{task_dir.name:32} YUKLENEMEDI  {type(e).__name__}: {str(e)[:80]}")
            rows.append({"task_id": task_dir.name, "rollouts": 0, "solved": 0,
                         "pass_rate": 0.0, "mean_partial": 0.0, "mean_turns": 0.0,
                         "band": "BOZUK", "errors": [str(e)[:200]]})
            continue

        n = 1 if args.reference else args.rollouts
        episodes: list[Episode] = []
        if args.ayrintili:
            cizgi = "=" * 78
            print(f"\n{cizgi}")
            print(f"[{sira}/{len(task_dirs)}] {task.id}   "
                  f"max_turns={task.max_turns}  check={len(task.checks)}")
            print(cizgi)
            print(f"GOREV: {' '.join(task.prompt.split())[:400]}\n")
        for r in range(1, n + 1):
            if args.ayrintili:
                print(f"----- rollout {r}/{n} -----")
            try:
                ep = (run_reference(task, verbose=args.ayrintili) if args.reference
                      else run_model(task, client, args.model, args.protocol,
                                     verbose=args.ayrintili))
            except Exception as e:  # noqa: BLE001 - one bad rollout must not stop the sweep
                ep = Episode(task_id=task.id, solved=False, reward=0.0, turns=0,
                             grade={"passed": 0, "total": 0, "checks": []},
                             error=f"{type(e).__name__}: {e}")
            if args.ayrintili:
                durum = 'COZDU' if ep.solved else 'COZEMEDI'
                print(f"  => {durum}  check {ep.grade.get('passed', 0)}/"
                     f"{ep.grade.get('total', 0)}  kismi={ep.partial:.2f}  tur={ep.turns}")
                for c in ep.grade.get('checks', []):
                    if not c['ok']:
                        print(f"     x {c['name']}: {c['detail'][:150]}")
                if ep.error:
                    print(f"     ! {ep.error[:200]}")
                print()
            episodes.append(ep)

        all_episodes.extend(episodes)
        row = summarize(task.id, episodes, n)
        rows.append(row)

        mark = "OK  " if row["pass_rate"] == 1.0 else ("FAIL" if row["pass_rate"] == 0 else "kismi")
        band = "" if args.reference else f"  [{row['band']}]"
        print(f"{task.id:32} {mark} {row['solved']}/{row['rollouts']}  "
              f"kismi={row['mean_partial']:.2f}  tur={row['mean_turns']:.1f}{band}")
        for err in dict.fromkeys(row["errors"]):
            print(f"{'':34}! {err[:100]}")

    # -- ozet ---------------------------------------------------------
    print()
    if args.reference:
        gecen = sum(1 for r in rows if r["pass_rate"] == 1.0)
        print(f"SAGLIK: {gecen}/{len(rows)} task referansinda geciyor")
        bozuk = [r["task_id"] for r in rows if r["pass_rate"] < 1.0]
        if bozuk:
            print(f"Referansi gecmeyen task'lar dataset'e giremez: {', '.join(bozuk)}")
        return 0 if not bozuk else 1

    partials = [r["mean_partial"] for r in rows]
    cozulen = sum(1 for r in rows if r["pass_rate"] == 1.0)
    ortalama_pass = statistics.mean([r["pass_rate"] for r in rows])
    print(f"Cozulen         : {cozulen}/{len(rows)}")
    print(f"Ortalama reward : {ortalama_pass:.3f}  (ikili odul)")
    print(f"Ortalama kismi  : {statistics.mean(partials):.3f}  (teshis)")
    print(f"Ortalama tur    : {statistics.mean([r['mean_turns'] for r in rows]):.1f}")

    if args.rollouts < 4:
        # Tek rollout pass rate hakkinda neredeyse hicbir sey soylemez;
        # bant atamasi yapmak yaniltici olur.
        print(f"\nZorluk bandi olculmedi: kalibrasyon icin --rollouts 4 veya "
              f"daha fazlasi gerekir (su an {args.rollouts}).")
    else:
        for band in ("BANT", "OLU-kolay", "OLU-zor", "BOZUK"):
            ids = [r["task_id"] for r in rows if r["band"] == band]
            if ids:
                print(f"{band:10} {len(ids):3}  {', '.join(ids)}")

        kullanilir = sum(1 for r in rows if r["band"] == "BANT")
        print(f"\nEgitimde ise yarar: {kullanilir}/{len(rows)} "
              f"({kullanilir / len(rows) * 100:.0f}%)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"summary": row}, ensure_ascii=False) + "\n")
            for ep in all_episodes:
                f.write(json.dumps({"episode": ep.to_dict()}, ensure_ascii=False) + "\n")
        print(f"kayitlar -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
