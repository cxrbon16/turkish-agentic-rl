"""Write the Turkish task prompt -- the creative-writing stage.

    python -m verifiable_dataset.terminal.prompts --all --envs envs_gen \
        --model mistral-medium-latest

Bu asama en sona birakiliyor cunku check'ler o noktada donmus oluyor:
prompt yazari gorevin anlamini kaydiramaz, yalnizca nasil anlatildigini
secer. Modele referans cozum GOSTERILMIYOR -- gorseydi cozumu prompt'a
sizdirirdi ve gorev talimat izlemeye donerdi.

Uslup seed'den geliyor. tr-001'den tr-005'e kadar butun elle yazilmis
gorevler numarali emir listesi; o dagilim modele hedef ayristirmayi degil
talimat izlemeyi ogretir, bu yuzden uslup rastgele degil orneklenen bir
eksen.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from verifiable_dataset.terminal.llm import make_client, preflight, resolve_model
from verifiable_dataset.terminal.task import Task

TURKCE_HARFLER = set("çğıöşüÇĞİÖŞÜ")

# generate.py bos prompt'u boyle yaziyor; elle yazilmis dosyalarda ise
# zaten dolu bir blok var ve onlara dokunulmuyor.
BOS_PROMPT_RE = re.compile(r'^prompt:[ \t]*(?:""|\'\')?[ \t]*$', re.MULTILINE)

USLUP_TARIFI = {
    "numarali-emir": "Numarali adimlar halinde yaz (1), 2), 3)). Her adim tek bir is.",
    "duz-hedef": ("Duz paragraf halinde, hedefi anlat. Adim adim bolme -- ne "
                  "istendigini soyle, nasil yapilacagini kullaniciya birak."),
    "terse": ("Cok kisa tut: en fazla iki cumle. Nezaket cumlesi, giris, ozet "
              "isteme yok. Gercek bir kullanicinin acele yazdigi gibi."),
    "baglam-gomulu": ("Once iki-uc cumle alakasiz baglam ver (neden bu ise "
                      "ihtiyac duyuldugu, gunun nasil gectigi), asil istegi "
                      "onun icine yerlestir."),
    "kisit-sakli": ("Onemli bir kisiti (bir dosyanin korunmasi, bir bicim "
                    "sarti gibi) paragrafin ortasina, vurgusuz sekilde yerlestir."),
    "kabul-kriteri": ("Istegi yazdiktan sonra 'Su sartlar saglanmali:' diye "
                      "acik kabul kriterleri sirala."),
}

TALIMAT = """\
Sen bir kullanicisin ve Linux terminalinde calisan bir yardimciya is
veriyorsun. Sana gorevin AMACI ve baslangictaki dosyalar verilecek; senin
isin bunu dogal bir Turkce gorev metnine cevirmek.

Yalnizca gorev metnini yaz. Basliga, aciklamaya, tirnak isaretine, "Iste
prompt:" gibi girise yer yok.

KURALLAR:
1. Tam Turkce yaz: cogu, sunlari, calisma, olustur -- kisaltilmis ya da
   ASCII'lestirilmis Turkce degil, gercek harflerle (c-c, g-g, i-i, o-o,
   s-s, u-u).
2. Uretilecek her dosya ve dizini ADIYLA an. Amac hangi dosyalari
   istiyorsa prompt da onlari istemeli; anmadigin bir dosya icin yardimci
   sorumlu tutulamaz.
3. COZUMU VERME. Hangi komutun kullanilacagini, hangi bayragin gerektigini
   yazma. Ne istedigini soyle, nasil yapilacagini degil.
4. Baslangicta var olan dosyalarin adlarini dogru kullan.
5. Sayilari ve bicim sartlarini amacta yazdigi gibi koru -- "sadece sayi
   olsun", "her satirda bir tane" gibi kisitlar kaybolmamali.
"""


@dataclass
class PromptSonuc:
    task_id: str
    yazildi: bool = False
    deneme: int = 0
    prompt: str = ""
    sorunlar: list[str] = field(default_factory=list)
    hata: str = ""


def brief(task: Task) -> str:
    seed = task.metadata.get("seed", {})
    uslup = seed.get("uslup", "duz-hedef")
    yollar = [o.get("path", "") for o in task.outputs if o.get("path")]
    setup = task.setup.strip()
    if len(setup) > 2000:
        setup = setup[:2000] + "\n... (kisaltildi)"
    return f"""\
AMAC (gorev metnine cevrilecek):
{task.goal_tr.strip()}

URETILMESI GEREKEN DOSYA/DIZINLER (hepsi metinde adiyla gecmeli):
{', '.join(yollar) or '(bildirilmemis)'}

BASLANGIC DUNYASI (bu komutlar gorev baslamadan calistirildi; kullanici
bunlari gormez ama dosya adlari buradan):
{setup}

USLUP: {uslup}
{USLUP_TARIFI.get(uslup, USLUP_TARIFI['duz-hedef'])}

Calisma dizini /workspace. Gorev metnini yaz."""


def denetle(task: Task, prompt: str) -> list[str]:
    """Prompt'u deterministik olarak denetle.

    En onemlisi kapsam: check'ler bu noktada donmus durumda, yani prompt
    bir cikti dosyasini anmiyorsa gorev cozulemez hale gelir -- yardimci
    adini bilemedigi bir dosyayi uretemez.
    """
    sorunlar: list[str] = []
    metin = prompt.strip()

    if len(metin) < 40:
        sorunlar.append("prompt cok kisa")
    if not (TURKCE_HARFLER & set(metin)):
        sorunlar.append("hic Turkce karakter yok (ASCII'lestirilmis olabilir)")

    for o in task.outputs:
        yol = o.get("path", "")
        if not yol:
            continue
        ad = Path(yol).name
        if ad not in metin:
            sorunlar.append(f"'{ad}' prompt'ta gecmiyor")
            continue
        dizin = Path(yol).parent.as_posix()
        if dizin not in {"", "."} and dizin.split("/")[0] not in metin:
            sorunlar.append(f"'{dizin.split('/')[0]}' dizini prompt'ta gecmiyor")

    # Cozum sizintisi: referansin bir satiri prompt'ta aynen geciyorsa
    # gorev hedef ayristirma olmaktan cikip komut kopyalamaya doner.
    for satir in task.reference_solution.splitlines():
        satir = satir.strip()
        if len(satir) >= 20 and satir in metin:
            sorunlar.append(f"referans cozumden satir sizmis: {satir[:50]!r}")
            break

    return sorunlar


def prompt_yaz(task_dir: Path, prompt: str) -> bool:
    """task.yaml'daki bos prompt alanini blok skalariyle doldur."""
    yol = task_dir / "task.yaml"
    metin = yol.read_text(encoding="utf-8")
    if not BOS_PROMPT_RE.search(metin):
        return False
    govde = "\n".join(f"  {ln}" if ln.strip() else ""
                      for ln in prompt.strip().split("\n"))
    yeni = BOS_PROMPT_RE.sub(lambda _: f"prompt: |\n{govde}", metin, count=1)
    yol.write_text(yeni, encoding="utf-8")
    return True


def yaz_bir(client, model: str, task: Task, max_deneme: int = 3,
            verbose: bool = True) -> PromptSonuc:
    sonuc = PromptSonuc(task_id=task.id)
    mesajlar = [
        {"role": "system", "content": TALIMAT},
        {"role": "user", "content": brief(task)},
    ]

    for deneme in range(1, max_deneme + 1):
        sonuc.deneme = deneme
        try:
            yanit = client.chat.completions.create(
                model=model, messages=mesajlar, temperature=0.8, max_tokens=1024)
        except Exception as e:  # noqa: BLE001 - bir task kosuyu bitirmemeli
            sonuc.hata = f"model cagrisi basarisiz: {type(e).__name__}: {e}"
            return sonuc

        ham = (yanit.choices[0].message.content or "").strip()
        # Model bazen butun metni fence icine aliyor.
        ham = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", ham).strip()

        sorunlar = denetle(task, ham)
        if not sorunlar:
            sonuc.yazildi = True
            sonuc.prompt = ham
            return sonuc

        sonuc.sorunlar = sorunlar
        sonuc.prompt = ham
        if verbose:
            for s in sorunlar:
                print(f"    deneme {deneme}: {s}")
        if deneme == max_deneme:
            return sonuc

        mesajlar += [
            {"role": "assistant", "content": ham},
            {"role": "user", "content":
             "Metin su sorunlar yuzunden kabul edilmedi, duzelt ve YALNIZCA "
             "yeni gorev metnini yaz:\n" + "\n".join(f"- {s}" for s in sorunlar)},
        ]
    return sonuc


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--envs", default="envs_gen")
    parser.add_argument("--model", default="",
                        help="bos birakilirsa .env'deki OPENAI_MODEL_NAME")
    parser.add_argument("--base-url", default="",
                        help="bos birakilirsa .env'deki OPENAI_BASE_URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--denemeler", type=int, default=3)
    parser.add_argument("--gecikme", type=float, default=0.5)
    parser.add_argument("--istek-araligi", type=float, default=0.0,
                        help="iki API istegi arasinda en az bu kadar saniye bekle")
    parser.add_argument("--limit", type=int, default=0,
                        help="en fazla kac task islensin (0 = hepsi)")
    args = parser.parse_args()

    args.model = resolve_model(args.model)
    if not args.model:
        print("--model verilmedi ve .env'de OPENAI_MODEL_NAME yok")
        return 2

    if args.all:
        task_dirs = sorted(p.parent for p in Path(args.envs).glob("*/task.yaml")
                           if not p.parent.name.startswith("_"))
    elif args.task:
        task_dirs = [Path(args.task)]
    else:
        print("--task ya da --all ver")
        return 2

    bekleyen = []
    for d in task_dirs:
        try:
            t = Task.load(d)
        except Exception as e:  # noqa: BLE001
            print(f"{d.name:34} YUKLENEMEDI {type(e).__name__}: {str(e)[:60]}")
            continue
        if t.prompt.strip():
            continue
        bekleyen.append((d, t))
    if args.limit:
        bekleyen = bekleyen[:args.limit]

    if not bekleyen:
        print("prompt bekleyen task yok")
        return 0

    ok_uc, bilgi_uc = preflight(args.base_url, args.api_key, args.model)
    if not ok_uc:
        print(f"UC KONTROLU BASARISIZ: {bilgi_uc}")
        return 2
    print(f"uc dogrulandi: {bilgi_uc}")

    client = make_client(args.base_url, args.api_key, args.istek_araligi)

    print(f"model={args.model}  {len(bekleyen)} task prompt bekliyor\n")

    yazilan = 0
    sayac: dict[str, int] = {}
    for i, (d, task) in enumerate(bekleyen, start=1):
        uslup = task.metadata.get("seed", {}).get("uslup", "?")
        print(f"[{i}/{len(bekleyen)}] {task.id}  uslup={uslup}")
        sonuc = yaz_bir(client, args.model, task, args.denemeler)
        if sonuc.hata:
            print(f"    HATA {sonuc.hata}")
        elif sonuc.yazildi and prompt_yaz(d, sonuc.prompt):
            yazilan += 1
            ilk = sonuc.prompt.strip().splitlines()[0]
            print(f"    YAZILDI ({sonuc.deneme}. denemede)  {ilk[:88]}")
        elif sonuc.yazildi:
            print("    ATLANDI: task.yaml'da bos prompt alani bulunamadi")
        else:
            print(f"    BASARISIZ: {'; '.join(sonuc.sorunlar[:3])}")
            for s in sonuc.sorunlar:
                sayac[s.split(":")[0]] = sayac.get(s.split(":")[0], 0) + 1
        if args.gecikme and i < len(bekleyen):
            time.sleep(args.gecikme)

    print(f"\nYAZILAN: {yazilan}/{len(bekleyen)}")
    if sayac:
        print("En sik sorunlar: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(sayac.items(), key=lambda kv: -kv[1])[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
