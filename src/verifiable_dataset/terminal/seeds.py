"""The seed grid a generated task is sampled from.

    python -m verifiable_dataset.terminal.seeds --n 12
    python -m verifiable_dataset.terminal.seeds --dagilim 400

Bir LLM'e "ilginc bir terminal gorevi uret" demek her seferinde ayni yirmi
fikri getirir: logda hata say, dosya adi degistir, sayilari topla. Cesitlilik
yaraticiliktan degil, ornekleme uzayindan gelir -- bu yuzden seed serbest
metin degil, bir gridden cekilen kucuk bir kayit.

Eksenler iki turlu. Bazilari yalnizca isimleri degistirir (konu); bazilari
ajanin ne yapmak zorunda kaldigini degistirir (kesif, kaynak, uslup). Grid'i
buyutmenin degeri ikincilerde: 12.000 kombinasyondan birkac yuz task
uretecegiz, yani sayi zaten kisit degil -- tur kisit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field

# -- eksen 1: araclar -------------------------------------------------
# Referansta gecmesi beklenen arac demetleri. Ogretmek istedigimiz yetenek
# yuzeyini dogrudan surer: seed ne isterse cozum onunla yazilmak zorunda.
# shuf bilerek yok -- rastgelelik determinizm kapisini duserirdi.
TOOL_BUNDLES: dict[str, list[str]] = {
    "say": ["grep", "wc"],
    "sirala": ["sort", "uniq"],
    "bul-tasi": ["find", "xargs"],
    "metin-duzenle": ["sed", "tr"],
    "sutun": ["cut", "paste"],
    "awk": ["awk"],
    "arsiv": ["tar", "gzip"],
    "karsilastir": ["diff", "comm"],
    "python-veri": ["python3"],
    "dosya-duzen": ["mkdir", "mv"],
    "birlestir": ["join", "sort"],
    "kume-farki": ["comm", "sort"],
    "satir-araligi": ["head", "tail"],
    "cok-dosya-ara": ["xargs", "grep"],
    "secerek-arsivle": ["tar", "find"],
    "dallandir": ["tee", "cut"],
    "ters-cevir": ["rev", "cut"],
    "uret": ["printf", "seq"],
    "yol-parcala": ["basename", "dirname"],
    "dizin-istatistik": ["wc", "find"],
}

# -- eksen 2: konu ----------------------------------------------------
# Yalnizca isimleri degistirir ama onsuz her dunya "sunucu logu" oluyor.
KONULAR = [
    "sunucu-loglari", "ogrenci-notlari", "kargo-takibi", "envanter",
    "muhasebe-kalemleri", "kutuphane-kayitlari", "restoran-siparisleri",
    "sensor-olcumleri", "personel-vardiyalari", "urun-yorumlari",
    "hastane-randevulari", "hava-durumu-kayitlari",
]

# -- eksen 3: hedef ---------------------------------------------------
HEDEFLER = [
    "donustur", "ozetle", "ayikla", "onar", "denetle", "yeniden-duzenle",
    "birlestir", "bol", "karsilastir", "dogrula", "grupla-say", "sirala-sec",
    "toplu-yeniden-adlandir", "arsivle",
]

# Hedef, arac demetinden bagimsiz orneklenirse `printf+seq` ile "arsivle"
# gibi birbirini tutmayan seed'ler cikiyor. Bunlar spec-writer'i zorlar ve
# sonunda arac uygunlugu kapisinda bosuna elenir -- yani bosa LLM cagrisi.
# Bu yuzden hedef, demetin makul sekilde hizmet edebilecegi islerden secilir.
BUNDLE_HEDEFLER: dict[str, list[str]] = {
    "say": ["ozetle", "denetle", "grupla-say", "dogrula", "ayikla"],
    "sirala": ["ozetle", "grupla-say", "sirala-sec", "denetle", "donustur"],
    "bul-tasi": ["yeniden-duzenle", "toplu-yeniden-adlandir", "ayikla", "denetle"],
    "metin-duzenle": ["donustur", "onar", "ayikla", "dogrula"],
    "sutun": ["donustur", "ayikla", "birlestir", "ozetle"],
    "awk": ["ozetle", "grupla-say", "donustur", "dogrula", "denetle", "sirala-sec"],
    "arsiv": ["arsivle", "bol", "donustur"],
    "karsilastir": ["karsilastir", "dogrula", "denetle"],
    "python-veri": ["ozetle", "grupla-say", "onar", "dogrula", "donustur", "denetle"],
    "dosya-duzen": ["yeniden-duzenle", "bol", "toplu-yeniden-adlandir"],
    "birlestir": ["birlestir", "donustur", "ozetle"],
    "kume-farki": ["karsilastir", "ayikla", "denetle"],
    "satir-araligi": ["ayikla", "bol", "ozetle"],
    "cok-dosya-ara": ["ayikla", "denetle", "ozetle", "grupla-say"],
    "secerek-arsivle": ["arsivle", "yeniden-duzenle"],
    "dallandir": ["donustur", "bol", "ayikla"],
    "ters-cevir": ["donustur", "ayikla"],
    "uret": ["donustur", "bol", "yeniden-duzenle"],
    "yol-parcala": ["yeniden-duzenle", "toplu-yeniden-adlandir", "denetle"],
    "dizin-istatistik": ["ozetle", "denetle", "grupla-say"],
}

assert set(BUNDLE_HEDEFLER) == set(TOOL_BUNDLES), "her demetin hedef listesi olmali"
assert all(set(v) <= set(HEDEFLER) for v in BUNDLE_HEDEFLER.values()), \
    "tanimsiz hedef"

# -- eksen 4: girdi bicimi --------------------------------------------
GIRDILER = ["duz-metin", "csv", "tsv", "json", "log-satiri", "ini", "sabit-genislik"]

# -- eksen 5: kaynak sekli --------------------------------------------
# Cok dosyali capraz referans burada: tek dosya task'larindan yapisal
# olarak farkli, join/comm/diff demetlerine gercek is cikariyor.
KAYNAKLAR = {
    "tek-dosya": 50,
    "iki-dosya-capraz": 18,
    "cok-dosya": 16,
    "dizin-agaci": 12,
    "tarball": 8,
}

# -- eksen 6: cikti bicimi --------------------------------------------
# Check op dagilimini dengeliyor: su an file_content_eq ve json_field_eq
# agir basiyor, dir_entries_eq ve not_exists az uretiliyor.
CIKTILAR = {
    "tek-sayi": 18,
    "satir-listesi": 22,
    "csv": 14,
    "json": 16,
    "dizin-yapisi": 12,
    "birden-cok-dosya": 10,
    "yeniden-adlandirilmis": 8,
}

# -- eksen 7: kesif gereksinimi ---------------------------------------
# Bugunku 8 task'in hicbirinde yok: model ilk turda ne yapacagini biliyor.
# Bu eksen bakmayi zorunlu kiliyor, yani cok turlu davranisi.
KESIFLER = {
    "yok": 50,
    "dosyayi-bul": 20,      # "en buyuk .log dosyasini ozetle"
    "veriye-bak": 20,       # kolon adlari prompt'ta yok
    "dolayli-hedef": 10,    # cikti adi bir ayar dosyasinin icinde
}

# -- eksen 8: spesifikasyon uslubu ------------------------------------
# Prompt yazimina gider ama burada orneklenir ki keyfe kalmasin.
# tr-001..005'in hepsi numarali emir listesi; o dagilim modele hedef
# ayristirmayi degil talimat izlemeyi ogretir.
USLUPLAR = {
    "numarali-emir": 24,
    "duz-hedef": 28,
    "terse": 14,
    "baglam-gomulu": 14,
    "kisit-sakli": 10,
    "kabul-kriteri": 10,
}

# -- eksen 9: ortam direnci -------------------------------------------
# Izin/chmod tabanli olanlar yok: container'da root'uz, read-only bir
# dosya root icin engel degil, yani task sahte olurdu.
DIRENCLER = {
    "yok": 55,
    "bosluklu-ad": 9,
    "crlf": 9,
    "son-newline-yok": 7,
    "bom": 5,
    "bos-dosya": 6,
    "tekrar-eden-kayit": 5,
    "baslik-yok": 4,
}

# -- eksen 10: bukulme ------------------------------------------------
BUKULMELER = {"yok": 45, "kirli-veri": 25, "tuzak-dosya": 20, "olcek": 10}

# Yol ve dizin uzerinde calisan demetler tek dosyalik bir dunyada anlamsiz
# kaliyor: `basename`/`dirname` ile tek bir CSV'yi "denetlemek" gorev degil.
COK_DOSYA_ISTEYEN = {
    "bul-tasi", "dosya-duzen", "yol-parcala", "secerek-arsivle",
    "dizin-istatistik", "cok-dosya-ara", "karsilastir", "kume-farki",
    "birlestir",
}

# -- imaj katmanlari --------------------------------------------------
# Base imaj envanteri olculdu: coreutils, grep, sed, find, xargs, awk
# (mawk), tar, gzip, diff, comm, paste, join, seq, tee ve python3 zaten
# var. Eksik olanlar ayri bir katman gerektiriyor.
IMAGE_TIERS: list[tuple[str, set[str]]] = [
    ("verifiable-dataset/base:latest", {
        "bash", "sh", "grep", "sed", "find", "xargs", "sort", "uniq", "cut",
        "tr", "wc", "head", "tail", "awk", "tar", "gzip", "diff", "comm",
        "paste", "join", "install", "rev", "python3", "mkdir", "mv", "cp",
        "rm", "cat", "echo", "printf", "touch", "ls", "basename", "dirname",
        "seq", "tee", "nl", "split", "od",
    }),
    ("verifiable-dataset/text:latest", {"gawk", "jq"}),
]

ALL_TOOLS = sorted({t for _, s in IMAGE_TIERS for t in s})


def tools_of_image(image: str) -> list[str]:
    """Bir imajda kurulu araclarin tamami (kumulatif katmanlar).

    Spec yazan modele veriliyor: yoksa imajda olmayan araclara uzaniyor
    (jshon, jq gibi) ve aday bosuna bir onarim turu harciyor.
    """
    available: set[str] = set()
    for name, tier in IMAGE_TIERS:
        available |= tier
        if name == image:
            return sorted(available)
    return sorted(available)


def image_for(tools: list[str]) -> str:
    """Istenen araclarin hepsini barindiran en dusuk katmani sec."""
    available: set[str] = set()
    for image, tier in IMAGE_TIERS:
        available |= tier
        if set(tools) <= available:
            return image
    eksik = sorted(set(tools) - available)
    raise ValueError(f"hicbir imaj su araclari saglamiyor: {eksik}")


@dataclass
class Seed:
    id: str
    bundle: str
    tools: list[str]
    konu: str
    hedef: str
    girdi: str
    kaynak: str
    cikti: str
    kesif: str
    uslup: str
    direnc: str
    bukulme: str
    adim: int
    image: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# -- ornekleme --------------------------------------------------------

def _pick(rng: random.Random, weighted: dict[str, int]) -> str:
    return rng.choices(list(weighted), weights=list(weighted.values()), k=1)[0]


def is_compatible(seed: Seed) -> tuple[bool, str]:
    """Anlamsiz kombinasyonlari ele.

    Her ekseni bagimsiz ornekleyip LLM'e vermek brief'i sisirir ve kaliteyi
    duserir: "kargo takibi + tar/gzip + onar + CRLF + kesif + terse + 5 adim"
    diye bir seed'den tutarli task cikmaz.
    """
    if seed.bundle in COK_DOSYA_ISTEYEN and seed.kaynak == "tek-dosya":
        return False, f"{seed.bundle} demeti tek dosyada anlamsiz kaliyor"
    if seed.kesif != "yok" and seed.adim < 2:
        return False, "kesif en az iki adim ister (once bak, sonra yap)"
    if seed.direnc != "yok" and seed.bukulme == "kirli-veri":
        return False, "direnc ve kirli-veri ayni yarayi iki kez aciyor"
    if seed.kaynak == "tarball" and not ({"tar", "gzip"} & set(seed.tools)):
        return False, "tarball kaynagi tar/gzip demeti ister"
    if seed.hedef == "toplu-yeniden-adlandir" and seed.kaynak == "tek-dosya":
        return False, "toplu yeniden adlandirma birden cok dosya ister"
    if seed.hedef in {"karsilastir", "birlestir"} and seed.kaynak == "tek-dosya":
        return False, f"{seed.hedef} en az iki kaynak ister"
    if seed.hedef == "arsivle" and seed.cikti in {"tek-sayi", "csv", "json"}:
        return False, "arsivleme ciktisi dosya/dizin olmali"
    if seed.hedef == "arsivle" and seed.kaynak == "tek-dosya":
        return False, "tek dosyayi arsivlemek anlamli bir gorev degil"
    if seed.cikti == "yeniden-adlandirilmis" and seed.kaynak == "tek-dosya":
        return False, "yeniden adlandirma ciktisi birden cok dosya ister"
    return True, ""


def sample(rng: random.Random, index: int, max_tries: int = 60) -> Seed:
    """Uyumluluk suzgecinden gecen bir seed cek."""
    for _ in range(max_tries):
        bundle = rng.choice(list(TOOL_BUNDLES))
        tools = TOOL_BUNDLES[bundle]
        seed = Seed(
            id=f"s-{index:04d}",
            bundle=bundle,
            tools=list(tools),
            konu=rng.choice(KONULAR),
            hedef=rng.choice(BUNDLE_HEDEFLER[bundle]),
            girdi=rng.choice(GIRDILER),
            kaynak=_pick(rng, KAYNAKLAR),
            cikti=_pick(rng, CIKTILAR),
            kesif=_pick(rng, KESIFLER),
            uslup=_pick(rng, USLUPLAR),
            direnc=_pick(rng, DIRENCLER),
            bukulme=_pick(rng, BUKULMELER),
            adim=rng.randint(1, 5),
            image=image_for(tools),
        )
        ok, _ = is_compatible(seed)
        if ok:
            return seed
    raise RuntimeError(f"{max_tries} denemede uyumlu seed cikmadi")


def sample_many(n: int, rng_seed: int = 0) -> list[Seed]:
    rng = random.Random(rng_seed)
    return [sample(rng, i) for i in range(n)]


# -- arac tespiti -----------------------------------------------------

def uses_tool(script: str, tool: str) -> bool:
    """Bir aracin betikte komut konumunda gecip gecmedigine bak.

    Duz substring aramasi yaniltir: dosya adinda ya da yorumda gecen bir
    kelime araci kullanmak sayilmaz. Komut basi, boru, ; ve `-exec` ile
    `xargs` sonrasi konumlar araniyor.
    """
    pattern = (rf"(?:^|[\n;|&(`]|\$\(|\bxargs\s+|-exec\s+|\bthen\s+|\bdo\s+)"
               rf"\s*{re.escape(tool)}\b")
    return re.search(pattern, script, re.MULTILINE) is not None


def tools_used(script: str, vocabulary: list[str] | None = None) -> list[str]:
    vocab = vocabulary or ALL_TOOLS
    return sorted(t for t in vocab if uses_tool(script, t))


# -- parmak izi -------------------------------------------------------

def fingerprint(reference_solution: str, checks: list[dict],
                seed: Seed | dict | None = None) -> tuple[str, dict]:
    """Task'in yapisal kimligi.

    Cesitliligi "yaratici ol" diyerek degil sayarak zorluyoruz: ayni parmak
    izinden belli bir sayidan fazlasi kabul edilmez.

    Konu bilerek disarida: ayni isi kargo yerine envanter uzerinde yapmak
    yeni bir yapi degil. Parmak izini konuyla ayirt etseydik dedup neredeyse
    hic tetiklenmezdi.
    """
    s = seed.to_dict() if isinstance(seed, Seed) else (seed or {})
    parts = {
        "tools": tools_used(reference_solution),
        "ops": sorted({c.get("op", "") for c in checks}),
        "hedef": s.get("hedef", ""),
        "kaynak": s.get("kaynak", ""),
        "kesif": s.get("kesif", ""),
        "cikti": s.get("cikti", ""),
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12], parts


# -- CLI --------------------------------------------------------------

def _grid_size() -> int:
    return (len(TOOL_BUNDLES) * len(KONULAR) * len(HEDEFLER) * len(GIRDILER)
            * len(KAYNAKLAR) * len(CIKTILAR) * len(KESIFLER) * len(USLUPLAR)
            * len(DIRENCLER) * len(BUKULMELER) * 5)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="ornek seed sayisi")
    parser.add_argument("--rng", type=int, default=0, help="tekrarurutilebilirlik icin")
    parser.add_argument("--dagilim", type=int, default=0,
                        help="N seed cekip eksen dagilimini ozetler")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.dagilim:
        seeds = sample_many(args.dagilim, args.rng)
        print(f"{args.dagilim} seed cekildi\n")
        for eksen in ("bundle", "hedef", "kaynak", "cikti", "kesif", "uslup",
                      "direnc", "bukulme"):
            sayim = Counter(getattr(s, eksen) for s in seeds)
            satir = "  ".join(f"{k}={v}" for k, v in sayim.most_common())
            print(f"{eksen:10} {satir}")
        return 0

    seeds = sample_many(args.n, args.rng)
    if args.json:
        print(json.dumps([s.to_dict() for s in seeds], ensure_ascii=False, indent=2))
        return 0

    print(f"grid buyuklugu: {_grid_size():,} kombinasyon "
          f"(uyumluluk suzgeci bunun bir kismini eliyor)\n")
    for s in seeds:
        print(f"{s.id}  {'+'.join(s.tools):14} {s.konu:20} {s.hedef:22} "
              f"{s.kaynak:17} {s.cikti:22}")
        print(f"{'':8}  girdi={s.girdi:15} kesif={s.kesif:14} uslup={s.uslup:14} "
              f"direnc={s.direnc:18} bukulme={s.bukulme:11} adim={s.adim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
