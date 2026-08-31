"""The seed grid a generated task is sampled from.

    python -m verifiable_dataset.terminal.seeds --n 12

Bir LLM'e "ilginc bir terminal gorevi uret" demek her seferinde ayni yirmi
fikri getirir: logda hata say, dosya adi degistir, sayilari topla. Cesitlilik
yaraticiliktan degil, ornekleme uzayindan gelir -- bu yuzden seed serbest
metin degil, bir gridden cekilen kucuk bir kayit.

Konu ekseni ozellikle onemli: onsuz butun dunyalar "sunucu logu" oluyor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field

# Referansta gecmesi beklenen arac demetleri. Ogretmek istedigimiz yetenek
# yuzeyini dogrudan surer: seed ne isterse cozum onunla yazilmak zorunda.
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
}

KONULAR = [
    "sunucu-loglari", "ogrenci-notlari", "kargo-takibi", "envanter",
    "muhasebe-kalemleri", "kutuphane-kayitlari", "restoran-siparisleri",
    "sensor-olcumleri", "personel-vardiyalari", "urun-yorumlari",
]

HEDEFLER = ["donustur", "ozetle", "ayikla", "onar", "denetle", "yeniden-duzenle"]

BUKULMELER = ["yok", "yok", "kirli-veri", "tuzak-dosya", "olcek"]

# Katman basina kurulu araclar. Base imaj envanteri olculdu: coreutils,
# grep, sed, find, xargs, awk (mawk), tar, gzip, diff, comm, paste, join
# ve python3 zaten var. Eksik olanlar ayri bir katman gerektiriyor.
IMAGE_TIERS: list[tuple[str, set[str]]] = [
    ("verifiable-dataset/base:latest", {
        "bash", "sh", "grep", "sed", "find", "xargs", "sort", "uniq", "cut",
        "tr", "wc", "head", "tail", "awk", "tar", "gzip", "diff", "comm",
        "paste", "join", "install", "rev", "python3", "mkdir", "mv", "cp",
        "rm", "cat", "echo", "printf", "touch", "ls", "basename", "dirname",
    }),
    ("verifiable-dataset/text:latest", {"gawk", "jq"}),
]

ALL_TOOLS = sorted({t for _, s in IMAGE_TIERS for t in s})


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
    bukulme: str
    adim: int
    image: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def sample(rng: random.Random, index: int) -> Seed:
    bundle = rng.choice(list(TOOL_BUNDLES))
    tools = TOOL_BUNDLES[bundle]
    return Seed(
        id=f"s-{index:04d}",
        bundle=bundle,
        tools=list(tools),
        konu=rng.choice(KONULAR),
        hedef=rng.choice(HEDEFLER),
        bukulme=rng.choice(BUKULMELER),
        adim=rng.randint(1, 5),
        image=image_for(tools),
    )


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
                hedef: str = "", konu: str = "") -> tuple[str, dict]:
    """Task'in yapisal kimligi.

    Cesitliligi "yaratici ol" diyerek degil sayarak zorluyoruz: ayni parmak
    izinden belli bir sayidan fazlasi kabul edilmez.
    """
    parts = {
        "tools": tools_used(reference_solution),
        "ops": sorted({c.get("op", "") for c in checks}),
        "hedef": hedef,
        "konu": konu,
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12], parts


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="ornek seed sayisi")
    parser.add_argument("--rng", type=int, default=0, help="tekrarurutilebilirlik icin")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds = sample_many(args.n, args.rng)
    if args.json:
        print(json.dumps([s.to_dict() for s in seeds], ensure_ascii=False, indent=2))
        return 0

    grid = len(TOOL_BUNDLES) * len(KONULAR) * len(HEDEFLER) * len(set(BUKULMELER)) * 5
    print(f"grid buyuklugu: {grid} kombinasyon\n")
    for s in seeds:
        print(f"{s.id}  {s.bundle:14} {'+'.join(s.tools):18} {s.konu:22} "
              f"{s.hedef:16} bukulme={s.bukulme:12} adim={s.adim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
