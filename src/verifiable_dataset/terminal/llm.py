"""One place to build the OpenAI-compatible client.

Anahtar cozumleme ve /v1 eklemesi dort ayri komutta kopyalanmisti; biri
duzeltilince digerleri geride kaliyordu. Hepsi buradan geciyor.
"""
from __future__ import annotations

import os
import pathlib

MISTRAL_URL = "https://api.mistral.ai/v1"

ANAHTAR_ADLARI = ("MISTRAL_API_KEY", "OPENAI_API_KEY")


def resolve_key(explicit: str = "") -> tuple[str, str]:
    """(anahtar, nereden geldigi) dondur.

    Kendi sunucun anahtar istemez ama istemci bos anahtari reddediyor, o
    yuzden hicbir yerde bulunamazsa yer tutucu doner.
    """
    if explicit:
        return explicit, "--api-key"
    for isim in ANAHTAR_ADLARI:
        if os.environ.get(isim):
            return os.environ[isim], isim
    env = pathlib.Path(".env")
    if env.exists():
        for satir in env.read_text(encoding="utf-8").splitlines():
            satir = satir.strip()
            for isim in ANAHTAR_ADLARI:
                if satir.startswith(f"{isim}="):
                    return satir.split("=", 1)[1].strip().strip('"').strip("'"), f".env:{isim}"
    return "dummy", "yok"


def resolve_base_url(base_url: str, kaynak: str) -> str:
    """Uc adresini duzelt; verilmemisse anahtarin geldigi yere gore tahmin et.

    --base-url unutuldugunda istemci sessizce OpenAI'ye gidip 401 aliyordu.
    Anahtar MISTRAL_API_KEY'den geldiyse kastedilen ucun Mistral olmasi
    neredeyse kesin.
    """
    if base_url:
        url = base_url.rstrip("/")
        return url if url.endswith("/v1") else url + "/v1"
    if "MISTRAL_API_KEY" in kaynak:
        return MISTRAL_URL
    return ""


def make_client(base_url: str = "", api_key: str = "", verbose: bool = True):
    """OpenAI uyumlu istemciyi kur ve nereye baglandigini soyle."""
    from openai import OpenAI

    key, kaynak = resolve_key(api_key)
    url = resolve_base_url(base_url, kaynak)
    if verbose:
        nereye = url or "https://api.openai.com/v1 (varsayilan)"
        print(f"uc: {nereye}   anahtar: {kaynak}")
        if not url and kaynak == "yok":
            print("uyari: --base-url verilmedi ve anahtar bulunamadi; "
                  "kendi sunucun icin --base-url ver")
    return OpenAI(base_url=url or None, api_key=key)
