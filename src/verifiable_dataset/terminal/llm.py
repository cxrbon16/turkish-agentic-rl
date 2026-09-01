"""One place to build the OpenAI-compatible client.

Anahtar cozumleme ve /v1 eklemesi dort ayri komutta kopyalanmisti; biri
duzeltilince digerleri geride kaliyordu. Hepsi buradan geciyor.
"""
from __future__ import annotations

import os
import pathlib
import threading
import time

MISTRAL_URL = "https://api.mistral.ai/v1"
OPENAI_URL = "https://api.openai.com/v1"


def _env(isim: str) -> str:
    """Once gercek ortam degiskeni, sonra .env dosyasi."""
    if os.environ.get(isim):
        return os.environ[isim]
    dosya = pathlib.Path(".env")
    if dosya.exists():
        for satir in dosya.read_text(encoding="utf-8").splitlines():
            satir = satir.strip()
            if satir.startswith(f"{isim}="):
                return satir.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def resolve_base_url(base_url: str = "") -> str:
    """Uc adresini bul ve /v1 ekini garanti et.

    Sirayla: acik verilen, .env'deki OPENAI_BASE_URL, yalnizca Mistral
    anahtari varsa Mistral, yoksa OpenAI varsayilani.
    """
    url = base_url or _env("OPENAI_BASE_URL")
    if not url:
        url = MISTRAL_URL if (_env("MISTRAL_API_KEY") and not _env("OPENAI_API_KEY")) \
            else OPENAI_URL
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def resolve_key(explicit: str = "", base_url: str = "") -> tuple[str, str]:
    """(anahtar, nereden geldigi) dondur -- uca gore dogru olani sec.

    Iki anahtar birden tanimliyken sabit bir sira izlemek, Mistral
    anahtarini OpenAI ucuna gondermek gibi teshisi zor 401'ler uretiyordu.
    Uc adresi hangi saglayiciya aitse once onun anahtari deneniyor.
    """
    if explicit:
        return explicit, "--api-key"
    mistral_uc = "mistral" in (base_url or "").lower()
    sira = (["MISTRAL_API_KEY", "OPENAI_API_KEY"] if mistral_uc
            else ["OPENAI_API_KEY", "MISTRAL_API_KEY"])
    for isim in sira:
        deger = _env(isim)
        if deger:
            return deger, isim
    # Kendi sunucun anahtar istemez ama istemci bosu reddediyor.
    return "dummy", "yok"


def resolve_model(explicit: str = "") -> str:
    return explicit or _env("OPENAI_MODEL_NAME")


class _Completions:
    """Istekler arasinda en az `aralik` saniye birakan ince sarmalayici."""

    def __init__(self, inner, aralik: float):
        self._inner = inner
        self._aralik = aralik
        self._son = 0.0
        # Es zamanli kosuda kisit butun is parcaciklari icin ortak olmali,
        # yoksa N parcacik N kat hizli istek atar.
        self._kilit = threading.Lock()

    def create(self, **kwargs):
        with self._kilit:
            bekle = self._aralik - (time.monotonic() - self._son)
            if bekle > 0:
                time.sleep(bekle)
            self._son = time.monotonic()
        return self._inner.create(**kwargs)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class ThrottledClient:
    """Yalnizca chat.completions.create yolunu kisitlar, gerisi aynen gecer."""

    def __init__(self, inner, aralik: float):
        self._inner = inner
        self.chat = _Chat(_Completions(inner.chat.completions, aralik))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def preflight(base_url: str = "", api_key: str = "", model: str = "") -> tuple[bool, str]:
    """Kosuya baslamadan once ucun bu modeli sunup sunmadigini dogrula.

    Yanlis uca gitmek 40 adayin 40'inda ayni hatayi uretip ancak kosunun
    sonunda fark edilebiliyordu. Tek bir /models sorgusu bunu basta soyler
    ve nereye gidildigini de gozle gorulur kilar.
    """
    import json
    import urllib.error
    import urllib.request

    url = resolve_base_url(base_url)
    key, kaynak = resolve_key(api_key, url)
    istek = urllib.request.Request(
        url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(istek, timeout=30) as yanit:
            veri = json.load(yanit)
    except urllib.error.HTTPError as e:
        govde = e.read().decode("utf-8", "replace")[:200]
        return False, f"{url} /models HTTP {e.code}: {govde}  (anahtar: {kaynak})"
    except Exception as e:  # noqa: BLE001 - ag hatasi da erken soylenmeli
        return False, f"{url} adresine ulasilamadi: {type(e).__name__}: {e}"

    adlar = [m.get("id", "") for m in veri.get("data", []) if isinstance(m, dict)]
    if model and model not in adlar:
        return False, (f"'{model}' bu ucta yok.\n  uc      : {url}\n"
                       f"  anahtar : {kaynak}\n  sunulan : {', '.join(adlar) or '(bos)'}")
    return True, f"{url}  model={model or '?'}  anahtar={kaynak}"


def make_client(base_url: str = "", api_key: str = "", istek_araligi: float = 0.0,
                verbose: bool = True):
    """OpenAI uyumlu istemciyi kur ve nereye baglandigini soyle.

    ``istek_araligi`` iki istek arasindaki en kisa sureyi saniye cinsinden
    verir. Bir episode her turda bir istek attigi icin kisitin episode
    basina degil istek basina olmasi gerekiyor.
    """
    from openai import OpenAI

    url = resolve_base_url(base_url)
    key, kaynak = resolve_key(api_key, url)
    if verbose:
        kisit = f"   istek araligi: {istek_araligi}sn" if istek_araligi else ""
        print(f"uc: {url}   anahtar: {kaynak}{kisit}")
        if kaynak == "yok":
            print("uyari: anahtar bulunamadi; kendi sunucun degilse --api-key ver")
    client = OpenAI(base_url=url, api_key=key)
    return ThrottledClient(client, istek_araligi) if istek_araligi > 0 else client
