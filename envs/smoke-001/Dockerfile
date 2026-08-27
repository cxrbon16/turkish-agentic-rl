FROM python:3.12-slim

# Deterministik davranis: locale'e bagli siralama/format farklarini kapat.
ENV LC_ALL=C \
    LANG=C \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ek paket kurmuyoruz: base imajda bash, coreutils, findutils, grep ve sed
# zaten var. Build'in ag gerektirmemesi hermetiklik acisindan da iyi.
# (gawk gibi araclar gerektiginde text split'i icin ayri bir imaj yapilacak.)

WORKDIR /workspace

CMD ["sleep", "infinity"]
