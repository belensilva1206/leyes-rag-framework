"""Normalización de texto en español compartida por BM25 y el TF-IDF del
modo lite. Aplicamos stemming (Snowball, 100% algorítmico — no descarga
datos ni modelos) para que preguntas y artículos que usan formas distintas
de la misma raíz calcen léxicamente:

    "¿cómo elimino mis datos?"  (elimin-o)
    "Derecho de supresión... la eliminación de los datos..." (elimin-ación)

Sin stemming, "eliminar" y "eliminación" son tokens distintos y ni BM25 ni
TF-IDF los relacionan — justo la clase de sinonimia superficial que en el
modo producción resolvería el embedding semántico de BGE-M3, pero que en
modo lite hay que compensar así.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)

_STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "al", "a", "en", "y", "o", "u",
    "que", "se", "su", "sus", "por", "para", "con", "sin", "un", "una",
    "unos", "unas", "es", "ser", "será", "sea", "lo", "como", "más", "menos",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "cual",
    "cuales", "sobre", "entre", "cuando", "donde", "si", "no", "ya", "le",
    "les", "también", "qué", "cómo", "puedo", "debo", "tengo",
}

try:
    from nltk.stem.snowball import SnowballStemmer

    _stemmer = SnowballStemmer("spanish")

    def _stem(word: str) -> str:
        return _stemmer.stem(word)

except ImportError:  # pragma: no cover - fallback si nltk no está instalado
    _SUFFIXES = ("ciones", "ción", "mente", "ando", "iendo", "ados", "adas", "es", "as", "os", "a", "o", "e")

    def _stem(word: str) -> str:
        for suf in _SUFFIXES:
            if len(word) > len(suf) + 3 and word.endswith(suf):
                return word[: -len(suf)]
        return word


def tokenize(text: str, stem: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens
