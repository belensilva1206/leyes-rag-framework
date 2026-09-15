"""Extracción y limpieza de texto desde los PDF de leyes chilenas (formato
Biblioteca del Congreso Nacional / LeyChile).

Estos PDF tienen tres particularidades que hay que resolver ANTES de poder
detectar la jerarquía (Título/Artículo/inciso):

1. Cada página repite un encabezado boilerplate (nombre de la ley, aviso de
   firma digital, "documento generado el...", "página N de M", metadatos de
   publicación/vigencia). Hay que eliminarlo o contamina cada chunk.
2. Muchas líneas traen, pegada al final tras un salto grande de espacios,
   una anotación marginal de "historia de la ley" (ej. "CPR Art. 1° D.O.
   24.10.1980  LEY N° 19.611 Art. único N°1 D.O. 16.06.1999"). Esto viene
   del layout a dos columnas del PDF original, que PyMuPDF concatena en una
   sola línea de texto.
3. El texto NO usa saltos de línea para indicar fin de párrafo: cada nuevo
   párrafo (artículo, inciso, numeral, literal) empieza con una sangría de
   ~4-8 espacios; las líneas de continuación de un párrafo que se wrappea
   empiezan en la columna 0. Hay que reconstruir los párrafos usando esa
   sangría, no los saltos de línea crudos.

La salida de este módulo es una lista de párrafos "limpios" en orden de
lectura, lista para que chunking.py detecte la jerarquía legal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Líneas completas (tras strip) que son puro boilerplate de LeyChile / BCN y
# se descartan siempre, sin importar en qué página aparezcan.
_BOILERPLATE_LINE_PATTERNS = [
    r"^(Ley|Decreto)\s+[\d.]+(\s*\(\d{4}\))?$",
    r"^Biblioteca del Congreso Nacional de Chile.*$",
    r"^Documento firmado digitalmente.*$",
    r"^Para validar,.*$",
    r"^Documento generado el.*$",
    r"^p[áa]gina\s+\d+\s+de\s+\d+$",
    r"^Fecha Publicaci[oó]n:.*$",
    r"^Fecha Promulgaci[oó]n:.*$",
    r"^Publicaci[oó]n:.*(Promulgaci[oó]n:.*)?$",
    r"^Tipo Versi[oó]n:.*$",
    r"^Versi[oó]n:.*$",
    r"^Inicio Vigencia:.*$",
    r"^Fin Vigencia:.*$",
    r"^Ultima Modificaci[oó]n:.*$",
    r"^Url Corta:.*$",
    r"^Tiene Texto Refundido:.*$",
]
_BOILERPLATE_RE = re.compile("|".join(f"(?:{p})" for p in _BOILERPLATE_LINE_PATTERNS))

# Umbral de "salto grande de espacios" que separa el texto real de una
# anotación marginal pegada al final de la línea (ver punto 2 del docstring).
_MARGIN_GAP_RE = re.compile(r"\s{3,}")

# Rango de sangría (en espacios) que marca el INICIO de un nuevo párrafo.
# 0 espacios = línea de continuación de un párrafo ya abierto.
# Sangrías > _MAX_PARA_INDENT (anotación marginal pura, sin texto real
# antes del primer salto grande) se descartan por completo.
_MIN_PARA_INDENT = 2
_MAX_PARA_INDENT = 12

# Los encabezados (Título/Capítulo/Párrafo/Artículo) a veces vienen
# centrados con sangría 0 (p.ej. "T I T U L O  I" pegado tras dos puntos),
# así que además de la sangría se detectan por patrón y SIEMPRE fuerzan un
# nuevo párrafo, sin importar la sangría de la línea.
_HEADING_START_RE = re.compile(
    r'^"?\s*('
    r"(?i:Cap[íi]tulo)\s"
    r"|(?i:T\s*[ÍI]\s*T\s*U\s*L\s*O)\s"
    r"|(?i:P\s*[ÁA]\s*R\s*R\s*A\s*F\s*O)\s"
    r"|Art[íi]culo\s+\S"
    r")"
)


@dataclass
class RawParagraph:
    page_no: int  # 1-indexed
    text: str


def _page_lines(page: "pymupdf.Page") -> list[str]:
    return page.get_text("text").split("\n")


def _strip_margin_annotation(line: str) -> str:
    """Corta la línea en el primer salto de >=3 espacios, que en estos PDF
    siempre separa el texto real (izquierda) de una anotación marginal de
    historia de la ley (derecha). El texto legal en español no usa saltos de
    varios espacios seguidos, así que el heurístico es seguro."""
    stripped = line.strip(" ")
    if not stripped:
        return ""
    m = _MARGIN_GAP_RE.search(stripped)
    if m:
        stripped = stripped[: m.start()]
    return stripped.strip()


def extract_raw_paragraphs(pdf_path: Path) -> list[RawParagraph]:
    """Devuelve los párrafos reconstruidos de un PDF, en orden de lectura,
    ya sin boilerplate de página ni anotaciones marginales."""
    doc = pymupdf.open(pdf_path)
    paragraphs: list[RawParagraph] = []
    current_buf: list[str] = []
    current_page = 1

    def flush():
        if current_buf:
            text = " ".join(current_buf).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                paragraphs.append(RawParagraph(page_no=current_page, text=text))
        current_buf.clear()

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_no = page_index + 1
        for raw_line in _page_lines(page):
            if not raw_line.strip():
                continue
            if _BOILERPLATE_RE.match(raw_line.strip()):
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            is_pure_annotation = indent > _MAX_PARA_INDENT
            if is_pure_annotation:
                # Línea que es 100% anotación marginal (sin texto real
                # antes del salto grande): se descarta y no rompe el
                # párrafo que se esté acumulando.
                continue

            content = _strip_margin_annotation(raw_line)
            if not content:
                continue

            is_new_para = indent >= _MIN_PARA_INDENT or bool(
                _HEADING_START_RE.match(content)
            )
            if is_new_para or not current_buf:
                if current_buf:
                    flush()
                current_page = page_no
                current_buf.append(content)
            else:
                current_buf.append(content)
    flush()
    return paragraphs
