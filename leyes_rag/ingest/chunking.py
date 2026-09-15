"""Parser jerárquico de leyes chilenas: convierte los párrafos limpios que
entrega pdf_extract.py en:

1. Un documento Markdown legible (para inspección humana / debugging), con
   encabezados '#' por Título/Capítulo/Párrafo y '##'/negrita por Artículo.
2. Una lista de LawChunk listos para indexar: cada uno lleva su ruta
   jerárquica completa (ley, capítulo, título, párrafo, artículo, inciso)
   como metadata, más una cita legible ("Ley 19.496, Artículo 3°, inciso 2°")
   que es lo que se le muestra al usuario final como fuente.

Jerarquía soportada (de mayor a menor, todas opcionales salvo Artículo):
    Capítulo > Título > Párrafo > Artículo > Inciso

Un chunk = 1 artículo completo si cabe en max_chars_per_chunk; si el
artículo es más largo, se subdivide por inciso (cada inciso es un párrafo
dentro del artículo: el texto introductorio, cada numeral "1.-", cada
literal "a)", etc. se cuentan como incisos correlativos), repitiendo el
encabezado jerárquico en cada sub-chunk para que sea recuperable de forma
autónoma sin perder el contexto de a qué ley/artículo pertenece.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from leyes_rag.config import ChunkingConfig, LAW_CATALOG
from leyes_rag.ingest.pdf_extract import RawParagraph

_ROMAN_OR_WORD = r"[IVXLCDM]+|[ÚúUu]nico|[Ff]inal|[Pp]reliminar|\d+"


# Nota: Capítulo/Título/Párrafo se detectan sin distinguir mayúsculas (en
# el corpus aparecen a veces en VERSALES "TÍTULO I" o espaciadas "T I T U L
# O", y a veces en formato título "Párrafo 2º"). "Artículo", en cambio, se
# mantiene sensible a mayúsculas (solo con A mayúscula): un "artículo" en
# minúscula casi siempre es una referencia cruzada DENTRO de un párrafo
# (p.ej. "conforme al artículo 28.") y no el inicio de un artículo nuevo.
_CAPITULO_RE = re.compile(
    rf'^"?\s*(?i:Cap[íi]tulo)\s+({_ROMAN_OR_WORD})\b[\.,]?\s*(.*)$'
)
_TITULO_RE = re.compile(
    rf'^"?\s*(?i:T\s*[ÍI]\s*T\s*U\s*L\s*O)\s+({_ROMAN_OR_WORD})\b[\.,]?\s*(.*)$'
)
_PARRAFO_RE = re.compile(
    rf'^"?\s*(?i:P\s*[ÁA]\s*R\s*R\s*A\s*F\s*O)\s+(\d+°?º?|{_ROMAN_OR_WORD})\b[\.,]?\s*(.*)$'
)
# "Artículo 1°.-", "Artículo 15 C.-", "Artículo primero.-",
# "Artículo primero transitorio.-", "Artículo transitorio.-"
_ARTICULO_RE = re.compile(
    r'^"?\s*Art[íi]culo\s+'
    r"("
    # "1°", "15 C", "39-C", "39 bis" ... (números con eventual sufijo de
    # letra introducido por posteriores modificaciones a la ley, separado
    # por espacio o guion indistintamente)
    r"\d+\s*[°ºª]?[\s\-]*[A-Z]?(?:\s*bis|\s*ter|\s*qu[áa]ter)?"
    r"|[A-ZÁÉÍÓÚa-záéíóúÑñ]+(?:\s+transitorio)?"
    r")"
    r"\s*[\.\-–,]+\s*(.*)$"
)

# Marcadores de inciso con etiqueta propia dentro de un artículo.
_LITERAL_RE = re.compile(r"^([a-z])\)\s*(.*)$")
_NUMERAL_RE = re.compile(r"^(\d+)[°º.\-]+\s*[-.]?\s*(.*)$")

_FILE_BY_NAME = {law["file"]: law for law in LAW_CATALOG}

# Encabezado (no siempre precedido de "Título"/"Capítulo") que marca el
# inicio de las disposiciones transitorias, donde la numeración de
# artículos suele reiniciarse.
_TRANSITORIAS_RE = re.compile(
    r'^"?\s*(DISPOSICI[ÓO]N(?:ES)?\s+TRANSITORIAS?|'
    r"[ÁA]rt[íi]culos?\s+transitorios)\s*$",
    re.IGNORECASE,
)


@dataclass
class Heading:
    level: str  # "capitulo" | "titulo" | "parrafo"
    label: str  # "I", "Preliminar", ...
    title_text: str  # texto descriptivo que suele venir en el mismo párrafo


@dataclass
class ArticleUnit:
    articulo_label: str  # "1°", "15 C", "primero", "transitorio"...
    heading_text: str  # eventual título del artículo ("Objeto y ámbito...")
    incisos: list["Inciso"] = field(default_factory=list)
    capitulo: Heading | None = None
    titulo: Heading | None = None
    parrafo: Heading | None = None
    page_no: int = 0
    # Las "disposiciones transitorias" suelen reiniciar la numeración de
    # artículos (Artículo 1° a 5° aparece dos veces: régimen permanente y
    # transitorio). Este flag evita confundir ambos y colisionar chunk_ids.
    transitorio: bool = False


@dataclass
class Inciso:
    index: int  # posición correlativa dentro del artículo (1-based)
    label: str  # "inciso 1°" | "numeral 3" | "literal c)"
    text: str


@dataclass
class LawChunk:
    chunk_id: str
    law_id: str
    law_name: str
    law_number: str
    source_file: str
    capitulo: str | None
    titulo: str | None
    parrafo: str | None
    articulo: str | None
    inciso_range: str | None
    citation: str
    text: str
    page_no: int


def _match_heading(text: str) -> tuple[str, Heading] | None:
    if m := _CAPITULO_RE.match(text):
        return "capitulo", Heading("capitulo", m.group(1), m.group(2).strip())
    if m := _TITULO_RE.match(text):
        return "titulo", Heading("titulo", m.group(1), m.group(2).strip())
    if m := _PARRAFO_RE.match(text):
        return "parrafo", Heading("parrafo", m.group(1), m.group(2).strip())
    return None


def parse_paragraphs_to_articles(
    paragraphs: list[RawParagraph],
) -> tuple[list[str], list[ArticleUnit]]:
    """Recorre los párrafos en orden y arma la estructura jerárquica.

    Devuelve (preambulo, articulos):
      - preambulo: párrafos anteriores al primer Artículo/Capítulo/Título
        detectado (nombre completo de la ley, "Teniendo presente...", etc.)
      - articulos: lista de ArticleUnit con sus incisos ya clasificados.
    """
    preamble: list[str] = []
    articles: list[ArticleUnit] = []

    cur_capitulo: Heading | None = None
    cur_titulo: Heading | None = None
    cur_parrafo: Heading | None = None
    cur_article: ArticleUnit | None = None
    seen_structure = False
    in_transitorias = False
    max_num_seen = -1

    for para in paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if _TRANSITORIAS_RE.match(text):
            in_transitorias = True
            cur_article = None
            continue

        heading = _match_heading(text)
        if heading:
            level, h = heading
            seen_structure = True
            if level == "capitulo":
                cur_capitulo, cur_titulo, cur_parrafo = h, None, None
            elif level == "titulo":
                cur_titulo, cur_parrafo = h, None
            elif level == "parrafo":
                cur_parrafo = h
            cur_article = None
            continue

        art_match = _ARTICULO_RE.match(text)
        if art_match:
            seen_structure = True
            label = art_match.group(1).strip()
            label = re.sub(r"(\d)\s*-\s*([A-Z])\b", r"\1 \2", label)
            label = re.sub(r"\s+", " ", label).strip()

            # Heurística de respaldo: si no hubo un encabezado explícito de
            # "Disposiciones Transitorias" pero la numeración retrocede
            # (p.ej. veníamos del Artículo 82 y ahora aparece "Artículo
            # 1º"), es casi seguro que entramos a las disposiciones
            # transitorias, que reinician la numeración.
            num_match = re.match(r"^(\d+)", label)
            if num_match and not in_transitorias:
                num = int(num_match.group(1))
                # Umbral conservador: algunas leyes "fijan" dentro de un
                # artículo el texto completo de OTRA ley con su propia
                # numeración desde el 1 (ver p.ej. Ley 20.416, Artículo
                # Décimo, que fija la Ley de Acuerdos de Producción
                # Limpia). Eso también es un "retroceso" de numeración
                # pero no son disposiciones transitorias, así que solo
                # disparamos el heurístico ante una caída grande y desde
                # una numeración ya avanzada (típico de un artículado
                # principal largo que termina y reinicia en las
                # transitorias).
                if max_num_seen >= 15 and num <= 10 and num < max_num_seen:
                    in_transitorias = True
            if num_match:
                max_num_seen = max(max_num_seen, int(num_match.group(1)))

            rest = art_match.group(2).strip()
            # El artículo a veces trae un título corto antes del texto,
            # p.ej. "Objeto y ámbito de aplicación. La presente ley..."
            heading_text = ""
            body = rest
            m2 = re.match(r"^([A-ZÁÉÍÓÚÑ][^.]{2,60})\.\s+(.*)$", rest)
            if m2 and len(m2.group(1).split()) <= 8:
                heading_text = m2.group(1).strip()
                body = m2.group(2).strip()
            cur_article = ArticleUnit(
                articulo_label=label,
                heading_text=heading_text,
                capitulo=cur_capitulo,
                titulo=cur_titulo,
                parrafo=cur_parrafo,
                page_no=para.page_no,
                transitorio=in_transitorias,
            )
            articles.append(cur_article)
            if body:
                cur_article.incisos.append(
                    Inciso(index=1, label="inciso 1°", text=body)
                )
            continue

        if cur_article is None:
            if not seen_structure:
                preamble.append(text)
            # Texto fuera de cualquier artículo tras haber visto estructura
            # (ej. encabezado de "Disposiciones transitorias" sin patrón
            # exacto): se ignora para no ensuciar el último artículo.
            continue

        idx = len(cur_article.incisos) + 1
        if m := _LITERAL_RE.match(text):
            label = f"literal {m.group(1)})"
            body = text
        elif m := _NUMERAL_RE.match(text):
            label = f"numeral {m.group(1)}"
            body = text
        else:
            label = f"inciso {idx}°"
            body = text
        cur_article.incisos.append(Inciso(index=idx, label=label, text=body))

    return preamble, articles


def _citation(
    law_name: str, articulo: str | None, inciso_labels: list[str], transitorio: bool = False
) -> str:
    parts = [law_name]
    if articulo:
        suffix = " (transitorio)" if transitorio else ""
        parts.append(f"Artículo {articulo}{suffix}")
    if inciso_labels:
        if len(inciso_labels) == 1:
            parts.append(inciso_labels[0].capitalize())
        else:
            parts.append(f"{inciso_labels[0]} a {inciso_labels[-1]}".capitalize())
    return ", ".join(parts)


def _heading_path_str(art: ArticleUnit) -> str:
    parts = []
    if art.capitulo:
        parts.append(f"Capítulo {art.capitulo.label}" + (f" ({art.capitulo.title_text})" if art.capitulo.title_text else ""))
    if art.titulo:
        parts.append(f"Título {art.titulo.label}" + (f" ({art.titulo.title_text})" if art.titulo.title_text else ""))
    if art.parrafo:
        parts.append(f"Párrafo {art.parrafo.label}" + (f" ({art.parrafo.title_text})" if art.parrafo.title_text else ""))
    return " > ".join(parts)


def build_chunks(
    law_id: str,
    articles: list[ArticleUnit],
    cfg: ChunkingConfig,
) -> list[LawChunk]:
    law = next(law for law in LAW_CATALOG if law["id"] == law_id)
    law_name = law["nombre_corto"]
    law_number = law["numero"]
    source_file = law["file"]

    chunks: list[LawChunk] = []
    chunk_id_seen: dict[str, int] = {}

    def unique_chunk_id(base: str) -> str:
        n = chunk_id_seen.get(base, 0) + 1
        chunk_id_seen[base] = n
        return base if n == 1 else f"{base}-{n}"

    for art in articles:
        full_text = " ".join(inc.text for inc in art.incisos).strip()
        header_prefix = _heading_path_str(art)
        trans_marker = " (transitorio)" if art.transitorio else ""
        art_title = f" — {art.heading_text}" if art.heading_text else ""
        art_slug_base = f"{law_id}__art{_slug(art.articulo_label)}" + (
            "-trans" if art.transitorio else ""
        )

        if len(full_text) <= cfg.max_chars_per_chunk or len(art.incisos) <= 1:
            body = full_text
            header = f"[{law_name}] Artículo {art.articulo_label}{trans_marker}{art_title}"
            if header_prefix:
                header = f"[{law_name}] {header_prefix} — Artículo {art.articulo_label}{trans_marker}{art_title}"
            chunk_text = f"{header}\n{body}" if body else header
            chunks.append(
                LawChunk(
                    chunk_id=unique_chunk_id(art_slug_base),
                    law_id=law_id,
                    law_name=law_name,
                    law_number=law_number,
                    source_file=source_file,
                    capitulo=art.capitulo.label if art.capitulo else None,
                    titulo=art.titulo.label if art.titulo else None,
                    parrafo=art.parrafo.label if art.parrafo else None,
                    articulo=art.articulo_label,
                    inciso_range=None,
                    citation=_citation(law_name, art.articulo_label, [], art.transitorio),
                    text=chunk_text,
                    page_no=art.page_no,
                )
            )
        else:
            # Artículo largo: un chunk por inciso (o agrupando incisos
            # consecutivos cortos hasta llenar el límite), repitiendo el
            # encabezado jerárquico para que cada trozo sea autocontenible.
            group: list[Inciso] = []
            group_len = 0

            def flush_group(idx_suffix: int):
                nonlocal group, group_len
                if not group:
                    return
                labels = [i.label for i in group]
                if len(labels) == 1 or labels[0].startswith(("numeral", "literal")):
                    # El grupo arranca en un numeral/literal explícito: los
                    # incisos genéricos que le siguen son su propia
                    # continuación, así que basta con el label inicial
                    # (más preciso que un rango "numeral 21–inciso 96°").
                    inciso_range = labels[0]
                else:
                    inciso_range = f"{labels[0]}–{labels[-1]}"
                header = f"[{law_name}] Artículo {art.articulo_label}{trans_marker}{art_title}"
                if header_prefix:
                    header = f"[{law_name}] {header_prefix} — Artículo {art.articulo_label}{trans_marker}{art_title}"
                body = " ".join(i.text for i in group)
                chunk_text = f"{header} ({inciso_range})\n{body}"
                chunks.append(
                    LawChunk(
                        chunk_id=unique_chunk_id(f"{art_slug_base}__{idx_suffix}"),
                        law_id=law_id,
                        law_name=law_name,
                        law_number=law_number,
                        source_file=source_file,
                        capitulo=art.capitulo.label if art.capitulo else None,
                        titulo=art.titulo.label if art.titulo else None,
                        parrafo=art.parrafo.label if art.parrafo else None,
                        articulo=art.articulo_label,
                        inciso_range=inciso_range,
                        citation=_citation(
                            law_name, art.articulo_label, [inciso_range], art.transitorio
                        ),
                        text=chunk_text,
                        page_no=art.page_no,
                    )
                )
                group = []
                group_len = 0

            group_idx = 0
            for inc in art.incisos:
                is_boundary_marker = inc.label.startswith("numeral") or inc.label.startswith("literal")
                should_flush = group and (
                    group_len + len(inc.text) > cfg.max_chars_per_chunk
                    # Un numeral/literal nuevo (p.ej. el Art. 19 de la
                    # Constitución enumera ~26 derechos distintos como
                    # numerales) es un punto de corte natural: mejor un
                    # chunk por numeral que diluir varios derechos
                    # distintos en un solo chunk genérico "inciso 90-95".
                    or is_boundary_marker
                )
                if should_flush:
                    group_idx += 1
                    flush_group(group_idx)
                group.append(inc)
                group_len += len(inc.text)
            group_idx += 1
            flush_group(group_idx)

    return chunks


def _slug(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", label.strip()).strip("-").lower() or "x"


def render_markdown(law_id: str, preamble: list[str], articles: list[ArticleUnit]) -> str:
    law = next(law for law in LAW_CATALOG if law["id"] == law_id)
    lines = [f"# {law['nombre_corto']}", ""]
    if preamble:
        lines.append("> " + " ".join(preamble))
        lines.append("")

    last_cap = last_tit = last_par = None
    for art in articles:
        if art.capitulo and art.capitulo.label != last_cap:
            lines.append(f"## Capítulo {art.capitulo.label}" + (f" — {art.capitulo.title_text}" if art.capitulo.title_text else ""))
            last_cap, last_tit, last_par = art.capitulo.label, None, None
        if art.titulo and art.titulo.label != last_tit:
            lines.append(f"### Título {art.titulo.label}" + (f" — {art.titulo.title_text}" if art.titulo.title_text else ""))
            last_tit, last_par = art.titulo.label, None
        if art.parrafo and art.parrafo.label != last_par:
            lines.append(f"#### Párrafo {art.parrafo.label}" + (f" — {art.parrafo.title_text}" if art.parrafo.title_text else ""))
            last_par = art.parrafo.label

        title_suffix = f" — {art.heading_text}" if art.heading_text else ""
        lines.append(f"\n**Artículo {art.articulo_label}{title_suffix}**\n")
        for inc in art.incisos:
            if inc.label.startswith("inciso"):
                lines.append(inc.text)
            else:
                lines.append(f"- ({inc.label.split()[-1]}) {inc.text}")
        lines.append("")
    return "\n".join(lines)
