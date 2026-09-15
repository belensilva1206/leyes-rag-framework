"""Helpers de presentación para el notebook de demo (no forman parte del
pipeline en sí, solo dan formato bonito a los resultados en Jupyter)."""
from __future__ import annotations

from IPython.display import Markdown

from leyes_rag.retrieval.pipeline import RetrievalTrace
from leyes_rag.rag_service import QueryResult


def show_trace(trace: RetrievalTrace) -> Markdown:
    lines = [f"### 🔎 Trace de recuperación para: *\"{trace.query}\"*", ""]
    lines.append(f"**1. Ruteo semántico** → leyes candidatas: `{trace.route.law_ids or 'sin filtro (todo el corpus)'}`")
    lines.append(f"> {trace.route.reasoning}")
    lines.append("")
    lines.append(f"**2. Búsqueda densa (coseno)** — top {min(5,len(trace.dense_results))} de {len(trace.dense_results)}:")
    for r in trace.dense_results[:5]:
        lines.append(f"- `{r['score']:.3f}` {r['metadata']['citation']}")
    lines.append("")
    lines.append(f"**3. Búsqueda dispersa (BM25)** — top {min(5,len(trace.sparse_results))} de {len(trace.sparse_results)}:")
    for r in trace.sparse_results[:5]:
        lines.append(f"- `{r['score']:.3f}` {r['metadata']['citation']}")
    lines.append("")
    lines.append(f"**4. Fusión RRF** — top {min(5,len(trace.fused))} de {len(trace.fused)}:")
    for r in trace.fused[:5]:
        lines.append(f"- `{r['rrf_score']:.4f}` {r['metadata']['citation']}")
    lines.append("")
    lines.append(f"**5. Reranker** — contexto final ({len(trace.final)} chunks):")
    for r in trace.final:
        lines.append(
            f"- `{r.rerank_score:.3f}` **{r.citation}** "
            f"(dense#{r.dense_rank or '-'}, sparse#{r.sparse_rank or '-'}, rrf={r.rrf_score:.4f})"
        )
    return Markdown("\n".join(lines))


def show_answer(result: QueryResult) -> Markdown:
    lines = [f"### 💬 Pregunta: *\"{result.trace.query}\"*", ""]
    lines.append(f"**Leyes consultadas por el router:** {', '.join(result.trace.route.law_ids) or 'todo el corpus'}")
    lines.append("")
    lines.append("**Respuesta:**")
    lines.append("")
    lines.append(result.answer.answer)
    lines.append("")
    if result.answer.citations:
        lines.append("**Fuentes citadas:**")
        for c in result.answer.citations:
            lines.append(f"- {c}")
    return Markdown("\n".join(lines))
