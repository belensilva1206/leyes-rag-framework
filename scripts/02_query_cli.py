"""Fase de consulta (query time) SIN reingesta.

A diferencia de 01_ingest_and_index.py (que convierte los PDF a Markdown,
hace el chunking jerárquico y RECONSTRUYE los índices desde cero), este
script asume que esos índices ya existen en disco -- data/chroma_db/,
data/bm25.pkl, data/embedder_state.pkl -- y solo los CARGA para responder
consultas. No toca los PDF ni vuelve a chunkear nada.

Requisito: haber corrido al menos una vez
    python scripts/01_ingest_and_index.py

Uso:
    python scripts/02_query_cli.py                      # modo interactivo
    python scripts/02_query_cli.py "¿pregunta puntual?"  # una sola consulta
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leyes_rag.config import CHROMA_DIR, PROJECT_ROOT, SETTINGS

_BM25_PATH = PROJECT_ROOT / "data" / "bm25.pkl"
_EMBEDDER_STATE_PATH = PROJECT_ROOT / "data" / "embedder_state.pkl"


def _check_index_exists() -> None:
    faltantes = [
        p
        for p in (CHROMA_DIR, _BM25_PATH, _EMBEDDER_STATE_PATH)
        if not p.exists()
    ]
    if faltantes:
        print("No encontré los índices ya construidos. Falta(n):")
        for p in faltantes:
            print(f"  - {p}")
        print(
            "\nCorre primero (una sola vez, o cada vez que cambien las leyes):\n"
            "    python scripts/01_ingest_and_index.py\n"
        )
        sys.exit(1)


def _print_answer(result) -> None:
    trace = result.trace
    leyes = ", ".join(trace.route.law_ids) or "todo el corpus (sin filtro de router)"
    print(f"\nLeyes consultadas: {leyes}")
    print("-" * 70)
    print(result.answer.answer)
    print("-" * 70)


def main() -> None:
    _check_index_exists()

    print(f"Modo RAG: {SETTINGS.mode}")
    t0 = time.time()
    print("Cargando índices ya existentes desde disco (sin reconvertir PDFs)...")
    from leyes_rag.rag_service import LeyesRAGService  # import tardío: recién aquí se cargan los pickles/Chroma

    svc = LeyesRAGService()
    print(f"Listo en {time.time() - t0:.1f}s.\n")

    if len(sys.argv) > 1:
        pregunta = " ".join(sys.argv[1:])
        _print_answer(svc.ask(pregunta))
        return

    print("Escribe tu pregunta como si fueras un cliente PYME.")
    print("('salir' o Ctrl+C para terminar)\n")
    while True:
        try:
            pregunta = input("PYME> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        _print_answer(svc.ask(pregunta))


if __name__ == "__main__":
    main()
