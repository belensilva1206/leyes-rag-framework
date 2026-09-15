"""Fase 1 de la arquitectura: Ingesta y preprocesamiento.

- Convierte cada PDF de raw_leyes/Leyes a Markdown (data/markdown/*.md).
- Chunking jerárquico (Título/Artículo/Inciso) -> data/chunks.jsonl
- Indexa en ChromaDB (dense) + BM25 (sparse) -> data/chroma_db/, data/bm25.pkl

Uso:
    python scripts/01_ingest_and_index.py
    RAG_MODE=production python scripts/01_ingest_and_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leyes_rag.config import LAW_CATALOG, MARKDOWN_DIR, RAW_LEYES_DIR, SETTINGS, PROJECT_ROOT
from leyes_rag.ingest.chunking import build_chunks, parse_paragraphs_to_articles, render_markdown
from leyes_rag.ingest.pdf_extract import extract_raw_paragraphs
from leyes_rag.indexing.vector_store import build_vector_store
from leyes_rag.indexing.sparse_index import build_bm25_index


def ingest_all() -> list:
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    print(f"{'Ley':45s} {'Artículos':>10s} {'Chunks':>8s}")
    print("-" * 68)
    for law in LAW_CATALOG:
        pdf_path = RAW_LEYES_DIR / law["file"]
        paragraphs = extract_raw_paragraphs(pdf_path)
        preamble, articles = parse_paragraphs_to_articles(paragraphs)
        chunks = build_chunks(law["id"], articles, SETTINGS.chunking)
        all_chunks.extend(chunks)

        md = render_markdown(law["id"], preamble, articles)
        md_path = MARKDOWN_DIR / f"{law['id']}.md"
        md_path.write_text(md, encoding="utf-8")

        print(f"{law['nombre_corto']:45s} {len(articles):>10d} {len(chunks):>8d}")

    chunks_path = PROJECT_ROOT / "data" / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    print("-" * 68)
    print(f"Total chunks: {len(all_chunks)}  ->  {chunks_path}")
    return all_chunks


def main():
    print(f"Modo RAG: {SETTINGS.mode}\n")
    chunks = ingest_all()

    print("\nIndexando en ChromaDB (dense)...")
    build_vector_store(chunks)

    print("Indexando BM25 (sparse)...")
    build_bm25_index(chunks)

    print("\nListo. Índices en data/chroma_db/ y data/bm25.pkl")


if __name__ == "__main__":
    main()
