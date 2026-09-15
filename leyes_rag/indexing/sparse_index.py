"""Índice disperso BM25 (rank_bm25) sobre los mismos chunks jerárquicos.

BM25 es igual en modo lite y producción (no depende de modelos pesados), tal
como pide la arquitectura: recuperación híbrida = denso (BGE-M3/Chroma) +
disperso (BM25).
"""
from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from leyes_rag.config import PROJECT_ROOT
from leyes_rag.ingest.chunking import LawChunk
from leyes_rag.nlp_utils import tokenize

_BM25_PATH = PROJECT_ROOT / "data" / "bm25.pkl"


def build_bm25_index(chunks: list[LawChunk]) -> "BM25Index":
    corpus_tokens = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    _BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_BM25_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": [c.chunk_id for c in chunks],
                "metadatas": [
                    {
                        "law_id": c.law_id,
                        "law_name": c.law_name,
                        "law_number": c.law_number,
                        "source_file": c.source_file,
                        "capitulo": c.capitulo or "",
                        "titulo": c.titulo or "",
                        "parrafo": c.parrafo or "",
                        "articulo": c.articulo or "",
                        "inciso_range": c.inciso_range or "",
                        "citation": c.citation,
                        "page_no": c.page_no,
                    }
                    for c in chunks
                ],
                "documents": [c.text for c in chunks],
            },
            f,
        )
    return load_bm25_index()


class BM25Index:
    def __init__(self, bm25: BM25Okapi | None = None, chunk_ids=None, metadatas=None, documents=None):
        if bm25 is None:
            with open(_BM25_PATH, "rb") as f:
                data = pickle.load(f)
            bm25, chunk_ids, metadatas, documents = (
                data["bm25"], data["chunk_ids"], data["metadatas"], data["documents"]
            )
        self.bm25: BM25Okapi = bm25
        self.chunk_ids: list[str] = chunk_ids
        self.metadatas: list[dict] = metadatas
        self.documents: list[str] = documents

    def search(self, query: str, top_k: int) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            out.append(
                {
                    "chunk_id": self.chunk_ids[i],
                    "score": float(scores[i]),
                    "metadata": self.metadatas[i],
                    "document": self.documents[i],
                }
            )
        return out


def load_bm25_index() -> BM25Index:
    return BM25Index()
