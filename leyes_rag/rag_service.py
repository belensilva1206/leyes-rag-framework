"""Fachada de alto nivel: une indexación ya construida + retrieval +
generación en un único objeto `LeyesRAGService`, que es lo que consume el
notebook de demo (y cualquier futura API/CLI) para simular preguntas de
clientes PYME.
"""
from __future__ import annotations

from dataclasses import dataclass

from leyes_rag.config import SETTINGS
from leyes_rag.generation.generator import GeneratedAnswer, get_generator
from leyes_rag.indexing.sparse_index import load_bm25_index
from leyes_rag.indexing.vector_store import load_embedder
from leyes_rag.retrieval.pipeline import RetrievalPipeline, RetrievalTrace


@dataclass
class QueryResult:
    trace: RetrievalTrace
    answer: GeneratedAnswer


class LeyesRAGService:
    def __init__(self, mode: str | None = None):
        self.mode = mode or SETTINGS.mode
        self.embedder = load_embedder()
        self.bm25_index = load_bm25_index()
        self.pipeline = RetrievalPipeline(self.embedder, self.bm25_index, mode=self.mode)
        self.generator = get_generator(self.mode)

    def ask(self, question: str, top_k_final: int | None = None) -> QueryResult:
        trace = self.pipeline.retrieve(question, top_k_final=top_k_final)
        answer = self.generator.generate(question, trace.final)
        return QueryResult(trace=trace, answer=answer)
