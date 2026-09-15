"""Orquesta la fase de recuperación (query time) tal como la pide la
arquitectura:

  Pregunta del usuario
    -> Ruteo semántico (Qwen3 / KeywordRouterLite)
    -> Búsqueda densa (Chroma, coseno)      \\
    -> Búsqueda dispersa (BM25)              > en paralelo lógico
    -> Fusión RRF
    -> Reranker (BGE FlagReranker / lexical lite)
    -> Contexto top-k (chunks + cita Ley/Artículo)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from leyes_rag.config import SETTINGS
from leyes_rag.indexing.embeddings import Embedder
from leyes_rag.indexing.sparse_index import BM25Index
from leyes_rag.indexing.vector_store import dense_search
from leyes_rag.retrieval.fusion import reciprocal_rank_fusion
from leyes_rag.retrieval.reranker import Reranker, get_reranker
from leyes_rag.retrieval.router import Router, RouteResult, get_router


@dataclass
class RetrievedChunk:
    chunk_id: str
    citation: str
    text: str
    document: str
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    rerank_score: float | None
    metadata: dict


@dataclass
class RetrievalTrace:
    query: str
    route: RouteResult
    dense_results: list[dict]
    sparse_results: list[dict]
    fused: list[dict]
    final: list[RetrievedChunk]


class RetrievalPipeline:
    def __init__(self, embedder: Embedder, bm25_index: BM25Index, mode: str | None = None):
        self.mode = mode or SETTINGS.mode
        self.embedder = embedder
        self.bm25_index = bm25_index
        self.router: Router = get_router(self.mode)
        self.reranker: Reranker = get_reranker(self.mode)
        self.cfg = SETTINGS.retrieval

    def _filter_by_route(self, results: list[dict], law_ids: list[str]) -> list[dict]:
        if not law_ids:
            return results
        filtered = [r for r in results if r["metadata"].get("law_id") in law_ids]
        # Si el filtro deja muy pocos resultados (router demasiado
        # agresivo), no lo forzamos: mejor mostrar algo de todo el corpus
        # que dejar al usuario sin respuesta.
        return filtered if len(filtered) >= 3 else results

    def retrieve(self, query: str, top_k_final: int | None = None) -> RetrievalTrace:
        top_k_final = top_k_final or self.cfg.top_k_final

        route = self.router.route(query)

        dense_results = dense_search(query, self.embedder, self.cfg.top_k_dense)
        sparse_results = self.bm25_index.search(query, self.cfg.top_k_sparse)

        dense_results = self._filter_by_route(dense_results, route.law_ids)
        sparse_results = self._filter_by_route(sparse_results, route.law_ids)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results], k=self.cfg.rrf_k, top_k=self.cfg.top_k_fused
        )

        reranked = self.reranker.rerank(query, fused, top_k=top_k_final)

        dense_rank_by_id = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
        sparse_rank_by_id = {r["chunk_id"]: i + 1 for i, r in enumerate(sparse_results)}

        final = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                citation=(r.get("metadata") or {}).get("citation", ""),
                text=r.get("document") or "",
                document=r.get("document") or "",
                dense_rank=dense_rank_by_id.get(r["chunk_id"]),
                sparse_rank=sparse_rank_by_id.get(r["chunk_id"]),
                rrf_score=r.get("rrf_score", 0.0),
                rerank_score=r.get("rerank_score"),
                metadata=r.get("metadata") or {},
            )
            for r in reranked
        ]

        return RetrievalTrace(
            query=query,
            route=route,
            dense_results=dense_results,
            sparse_results=sparse_results,
            fused=fused,
            final=final,
        )
