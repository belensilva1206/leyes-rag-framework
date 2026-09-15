"""Reranker: re-ordena el conjunto fusionado (RRF) de candidatos con una
señal más cara pero más precisa que un cross-encoder (denso+disperso solo
comparan vectores/tokens por separado; el reranker mira la pregunta y cada
chunk JUNTOS).

- BGERerankerProduction: implementación pedida en la arquitectura
  (BAAI/bge-reranker-v2-m3 vía FlagEmbedding.FlagReranker). Requiere
  descargar los pesos desde HuggingFace.
- LexicalRerankerLite: heurística local sin modelo — combina solapamiento
  de tokens (Jaccard), coincidencia de frase exacta y coincidencia del
  número de artículo/ley mencionado explícitamente en la pregunta (una
  señal muy fuerte en dominio legal: si el usuario pregunta "qué dice el
  artículo 3 de la ley del consumidor", ese chunk debe subir fuerte).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from leyes_rag.indexing.sparse_index import tokenize


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        ...


_ARTICULO_MENTION_RE = re.compile(
    r"art[íi]culo\s+(\d+\s*[°ºªa-z]?|primer[oa]|segund[oa]|tercer[oa]|cuart[oa]|quint[oa])",
    re.IGNORECASE,
)


class LexicalRerankerLite(Reranker):
    """Heurística léxica que SIEMPRE parte del ranking de RRF (denso+disperso
    combinados) y solo lo reordena cuando encuentra una señal léxica fuerte
    (mención explícita del número de artículo, o la frase completa de la
    pregunta calzando textual en el chunk).

    Nota de diseño: una versión anterior calculaba el reranking SOLO con
    Jaccard(pregunta, chunk), y eso producía resultados peores que el
    ranking de RRF de entrada: un chunk corto y genérico que por azar
    comparte 1-2 palabras con la pregunta obtiene un Jaccard artificialmente
    alto (el denominador — la unión de tokens — es pequeño), y así
    desplazaba a un chunk largo y realmente relevante pero con más texto
    "de más". Por eso ahora el puntaje base viene del RANKING de entrada
    (1 / posición en la lista fusionada) y el léxico solo suma bonos
    encima, no reemplaza la señal híbrida.
    """

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        q_tokens = set(tokenize(query))
        q_lower = query.lower()
        mentioned_articles = {
            m.group(1).strip().lower().replace("°", "").replace("º", "")
            for m in _ARTICULO_MENTION_RE.finditer(query)
        }

        scored = []
        for rrf_rank, cand in enumerate(candidates, start=1):
            doc = cand.get("document") or ""
            doc_tokens = set(tokenize(doc))
            if not doc_tokens:
                jaccard = 0.0
            else:
                jaccard = len(q_tokens & doc_tokens) / len(q_tokens | doc_tokens)

            phrase_bonus = 0.3 if len(q_lower) > 12 and q_lower in doc.lower() else 0.0

            article_bonus = 0.0
            meta = cand.get("metadata") or {}
            art = str(meta.get("articulo", "")).lower().replace("°", "").replace("º", "")
            if mentioned_articles and art:
                if any(art == m or art.startswith(m) for m in mentioned_articles):
                    article_bonus = 0.5

            # Base fuerte por posición en RRF (1.0, 0.5, 0.33, 0.25, ...) +
            # bonos léxicos acotados (jaccard pesa poco a propósito: es la
            # señal que más se presta a falsos positivos en chunks cortos).
            rerank_score = (1.0 / rrf_rank) + 0.15 * jaccard + phrase_bonus + article_bonus
            new_cand = dict(cand)
            new_cand["rerank_score"] = rerank_score
            scored.append(new_cand)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


class BGERerankerProduction(Reranker):
    """Implementación de PRODUCCIÓN: BAAI/bge-reranker-v2-m3 (cross-encoder)
    vía FlagEmbedding.FlagReranker. Requiere descargar los pesos desde
    HuggingFace."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = False):
        self.model_name = model_name
        self._model = None
        self._use_fp16 = use_fp16

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            raise ImportError(
                "Falta 'FlagEmbedding'. Instala con: pip install FlagEmbedding torch\n"
                "Además necesitas acceso a Internet a huggingface.co para descargar "
                f"los pesos de {self.model_name} la primera vez."
            ) from e
        self._model = FlagReranker(self.model_name, use_fp16=self._use_fp16)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        self._ensure_loaded()
        pairs = [[query, c.get("document") or ""] for c in candidates]
        scores = self._model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        scored = []
        for cand, score in zip(candidates, scores):
            new_cand = dict(cand)
            new_cand["rerank_score"] = float(score)
            scored.append(new_cand)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


def get_reranker(mode: str) -> Reranker:
    if mode == "production":
        return BGERerankerProduction()
    return LexicalRerankerLite()
