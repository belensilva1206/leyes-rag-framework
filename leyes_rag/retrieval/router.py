"""Ruteo semántico: antes de buscar, se decide a qué ley(es) del catálogo
pertenece la pregunta del usuario, para poder filtrar/priorizar la
recuperación (una PYME que pregunta por boletas y garantías no debería
competir en el índice con artículos de la Constitución sobre nacionalidad).

- QwenRouterProduction: implementación pedida en la arquitectura (Qwen3 vía
  HuggingFace transformers u Ollama), clasifica con un prompt estructurado.
- KeywordRouterLite: stand-in local sin LLM, por solapamiento léxico
  TF-IDF entre la pregunta y la descripción temática de cada ley en
  config.LAW_CATALOG. Si ninguna ley destaca con claridad, no filtra
  (deja que la búsqueda híbrida compita en todo el corpus) — así una
  pregunta ambigua nunca se queda sin resultados por un ruteo equivocado.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from leyes_rag.config import LAW_CATALOG, SETTINGS


@dataclass
class RouteResult:
    law_ids: list[str]  # leyes candidatas (vacío = sin filtro, buscar en todo)
    reasoning: str  # explicación breve, útil para debug/demo con el cliente


class Router(ABC):
    @abstractmethod
    def route(self, query: str) -> RouteResult:
        ...


class KeywordRouterLite(Router):
    """Router léxico: sin LLM, sin descargas. Compara la pregunta contra la
    descripción temática de cada ley usando TF-IDF + coseno, ya que ese
    vectorizador no requiere descargar ningún modelo."""

    def __init__(self, min_score: float = 0.08, max_laws: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer

        from leyes_rag.nlp_utils import tokenize

        self.min_score = min_score
        self.max_laws = max_laws
        corpus = [f"{law['nombre_corto']} {law['tema']}" for law in LAW_CATALOG]
        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2))
        self._law_matrix = self._vectorizer.fit_transform(corpus)
        self._law_ids = [law["id"] for law in LAW_CATALOG]

    def route(self, query: str) -> RouteResult:
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._law_matrix)[0]
        ranked = sorted(zip(self._law_ids, sims), key=lambda x: x[1], reverse=True)

        picked = [(lid, s) for lid, s in ranked if s >= self.min_score][: self.max_laws]
        if not picked:
            return RouteResult(
                law_ids=[],
                reasoning=(
                    "Sin coincidencia temática clara (score < "
                    f"{self.min_score}); se busca en todo el corpus."
                ),
            )
        names = {law["id"]: law["nombre_corto"] for law in LAW_CATALOG}
        detail = ", ".join(f"{names[lid]} ({s:.2f})" for lid, s in picked)
        return RouteResult(law_ids=[lid for lid, _ in picked], reasoning=f"Router léxico -> {detail}")


_QWEN_ROUTER_SYSTEM_PROMPT = """Eres un router semántico para un sistema RAG legal chileno.
Dada la pregunta de un usuario (típicamente una PYME), decide a cuál(es) de
las siguientes leyes pertenece, usando SOLO los ids del catálogo. Si la
pregunta es transversal o no calza claramente con ninguna, devuelve una
lista vacía.

Catálogo:
{catalog}

Responde SOLO un JSON con este formato, sin texto adicional:
{{"law_ids": ["id1", "id2"], "reasoning": "..."}}
"""


class QwenRouterProduction(Router):
    """Implementación de PRODUCCIÓN: usa Qwen3 (HuggingFace transformers u
    Ollama, según ModelConfig.llm_backend) para clasificar la pregunta.
    Requiere descargar los pesos del modelo (o tener un servidor Ollama/vLLM
    corriendo), por lo que no funciona en un entorno sin acceso a Internet
    para modelos."""

    def __init__(self):
        self.model_name = SETTINGS.models.router_llm
        self.backend = SETTINGS.models.llm_backend
        self._pipe = None

    def _catalog_str(self) -> str:
        return "\n".join(
            f"- id=\"{law['id']}\" | {law['nombre_corto']} | tema: {law['tema']}"
            for law in LAW_CATALOG
        )

    def _ensure_loaded(self):
        if self._pipe is not None or self.backend != "hf":
            return
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "Falta 'transformers'. Instala con: pip install transformers torch accelerate\n"
                "Además necesitas acceso a Internet a huggingface.co para descargar "
                f"los pesos de {self.model_name} la primera vez."
            ) from e
        self._pipe = pipeline("text-generation", model=self.model_name, device_map="auto")

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "ollama":
            import requests

            resp = requests.post(
                f"{SETTINGS.models.ollama_host}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"]

        self._ensure_loaded()
        out = self._pipe(prompt, max_new_tokens=200, do_sample=False)
        return out[0]["generated_text"][len(prompt):]

    def route(self, query: str) -> RouteResult:
        prompt = _QWEN_ROUTER_SYSTEM_PROMPT.format(catalog=self._catalog_str())
        prompt += f"\nPregunta del usuario: {query}\n"
        raw = self._call_llm(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return RouteResult(law_ids=[], reasoning=f"Qwen3 no devolvió JSON válido: {raw[:200]}")
        try:
            data = json.loads(match.group())
            valid_ids = {law["id"] for law in LAW_CATALOG}
            law_ids = [lid for lid in data.get("law_ids", []) if lid in valid_ids]
            return RouteResult(law_ids=law_ids, reasoning=data.get("reasoning", ""))
        except json.JSONDecodeError:
            return RouteResult(law_ids=[], reasoning=f"JSON inválido de Qwen3: {raw[:200]}")


def get_router(mode: str) -> Router:
    if mode == "production":
        return QwenRouterProduction()
    return KeywordRouterLite()
