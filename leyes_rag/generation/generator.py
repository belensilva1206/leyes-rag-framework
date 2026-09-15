"""Generación de la respuesta final: toma la pregunta del usuario + los
chunks recuperados (ya rerankeados, con su cita Ley/Artículo/Inciso) y
redacta una respuesta en lenguaje natural que SIEMPRE cita la fuente legal
exacta, para que una PYME pueda verificarla.

- QwenGeneratorProduction: implementación pedida en la arquitectura (Qwen3
  vía HuggingFace transformers u Ollama), con un prompt que fuerza citar
  Ley/Artículo y prohíbe inventar contenido fuera del contexto entregado.
- ExtractiveGeneratorLite: sin LLM. Compone la respuesta citando y
  presentando directamente el texto de los chunks top-k (fiel al 100% al
  texto legal, porque no "redacta" nada, pero menos fluido). Sirve para
  probar el pipeline completo de punta a punta sin depender de un modelo
  de lenguaje pesado.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from leyes_rag.config import SETTINGS
from leyes_rag.retrieval.pipeline import RetrievedChunk

_QWEN_GENERATION_SYSTEM_PROMPT = """Eres un asistente legal que ayuda a PYMES \
chilenas a entender leyes. Responde la pregunta del usuario USANDO SOLO la \
información del CONTEXTO LEGAL entregado abajo. Reglas estrictas:

1. Cita SIEMPRE la ley y el artículo exacto de donde sacas cada afirmación, \
   con el formato "(Ley X, Artículo Y)".
2. Si el contexto no contiene la respuesta, dilo explícitamente: no \
   inventes ni completes con conocimiento general.
3. Usa lenguaje simple, dirigido a una PYME sin formación legal, pero sin \
   perder precisión jurídica.
4. Si hay ambigüedad o la respuesta depende del caso concreto, dilo y \
   sugiere consultar a un abogado o al organismo fiscalizador competente.

CONTEXTO LEGAL:
{context}

PREGUNTA: {question}

RESPUESTA:"""


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[str]
    used_chunks: list[RetrievedChunk]


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        ...


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[Fuente {i} — {c.citation}]\n{c.text}")
    return "\n\n".join(blocks)


class ExtractiveGeneratorLite(Generator):
    """Sin LLM: compone la respuesta mostrando, en orden de relevancia, los
    chunks recuperados con su cita. Es 100% fiel al texto legal (no
    parafrasea, así que no puede "alucinar"), a costa de ser menos fluido
    que una respuesta redactada por un LLM."""

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer=(
                    "No encontré artículos relevantes en las leyes indexadas para "
                    "responder esta pregunta. Te recomiendo reformularla o "
                    "consultar directamente la ley aplicable."
                ),
                citations=[],
                used_chunks=[],
            )

        lines = [
            "Esto es lo que encontré en las leyes indexadas relacionado con tu "
            "pregunta (modo LITE: extracto textual sin redacción por LLM; "
            "verifica siempre citando la fuente):",
            "",
        ]
        citations = []
        for c in chunks:
            body = c.text.split("\n", 1)[-1] if "\n" in c.text else c.text
            lines.append(f"• ({c.citation}) {body.strip()}")
            citations.append(c.citation)
        return GeneratedAnswer(answer="\n\n".join(lines), citations=citations, used_chunks=chunks)


class QwenGeneratorProduction(Generator):
    """Implementación de PRODUCCIÓN: Qwen3 vía HuggingFace transformers u
    Ollama (según ModelConfig.llm_backend), con el mismo prompt "grounded"
    usado por el router. Requiere descargar los pesos del modelo (o un
    servidor Ollama/vLLM corriendo)."""

    def __init__(self):
        self.model_name = SETTINGS.models.generator_llm
        self.backend = SETTINGS.models.llm_backend
        self._pipe = None

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
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["response"]

        self._ensure_loaded()
        out = self._pipe(prompt, max_new_tokens=600, do_sample=False)
        return out[0]["generated_text"][len(prompt):]

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer="No encontré artículos relevantes para responder esta pregunta.",
                citations=[],
                used_chunks=[],
            )
        context = _format_context(chunks)
        prompt = _QWEN_GENERATION_SYSTEM_PROMPT.format(context=context, question=question)
        answer = self._call_llm(prompt).strip()
        return GeneratedAnswer(
            answer=answer, citations=[c.citation for c in chunks], used_chunks=chunks
        )


def get_generator(mode: str) -> Generator:
    if mode == "production":
        return QwenGeneratorProduction()
    return ExtractiveGeneratorLite()
