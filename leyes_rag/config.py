"""Configuración central del framework RAG de interpretación de leyes.

Un único interruptor (RAG_MODE) decide qué implementaciones concretas se
inyectan en el pipeline:

- "lite": todo corre localmente con librerías 100% pip-instalables, sin
  descargar pesos de modelos ni llamar APIs externas. Sirve para desarrollar
  y probar el pipeline completo (ingesta -> chunking -> índice híbrido ->
  ruteo -> fusión RRF -> rerank -> generación) en entornos sin acceso a
  Internet para modelos (p. ej. este sandbox).
- "production": usa los modelos reales pedidos en la arquitectura
  (BGE-M3, BM25, Qwen3 para ruteo/generación, BGE FlagReranker). Requiere
  GPU/CPU con acceso a HuggingFace (o un servidor Ollama/vLLM local) y más
  RAM. Se activa cambiando esta variable o la env var RAG_MODE=production.

Todo el resto del código depende de interfaces abstractas (ver
indexing/embeddings.py, retrieval/router.py, retrieval/reranker.py,
generation/generator.py), así que cambiar de modo no requiere tocar el
pipeline, solo esta config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_LEYES_DIR = PROJECT_ROOT / "raw_leyes" / "Leyes"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
CHROMA_COLLECTION = "leyes_chile"

# "lite" (default, corre en cualquier máquina sin GPU/Internet) | "production"
RAG_MODE = os.environ.get("RAG_MODE", "lite")


@dataclass
class ChunkingConfig:
    # Un artículo completo se guarda como un solo chunk si cabe en este
    # límite; si es más largo, se subdivide por inciso (jerarquía real de
    # la ley), cada sub-chunk con el encabezado Ley/Título/Artículo repetido
    # para que sea recuperable de forma autónoma.
    max_chars_per_chunk: int = 1200
    min_chars_to_split_by_inciso: int = 1200


@dataclass
class RetrievalConfig:
    top_k_dense: int = 15
    top_k_sparse: int = 15
    rrf_k: int = 60  # constante estándar de Reciprocal Rank Fusion
    top_k_fused: int = 12
    top_k_final: int = 5  # tras el reranker, lo que se pasa al generador


@dataclass
class ModelConfig:
    # --- producción ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Variante "Instruct-2507": no-thinking (no gasta tokens en razonamiento
    # largo antes de responder), ideal para clasificación JSON del router y
    # para generación grounded de respuestas cortas con cita.
    router_llm: str = "Qwen/Qwen3-4B-Instruct-2507"
    generator_llm: str = "Qwen/Qwen3-4B-Instruct-2507"
    # Si tienes un servidor Ollama/vLLM corriendo, apunta aquí en vez de
    # cargar los pesos en el propio proceso Python.
    llm_backend: str = os.environ.get("LLM_BACKEND", "hf")  # "hf" | "ollama" | "vllm"
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class RagSettings:
    mode: str = RAG_MODE
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    models: ModelConfig = field(default_factory=ModelConfig)


SETTINGS = RagSettings()

# Categorías usadas por el router semántico para clasificar la pregunta del
# usuario antes de restringir/priorizar la búsqueda a una o más leyes.
LAW_CATALOG = [
    {
        "id": "proteccion_datos_2024",
        "file": "Ley-21719_Proteccion_Datos.pdf",
        "nombre_corto": "Ley 21.719 - Protección de Datos Personales",
        "numero": "21.719",
        "tema": (
            "Protección y tratamiento de datos personales, Agencia de Protección de Datos "
            "Personales, sanciones y multas por mal uso o filtración de datos. Derechos ARCO: "
            "acceso, rectificación, cancelación/supresión (eliminar o borrar los datos de un "
            "cliente), oposición y portabilidad. Consentimiento para usar datos de clientes en "
            "marketing, correos publicitarios o campañas comerciales. Brechas de seguridad."
        ),
    },
    {
        "id": "vida_privada_1999",
        "file": "LEY-19628_Proteccion_Vida_Privada.pdf",
        "nombre_corto": "Ley 19.628 - Protección de la Vida Privada",
        "numero": "19.628",
        "tema": (
            "Ley histórica de datos de carácter personal (antecesora de la 21.719), vida "
            "privada, bancos de datos, registro de información comercial (DICOM/boletín "
            "comercial), datos de morosidad y deudas."
        ),
    },
    {
        "id": "pymes_2010",
        "file": "Ley-20416_Especial_Pymes.pdf",
        "nombre_corto": "Ley 20.416 - Estatuto PYME",
        "numero": "20.416",
        "tema": (
            "Normas especiales para empresas de menor tamaño (microempresa, pequeña y mediana "
            "empresa): cómo se clasifica una PYME según sus ventas anuales, plazos de pago "
            "entre empresas, multas y fiscalización laboral graduadas para PYMES, acceso a "
            "compras públicas (Mercado Público), beneficios y trámites simplificados de "
            "inicio, funcionamiento y cierre de una empresa."
        ),
    },
    {
        "id": "acceso_info_publica_2008",
        "file": "LEY-20285_Acceso_Informacion_Publica.pdf",
        "nombre_corto": "Ley 20.285 - Acceso a la Información Pública",
        "numero": "20.285",
        "tema": (
            "Transparencia de la función pública, derecho de acceso a información de "
            "organismos del Estado, municipalidades y servicios públicos. Solicitudes de "
            "información, licitaciones y adjudicaciones de compras públicas, transparencia "
            "activa, Consejo para la Transparencia."
        ),
    },
    {
        "id": "consumidores_1997",
        "file": "Ley-19496_Proteccion_Derechos_Consumidores.pdf",
        "nombre_corto": "Ley 19.496 - Protección de los Derechos de los Consumidores",
        "numero": "19.496",
        "tema": (
            "Derechos y deberes de consumidores y proveedores, clientes y tiendas. Garantía "
            "legal de productos y servicios, devolución, cambio o reembolso de un producto, "
            "producto fallado o con desperfectos, derecho a retracto, boletas y facturas, "
            "publicidad engañosa, promociones y ofertas, contratos de adhesión, cobranza "
            "extrajudicial, reclamos ante el SERNAC."
        ),
    },
    {
        "id": "constitucion_2005",
        "file": "DTO-100_Constitucion_Politica_Republica.pdf",
        "nombre_corto": "Constitución Política de la República",
        "numero": "DTO-100",
        "tema": (
            "Bases de la institucionalidad, derechos y deberes constitucionales fundamentales, "
            "organización del Estado. Libertad para desarrollar y emprender cualquier "
            "actividad económica, derecho de propiedad privada, igualdad ante la ley, recurso "
            "de protección."
        ),
    },
]
