"""Interfaz de embeddings (búsqueda densa) + dos implementaciones:

- BGEM3Embedder: la implementación de PRODUCCIÓN pedida en la arquitectura
  (BAAI/bge-m3 vía FlagEmbedding, corre en GPU o CPU con más RAM). Requiere
  descargar los pesos desde HuggingFace, así que no funciona en un entorno
  sin acceso a Internet para modelos.
- TFIDFLiteEmbedder: stand-in 100% local (scikit-learn TF-IDF + SVD para
  obtener vectores densos de dimensión fija) que no descarga nada. No tiene
  la calidad semántica de BGE-M3 (no captura sinónimos/paráfrasis), pero
  implementa el mismo contrato y permite probar el pipeline híbrido +
  fusión RRF + rerank end-to-end hoy mismo.

Cualquier código de retrieval que dependa de "Embedder" funciona igual con
cualquiera de las dos clases: eso es lo que permite cambiar de modo lite a
producción sin tocar el resto del pipeline.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Embedder(ABC):
    """Contrato común para el embedder denso del pipeline RAG."""

    name: str
    dim: int

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Ajusta el modelo al corpus si hace falta (no-op en modelos
        preentrenados como BGE-M3; imprescindible en TF-IDF)."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Vectoriza documentos (chunks) para indexarlos."""

    def encode_query(self, text: str) -> np.ndarray:
        """Vectoriza una consulta de usuario. Por defecto usa el mismo
        método que los documentos (simétrico); BGE-M3 es simétrico."""
        return self.encode([text])[0]

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...


class TFIDFLiteEmbedder(Embedder):
    """Embedder 100% local basado en TF-IDF + reducción de dimensionalidad
    (TruncatedSVD), usado en RAG_MODE=lite. No requiere descargar modelos.

    Limitación conocida (documentada para el cliente): al ser léxico en su
    base, no reconoce sinónimos ("boleta" vs "comprobante de compra") tan
    bien como un embedding semántico entrenado como BGE-M3. Por eso el modo
    lite se apoya más en BM25 + el router por palabras clave para compensar.
    """

    def __init__(self, dim: int = 384):
        self.name = "tfidf-lite"
        self.dim = dim
        self._vectorizer = None
        self._svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from leyes_rag.nlp_utils import tokenize

        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
        )
        tfidf = self._vectorizer.fit_transform(texts)
        n_components = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_components = max(n_components, 2)
        self.dim = n_components
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TFIDFLiteEmbedder no ha sido entrenado: llama a fit() primero.")
        tfidf = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self._vectorizer, "svd": self._svd, "dim": self.dim}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._vectorizer = data["vectorizer"]
        self._svd = data["svd"]
        self.dim = data["dim"]


class BGEM3Embedder(Embedder):
    """Implementación de PRODUCCIÓN: BAAI/bge-m3 vía la librería
    FlagEmbedding (`pip install FlagEmbedding`). Necesita descargar los
    pesos desde HuggingFace la primera vez (~2.2 GB) y, sin GPU, es lento
    pero funcional en CPU.

    No requiere fit(): es un modelo preentrenado.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        self.name = model_name
        self.dim = 1024  # dimensión nativa del embedding denso de bge-m3
        self._model = None
        self._use_fp16 = use_fp16

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise ImportError(
                "Falta 'FlagEmbedding'. Instala con: pip install FlagEmbedding torch\n"
                "Además necesitas acceso a Internet a huggingface.co para descargar "
                f"los pesos de {self.name} la primera vez."
            ) from e
        self._model = BGEM3FlagModel(self.name, use_fp16=self._use_fp16)

    def fit(self, texts: list[str]) -> None:
        # bge-m3 es un modelo preentrenado: no requiere ajuste al corpus.
        self._ensure_loaded()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        out = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return np.asarray(out["dense_vecs"])

    def save(self, path: Path) -> None:
        # No hay estado propio que persistir aparte del nombre del modelo
        # (los pesos los cachea HuggingFace en ~/.cache).
        with open(path, "wb") as f:
            pickle.dump({"model_name": self.name}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.name = data["model_name"]


def get_embedder(mode: str) -> Embedder:
    if mode == "production":
        return BGEM3Embedder()
    return TFIDFLiteEmbedder()
