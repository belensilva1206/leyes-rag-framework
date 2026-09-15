"""Índice vectorial (búsqueda densa) sobre ChromaDB.

Usamos ChromaDB en modo "trae tu propio embedding": nosotros calculamos los
vectores con el Embedder que corresponda al modo activo (TF-IDF lite o
BGE-M3 producción) y solo le pedimos a Chroma que los guarde y haga la
búsqueda por coseno. Así el mismo código de indexación/consulta sirve para
ambos modos sin condicionales.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import chromadb

from leyes_rag.config import CHROMA_COLLECTION, CHROMA_DIR, PROJECT_ROOT, SETTINGS
from leyes_rag.indexing.embeddings import Embedder, get_embedder
from leyes_rag.ingest.chunking import LawChunk

_EMBEDDER_STATE_PATH = PROJECT_ROOT / "data" / "embedder_state.pkl"


def _chunk_metadata(c: LawChunk) -> dict:
    # Chroma solo acepta valores escalares (str/int/float/bool) en metadata.
    return {
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


def build_vector_store(chunks: list[LawChunk]) -> Embedder:
    """Indexa todos los chunks en ChromaDB usando el Embedder del modo
    activo (SETTINGS.mode). Devuelve el embedder ya entrenado/cargado para
    que el pipeline de consulta lo reutilice sin recalcular nada."""
    embedder = get_embedder(SETTINGS.mode)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    ids = [c.chunk_id for c in chunks]
    metadatas = [_chunk_metadata(c) for c in chunks]
    documents = texts

    batch = 200
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i : i + batch],
            embeddings=vectors[i : i + batch].tolist(),
            metadatas=metadatas[i : i + batch],
            documents=documents[i : i + batch],
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embedder.save(_EMBEDDER_STATE_PATH)
    return embedder


def load_embedder() -> Embedder:
    embedder = get_embedder(SETTINGS.mode)
    embedder.load(_EMBEDDER_STATE_PATH)
    return embedder


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


def dense_search(query: str, embedder: Embedder, top_k: int) -> list[dict]:
    """Búsqueda densa por coseno. Devuelve una lista de dicts
    {chunk_id, score, metadata, document} ordenados por similitud
    descendente (score = 1 - distancia coseno, en [0, 1] aprox.)."""
    collection = get_collection()
    query_vec = embedder.encode_query(query)
    res = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    out = []
    for chunk_id, meta, doc, dist in zip(
        res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]
    ):
        out.append(
            {
                "chunk_id": chunk_id,
                "score": 1.0 - dist,
                "metadata": meta,
                "document": doc,
            }
        )
    return out
