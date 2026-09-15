# Framework RAG para interpretación de leyes chilenas

Pipeline: **Ingesta (PDF→Markdown, chunking jerárquico) → Indexación híbrida
(BGE-M3 + BM25) → Recuperación (ruteo semántico → dense+sparse → RRF →
reranker BGE) → Generación con cita `Ley/Artículo`**.

## Estructura

```
leyes_rag/
  config.py                  # único interruptor lite/production + catálogo de leyes
  ingest/
    pdf_extract.py            # limpieza de PDFs BCN (boilerplate, anotaciones marginales)
    chunking.py                # parser jerárquico Capítulo/Título/Párrafo/Artículo/Inciso
  indexing/
    embeddings.py               # Embedder: TFIDFLiteEmbedder | BGEM3Embedder
    vector_store.py              # ChromaDB (denso)
    sparse_index.py               # BM25 (rank_bm25) + stemming español
  retrieval/
    router.py                      # Router: KeywordRouterLite | QwenRouterProduction
    fusion.py                       # Reciprocal Rank Fusion
    reranker.py                      # Reranker: LexicalRerankerLite | BGERerankerProduction
    pipeline.py                       # orquesta: ruteo -> dense+sparse -> RRF -> rerank
  generation/
    generator.py                       # Generator: ExtractiveGeneratorLite | QwenGeneratorProduction
  rag_service.py                        # fachada: LeyesRAGService.ask(pregunta)
  nlp_utils.py                           # tokenización + stemming español compartido

scripts/01_ingest_and_index.py    # CLI: ingesta + indexación completa (reconstruye todo desde los PDF)
scripts/02_query_cli.py            # CLI: solo consultas, carga los índices ya existentes (no re-procesa PDFs)
notebooks/01_pipeline_demo.ipynb  # demo end-to-end + simulación de consultas PYME
data/                              # markdown/, chroma_db/, bm25.pkl, chunks.jsonl (generados)
raw_leyes/Leyes/                    # los 6 PDF originales
```

## Modo lite vs. producción

Un solo interruptor (`RAG_MODE`, en `config.py` o variable de entorno) decide
qué implementación concreta se inyecta en cada interfaz — el resto del
pipeline no cambia:

| Componente | lite (por defecto, sin GPU/Internet) | production |
|---|---|---|
| Embeddings densos | TF-IDF + SVD (scikit-learn) | BAAI/bge-m3 |
| Reranker | heurística léxica (Jaccard + N° artículo) | BAAI/bge-reranker-v2-m3 |
| Router semántico | TF-IDF vs. catálogo de leyes | Qwen3 |
| Generación | extractiva (cita el chunk tal cual) | Qwen3 |
| BM25 / ChromaDB | igual en ambos modos | igual en ambos modos |

```bash
# Lite (por defecto)
pip install -r requirements-lite.txt
python scripts/01_ingest_and_index.py

# Producción (requiere GPU/Internet para descargar modelos de HuggingFace)
pip install -r requirements-lite.txt -r requirements-production.txt
RAG_MODE=production python scripts/01_ingest_and_index.py
```

## Uso rápido

**Primera vez** (o cuando cambien las leyes fuente): construye los índices desde los PDF.

```bash
python scripts/01_ingest_and_index.py
```

**Para consultar después** (rápido, no vuelve a tocar los PDF ni a chunkear —
solo carga `data/chroma_db/`, `data/bm25.pkl` y `data/embedder_state.pkl`
ya existentes):

```bash
python scripts/02_query_cli.py                                    # modo interactivo
python scripts/02_query_cli.py "¿qué es una empresa de menor tamaño?"  # una sola consulta
```

O desde código / notebook:

```python
from leyes_rag.rag_service import LeyesRAGService

svc = LeyesRAGService()  # carga los índices ya existentes, NO reprocesa los PDF
result = svc.ask("¿Qué debo hacer si un cliente pide eliminar sus datos personales?")
print(result.answer.answer)
for c in result.answer.citations:
    print(" -", c)
```

O abre `notebooks/01_pipeline_demo.ipynb` para el recorrido completo con
trazas paso a paso y preguntas simuladas de clientes PYME (las celdas de
ingesta/indexación de ese notebook sí reconstruyen los índices cada vez que
las corres — para consultas repetidas usa `LeyesRAGService()` o
`02_query_cli.py` como arriba).

## Limitaciones conocidas del modo lite

- **Sin comprensión semántica real**: TF-IDF no entiende sinónimos que no
  comparten raíz léxica (ej. "borrar" vs. "suprimir"). Se mitiga con
  stemming en español, pero no reemplaza a un embedding entrenado como
  BGE-M3.
- **Reranker heurístico**: no es un cross-encoder, así que en artículos muy
  largos (muchos incisos concatenados) puede subestimar su relevancia por
  dilución del solape léxico.
- **Generación extractiva**: no redacta, solo muestra el texto de los
  chunks recuperados con su cita. Es 100% fiel al texto legal pero menos
  natural que una respuesta de Qwen3.
- **Catálogo de 6 leyes**: ampliar el corpus es tan simple como agregar
  entradas a `LAW_CATALOG` en `config.py` con el PDF correspondiente.
