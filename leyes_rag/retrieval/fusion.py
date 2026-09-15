"""Reciprocal Rank Fusion (RRF): combina el ranking denso (Chroma/coseno) y
el ranking disperso (BM25) en uno solo, sin necesidad de normalizar sus
escalas de score (que no son comparables entre sí: coseno en [0,1] vs.
BM25 sin cota superior).

score_RRF(d) = sum_r 1 / (k + rank_r(d))

donde rank_r(d) es la posición (1-indexada) del documento d en el ranking
r, y k es una constante (por defecto 60, el valor estándar de la
literatura de RRF) que amortigua el peso de las posiciones muy altas.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """ranked_lists: cada elemento es una lista de resultados (dicts con al
    menos 'chunk_id'), ya ordenada de más a menos relevante por esa fuente.

    Devuelve una lista fusionada de dicts con 'chunk_id', 'rrf_score', y
    'sources' (detalle de rank/score que aportó cada lista), ordenada por
    rrf_score descendente.
    """
    fused: dict[str, dict] = {}

    for list_idx, results in enumerate(ranked_lists):
        for rank, item in enumerate(results, start=1):
            cid = item["chunk_id"]
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "rrf_score": 0.0,
                    "metadata": item.get("metadata"),
                    "document": item.get("document"),
                    "sources": {},
                }
            fused[cid]["rrf_score"] += 1.0 / (k + rank)
            fused[cid]["sources"][f"list_{list_idx}"] = {
                "rank": rank,
                "raw_score": item.get("score"),
            }

    out = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out
