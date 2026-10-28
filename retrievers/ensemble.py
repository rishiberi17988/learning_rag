"""Ensemble retriever: combine multiple retrievers of any kind using RRF.

While ``hybrid.py`` focuses on the classic 2-way combination of BM25 keyword
search + vector similarity search, an **Ensemble Retriever** can combine *any*
number and type of retrievers — for example:
* BM25 (exact keywords) + Similarity (top semantic matches) + MMR (diverse hits)
* Retrievers backed by different embedding models or different vector stores.

See ``hybrid.py`` for the underlying ``build_ensemble_retriever`` helper.
"""

from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

from .bm25 import build_bm25_retriever
from .hybrid import build_ensemble_retriever
from .mmr import build_mmr_retriever
from .similarity import build_similarity_retriever


def build_multi_strategy_ensemble_retriever(
    store: Any,
    documents: Iterable[Document],
    *,
    k: int = 4,
    weights: Sequence[float] = (0.4, 0.3, 0.3),
) -> Any:
    """Combine three complementary strategies: BM25 + Similarity + MMR."""
    bm25 = build_bm25_retriever(documents, k=k)
    similarity = build_similarity_retriever(store, k=k)
    mmr = build_mmr_retriever(store, k=k, fetch_k=max(k * 3, 8))
    return build_ensemble_retriever([bm25, similarity, mmr], weights=weights)


__all__ = ["build_ensemble_retriever", "build_multi_strategy_ensemble_retriever"]

