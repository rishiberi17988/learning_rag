"""Ensemble & Hybrid retrieval: combine any set of retrievers with RRF.

Ensemble vs. Hybrid
-------------------
* **EnsembleRetriever** is the general mechanism: it takes *any* list of
  retrievers (e.g. BM25 + Similarity + MMR, or retrievers over two different
  vector stores), runs all of them, and merges their ranked lists using
  weighted **Reciprocal Rank Fusion (RRF)**.
* **Hybrid search** is the most popular special case of an ensemble: combining
  a *sparse* keyword retriever (BM25) with a *dense* vector retriever.

How weighted Reciprocal Rank Fusion works
-----------------------------------------
Each retriever returns an ordered list of documents. A document at rank ``r``
(1-indexed) in retriever ``i`` gets ``weights[i] / (r + c)`` points. Points
from all retrievers are summed and documents are sorted highest-first.
Because only *ranks* are used, retrievers whose raw scores live on incompatible
scales (BM25 scores vs. cosine similarity vs. L2 distance) combine cleanly.

Requires: ``pip install rank_bm25`` (when using ``build_hybrid_retriever``).
"""

from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

from .bm25 import build_bm25_retriever
from .similarity import build_similarity_retriever


def build_ensemble_retriever(
    retrievers: Sequence[Any],
    *,
    weights: Sequence[float] | None = None,
    c: int = 60,
) -> Any:
    """Combine arbitrary ``retrievers`` with weighted Reciprocal Rank Fusion.

    If ``weights`` is omitted, every retriever gets an equal weight.
    """
    from langchain_classic.retrievers import EnsembleRetriever

    retriever_list = list(retrievers)
    if not retriever_list:
        raise ValueError("Pass at least one retriever to build_ensemble_retriever.")
    if weights is None:
        weights = [1.0 / len(retriever_list)] * len(retriever_list)
    return EnsembleRetriever(retrievers=retriever_list, weights=list(weights), c=c)


def build_hybrid_retriever(
    store: Any,
    documents: Iterable[Document],
    *,
    k: int = 4,
    weights: tuple[float, float] = (0.5, 0.5),
) -> Any:
    """Build a classic Hybrid retriever: BM25 keyword search + vector similarity.

    ``documents`` should be the same documents indexed in ``store`` so both
    retrievers search the same corpus. ``weights`` = (BM25 weight, vector weight).
    """
    keyword_retriever = build_bm25_retriever(documents, k=k)
    vector_retriever = build_similarity_retriever(store, k=k)
    return build_ensemble_retriever([keyword_retriever, vector_retriever], weights=weights)
