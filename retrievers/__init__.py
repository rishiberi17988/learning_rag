"""Builders for the common retrieval strategies, one strategy per module.

Every ``build_*`` function returns a standard LangChain retriever, so they
are all used the same way::

    docs = retriever.invoke("your question")

Vector-store-based builders accept either one of our ``vector_stores``
classes (FAISSStore, ChromaStore, ...) or a plain LangChain vector store.
"""

from ._utils import build_gemini_llm  # Gemini chat model used by LLM-assisted retrievers.
from .bm25 import build_bm25_retriever  # Sparse keyword search, no embeddings.
from .contextual_compression import build_compression_retriever  # Retrieve, then filter/trim.
from .ensemble import build_ensemble_retriever, build_multi_strategy_ensemble_retriever  # Combine any retrievers.
from .fusion import build_fusion_retriever  # RAG-Fusion: multi-query + Reciprocal Rank Fusion.
from .hybrid import build_hybrid_retriever  # BM25 + vector search merged with RRF.
from .hyde import build_hyde_retriever  # HyDE: embed a hypothetical LLM answer.
from .metadata_filter import build_metadata_filter_retriever  # Search within matching metadata.
from .mmr import build_mmr_retriever  # Relevant but varied results (Maximal Marginal Relevance).
from .multi_query import build_multi_query_retriever  # LLM rephrases the question several ways.
from .parent_document import build_parent_document_retriever  # Match small chunks, return full parents.
from .reranking import build_reranking_retriever  # First-stage recall + LLM/CrossEncoder reranker.
from .score_threshold import build_score_threshold_retriever  # Drop weak matches below a threshold.
from .self_query import build_self_query_retriever  # LLM extracts metadata filters from the query.
from .similarity import build_similarity_retriever  # Plain nearest-neighbour vector search.

__all__ = [
    "build_bm25_retriever",
    "build_compression_retriever",
    "build_ensemble_retriever",
    "build_fusion_retriever",
    "build_gemini_llm",
    "build_hybrid_retriever",
    "build_hyde_retriever",
    "build_metadata_filter_retriever",
    "build_mmr_retriever",
    "build_multi_query_retriever",
    "build_multi_strategy_ensemble_retriever",
    "build_parent_document_retriever",
    "build_reranking_retriever",
    "build_score_threshold_retriever",
    "build_self_query_retriever",
    "build_similarity_retriever",
]
