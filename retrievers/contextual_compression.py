"""Contextual compression: retrieve, then trim or drop what isn't relevant.

The problem it solves
---------------------
Retrieved documents often contain only a sentence or two that actually
answer the question. Sending everything to the LLM wastes tokens and can
distract it.

How it works
------------
1. A *base retriever* fetches candidates as usual (similarity search here).
2. A *compressor* post-processes them. Two options are provided:

   * ``mode="embeddings"`` (EmbeddingsFilter): re-embeds each document and
     drops those whose similarity to the question is below
     ``similarity_threshold``. Cheap: no LLM calls.
   * ``mode="llm"`` (LLMChainExtractor): asks an LLM to copy out only the
     parts of each document relevant to the question, and drops documents
     with nothing relevant. More precise, but one LLM call per document.
"""

from typing import Any

from ._utils import build_gemini_llm
from .similarity import build_similarity_retriever


def build_compression_retriever(
    store: Any,
    *,
    mode: str = "embeddings",
    embeddings: Any = None,
    similarity_threshold: float = 0.6,
    llm: Any = None,
    k: int = 4,
) -> Any:
    """Return a retriever whose results are filtered (``"embeddings"``) or trimmed (``"llm"``).

    * ``mode="embeddings"`` needs ``embeddings`` (use the same model as the store).
    * ``mode="llm"`` uses ``llm`` (defaults to Gemini Flash).
    """
    from langchain_classic.retrievers import ContextualCompressionRetriever

    if mode == "embeddings":
        from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

        if embeddings is None:
            raise ValueError("mode='embeddings' needs the embeddings model, e.g. build_gemini_embeddings().")
        # Keeps only documents whose similarity to the question is high enough.
        compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)
    elif mode == "llm":
        from langchain_classic.retrievers.document_compressors import LLMChainExtractor

        # Asks the LLM to extract the relevant sentences from each document.
        compressor = LLMChainExtractor.from_llm(llm or build_gemini_llm())
    else:
        raise ValueError("mode must be 'embeddings' or 'llm'.")

    return ContextualCompressionRetriever(
        base_compressor=compressor,  # Step 2: filter/trim.
        base_retriever=build_similarity_retriever(store, k=k),  # Step 1: fetch candidates.
    )

