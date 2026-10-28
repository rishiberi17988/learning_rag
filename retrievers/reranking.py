"""Reranking: fast first-stage retrieval followed by accurate second-stage scoring.

The problem it solves
---------------------
Embedding (bi-encoder) search is fast because documents and queries are
embedded *separately* ahead of time, so searching is just a dot product.
The catch is that the embedding model never sees the query and the document
*together*, so subtle relevance signals can be missed and the best document
might land at rank 4 or 5 instead of rank 1.

How two-stage reranking works
-----------------------------
1. **Stage 1 (Recall):** Fetch a wider candidate pool (``fetch_k``, e.g. 8)
   using a fast retriever (vector similarity or hybrid).
2. **Stage 2 (Precision):** A **reranker** examines each ``(query, document)``
   pair directly, assigns a relevance score, sorts the candidates from best
   to worst, and keeps the top ``top_n``.

Two reranker modes are provided here:
* ``mode="cross_encoder"`` (default): uses a local Hugging Face cross-encoder
  (``cross-encoder/ms-marco-MiniLM-L-6-v2``) via ``HuggingFaceCrossEncoder``
  and ``CrossEncoderReranker`` wrapped in ``ContextualCompressionRetriever``.
* ``mode="llm"``: uses Gemini to score each candidate's relevance (0-10) in a
  single prompt and reorders them — works with zero extra model downloads.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.prompts import ChatPromptTemplate

from ._utils import build_gemini_llm
from .similarity import build_similarity_retriever

LLM_RERANK_PROMPT = ChatPromptTemplate.from_template(
    "You are a relevance ranker. Rate how well each candidate passage answers "
    "the question on a scale from 0 (completely irrelevant) to 10 (directly answers).\n\n"
    "Question: {question}\n\n"
    "Passages:\n{passages}\n\n"
    "Reply with ONLY one line per passage in the exact format:\n"
    "0: <score>\n1: <score>\n..."
)


class LLMReranker(BaseDocumentCompressor):
    """Document compressor that scores and reorders candidates using an LLM."""

    llm: Any
    top_n: int = 3

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        docs = list(documents)
        if not docs:
            return []

        numbered = "\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs))
        response = self.llm.invoke(LLM_RERANK_PROMPT.format_messages(question=query, passages=numbered))
        content = response.content
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)

        scores: dict[int, float] = {}
        for line in str(content).splitlines():
            match = re.search(r"(\d+)\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                idx, val = int(match.group(1)), float(match.group(2))
                if 0 <= idx < len(docs):
                    scores[idx] = val

        # Sort by LLM score (descending); keep original rank order on ties.
        ranked_indices = sorted(range(len(docs)), key=lambda i: (scores.get(i, 0.0), -i), reverse=True)
        reranked: list[Document] = []
        for idx in ranked_indices[: self.top_n]:
            doc = docs[idx]
            copied = Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": scores.get(idx, 0.0)})
            reranked.append(copied)
        return reranked


def build_reranking_retriever(
    store: Any,
    *,
    mode: str = "llm",
    top_n: int = 3,
    fetch_k: int = 6,
    llm: Any = None,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Any:
    """Return a two-stage retriever that fetches ``fetch_k`` docs and reranks to ``top_n``.

    * ``mode="llm"`` (default): uses Gemini to score and reorder candidates
      (no extra model download required).
    * ``mode="cross_encoder"``: uses a local Hugging Face cross-encoder
      (downloads ``cross-encoder/ms-marco-MiniLM-L-6-v2`` on first run).
    """
    from langchain_classic.retrievers import ContextualCompressionRetriever

    base_retriever = build_similarity_retriever(store, k=fetch_k)

    if mode == "llm":
        compressor: BaseDocumentCompressor = LLMReranker(llm=llm or build_gemini_llm(), top_n=top_n)
    elif mode == "cross_encoder":
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        model = HuggingFaceCrossEncoder(model_name=cross_encoder_model)
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
    else:
        raise ValueError("mode must be 'llm' or 'cross_encoder'.")

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

