"""RAG-Fusion: Multi-Query generation + Reciprocal Rank Fusion (RRF).

The problem it solves
---------------------
Standard Multi-Query retrieval generates alternative phrasings of a question
and simply takes the *union* of all retrieved documents. Because it just
deduplicates them in arbitrary order, a document that weakly matched only one
phrasing can appear ahead of a document that strongly matched all of them.

How RAG-Fusion works
--------------------
1. An LLM generates ``num_queries`` alternative phrasings of the user's
   question (plus we keep the original question).
2. The base retriever runs a search for each query, producing a separate
   ranked list of documents for each one.
3. **Reciprocal Rank Fusion (RRF)** merges those ranked lists:
       score(doc) = sum( 1 / (rank_in_list + c) )
   where ``rank_in_list`` is 1, 2, 3, ... and ``c`` (usually 60) is a constant
   that dampens the impact of outlier rankings.
4. Documents that consistently rank near the top across *multiple* query
   variations bubble to the very top of the final list.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from ._utils import build_gemini_llm
from .similarity import build_similarity_retriever

FUSION_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} different search queries related to: "
    "{question}\n"
    "Output ONLY the queries, one per line, without numbering or bullets."
)


def reciprocal_rank_fusion(
    results: list[list[Document]],
    *,
    c: int = 60,
    k: int = 4,
) -> list[Document]:
    """Merge multiple ranked document lists using Reciprocal Rank Fusion (RRF)."""
    fused_scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for doc_list in results:
        for rank, doc in enumerate(doc_list, start=1):
            # Identify each document by its metadata 'id' if present, else content.
            key = str(doc.metadata.get("id") or doc.page_content)
            docs_by_key[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rank + c)

    # Sort keys from highest RRF score to lowest and return the top ``k`` docs.
    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    return [docs_by_key[key] for key in ranked_keys[:k]]


class RAGFusionRetriever(BaseRetriever):
    """LangChain retriever implementing RAG-Fusion (query expansion + RRF)."""

    base_retriever: Any
    llm: Any
    num_queries: int = 3
    k: int = 4
    rrf_c: int = 60
    last_generated_queries: list[str] = []

    def _generate_queries(self, question: str) -> list[str]:
        """Ask the LLM for alternative search queries and include the original."""
        messages = FUSION_PROMPT.format_messages(question=question, num_queries=self.num_queries)
        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        rewrites = [line.strip(" -*\t0123456789.") for line in str(content).splitlines() if line.strip()]
        queries = [question] + rewrites[: self.num_queries]
        self.last_generated_queries = queries
        return queries

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        queries = self._generate_queries(query)
        # Retrieve a ranked list of documents for each query variation.
        ranked_lists = [self.base_retriever.invoke(q) for q in queries]
        # Combine and re-rank all lists using Reciprocal Rank Fusion.
        return reciprocal_rank_fusion(ranked_lists, c=self.rrf_c, k=self.k)


def build_fusion_retriever(
    store: Any,
    *,
    llm: Any = None,
    k: int = 4,
    fetch_k: int = 6,
    num_queries: int = 3,
) -> RAGFusionRetriever:
    """Build a RAG-Fusion retriever over ``store``.

    * ``fetch_k``: how many candidates to pull per generated query.
    * ``k``: how many fused documents to return at the end.
    * ``num_queries``: how many extra queries the LLM generates.
    """
    return RAGFusionRetriever(
        base_retriever=build_similarity_retriever(store, k=fetch_k),
        llm=llm or build_gemini_llm(),
        num_queries=num_queries,
        k=k,
    )

