"""Multi-query retrieval: let an LLM rephrase the question several ways.

The problem it solves
---------------------
Vector search is sensitive to wording. A question phrased differently from
how the documents are written can miss good matches.

How it works
------------
1. An LLM rewrites your question into several alternative versions (3 by
   default), e.g. "How do I split documents?" might become
   "What is text chunking?" and "Ways to break long texts into passages".
2. Each version is searched with the base retriever.
3. All results are combined and duplicates removed.

Cost: one LLM call plus one search per generated question, so it's slower
than a single search. Tip: turn on logging for
``langchain_classic.retrievers.multi_query`` at INFO level to see the
generated questions.
"""

from typing import Any

from ._utils import build_gemini_llm
from .similarity import build_similarity_retriever


def build_multi_query_retriever(store: Any, *, llm: Any = None, k: int = 4) -> Any:
    """Return a retriever that searches with several LLM-generated rewrites of the question.

    ``llm`` defaults to Gemini Flash (needs ``GOOGLE_API_KEY``).
    """
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    return MultiQueryRetriever.from_llm(
        retriever=build_similarity_retriever(store, k=k),  # Runs each rewritten question.
        llm=llm or build_gemini_llm(),  # Writes the alternative questions.
    )

