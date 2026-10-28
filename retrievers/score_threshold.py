"""Similarity search with a score threshold: only return *good enough* matches.

The problem it solves
---------------------
Similarity search always returns ``k`` documents, even when none of them are
related to the question. Feeding unrelated text to an LLM invites made-up
answers. A threshold lets the retriever return fewer documents, or none.

How it works
------------
Every match gets a *relevance score* between 0 and 1 (higher = more
relevant). LangChain converts each provider's raw score (distance or cosine
similarity) to this common scale, so one threshold works for every store.
Documents below ``score_threshold`` are dropped.

Tip: good thresholds depend on the embedding model and your data. Print the
scores for a few questions first (``store.similarity_search_with_relevance_scores``)
and pick a value between "clearly relevant" and "clearly unrelated".
"""

from typing import Any

from ._utils import to_langchain_store


def build_score_threshold_retriever(store: Any, *, score_threshold: float = 0.6, k: int = 4) -> Any:
    """Return a retriever that keeps at most ``k`` documents scoring >= ``score_threshold``."""
    return to_langchain_store(store).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": score_threshold,  # Minimum relevance (0 to 1).
            "k": k,  # Upper limit on documents returned.
        },
    )

