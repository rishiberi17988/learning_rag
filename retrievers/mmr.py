"""MMR (Maximal Marginal Relevance): relevant *and* varied results.

The problem it solves
---------------------
Plain similarity search often returns several documents that say almost the
same thing. If you only pass 3 documents to an LLM, 3 near-duplicates waste
the space.

How it works
------------
1. Fetch a larger pool of candidates by similarity (``fetch_k``, e.g. 20).
2. Pick the best match first.
3. For each next pick, prefer documents that are relevant to the question
   but *different* from what was already picked.
4. Stop after ``k`` documents.

``lambda_mult`` controls the balance:
    1.0 -> pure relevance (same as similarity search)
    0.0 -> maximum variety
    0.5 -> a balance of both (a good starting point)
"""

from typing import Any

from ._utils import to_langchain_store


def build_mmr_retriever(store: Any, *, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> Any:
    """Return a retriever that uses MMR to pick ``k`` diverse, relevant documents."""
    return to_langchain_store(store).as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,  # Documents finally returned.
            "fetch_k": fetch_k,  # Candidates considered before diversifying.
            "lambda_mult": lambda_mult,  # Relevance (1.0) vs variety (0.0).
        },
    )

