"""Similarity search: the default way to retrieve from a vector store.

How it works
------------
1. Your question is turned into a vector with the same embedding model that
   was used for the documents.
2. The store finds the ``k`` stored vectors closest to it.
3. Those ``k`` documents are returned, best match first.

Strengths: simple, fast, understands meaning ("split" matches "chunking").
Weakness: the top results can be near-duplicates of each other, and it can
miss exact keywords such as product codes or names (see ``mmr.py`` and
``bm25.py`` / ``hybrid.py`` for fixes).
"""

from typing import Any

from ._utils import to_langchain_store


def build_similarity_retriever(store: Any, *, k: int = 4) -> Any:
    """Return a retriever that fetches the ``k`` most similar documents.

    ``store`` can be one of our ``vector_stores`` classes or a LangChain
    vector store. Call ``.invoke("your question")`` on the result.
    """
    return to_langchain_store(store).as_retriever(
        search_type="similarity",  # Plain "nearest vectors" search.
        search_kwargs={"k": k},  # How many documents to return.
    )

