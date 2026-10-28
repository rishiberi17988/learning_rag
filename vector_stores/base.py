"""Common methods shared by the provider-specific vector store classes.

How the pieces fit together
---------------------------
A *vector store* saves text chunks together with their *embeddings* (lists of
numbers that capture meaning). When you search, your question is embedded too,
and the store returns the chunks whose vectors are closest to it.

LangChain already ships a class for every provider (FAISS, Chroma, Pinecone,
...). Each provider class in this folder wraps one of those LangChain objects
and stores it in ``self._store``. This base class then forwards the common
calls (add, search, retrieve, delete) to that object, so every provider can be
used with exactly the same method names.
"""

# Lets us write type hints such as ``list[str] | None`` on older Pythons too.
from __future__ import annotations

from typing import Any, Iterable

# ``Document`` = a piece of text (``page_content``) plus a ``metadata`` dict.
from langchain_core.documents import Document


class BaseVectorStore:
    """Small wrapper exposing the usual add, search, and retrieval methods."""

    def __init__(self, embeddings: Any) -> None:
        # The embedding model turns text into vectors. It is used both when
        # documents are added and when a search query comes in.
        self.embeddings = embeddings
        # The real LangChain vector store. Subclasses fill this in, either
        # right away (Chroma, Pinecone, InMemory) or lazily (FAISS).
        self._store: Any = None

    def _ensure_store(self, documents: list[Document] | None = None) -> Any:
        """Return the underlying LangChain store, or fail with a clear message.

        ``documents`` is unused here; FAISS overrides this method and needs
        documents to build its index the first time.
        """
        if self._store is None:
            raise RuntimeError("The vector store is not initialized. Add documents first.")
        return self._store

    def add_documents(self, documents: Iterable[Document], *, ids: list[str] | None = None) -> list[str]:
        """Embed ``documents`` and save them. Returns the ID of each document.

        Pass your own ``ids`` to control them. Adding a document again with the
        same ID overwrites it instead of creating a duplicate.
        """
        docs = list(documents)  # Accept any iterable (list, generator, ...).
        if not docs:
            return []  # Nothing to add, so skip the embedding call.
        return self._ensure_store(docs).add_documents(docs, ids=ids)

    def similarity_search(self, query: str, *, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return the ``k`` documents whose meaning is closest to ``query``."""
        return self._ensure_store().similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(self, query: str, *, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        """Like ``similarity_search`` but also returns a score for each match.

        Note: the meaning of the score depends on the provider.
          * InMemory / Pinecone (cosine): similarity, so HIGHER is better.
          * FAISS / Chroma (L2): distance, so LOWER is better.
        """
        return self._ensure_store().similarity_search_with_score(query, k=k, **kwargs)

    def as_retriever(self, **kwargs: Any) -> Any:
        """Wrap the store as a LangChain *retriever* for use in RAG chains.

        Example: ``store.as_retriever(search_kwargs={"k": 3}).invoke("question")``
        """
        return self._ensure_store().as_retriever(**kwargs)

    def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.

        Caution: some providers (InMemory, Chroma, Pinecone) return ``None``
        even when the delete worked, which shows up here as ``False``. Search
        again if you need to confirm that a document is gone.
        """
        return bool(self._ensure_store().delete(ids=ids))
