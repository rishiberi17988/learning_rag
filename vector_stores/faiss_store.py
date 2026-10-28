"""FAISS vector store running locally on CPU.

FAISS (Facebook AI Similarity Search) is a very fast library for finding the
nearest vectors. Two things differ from the other stores:

1. The index is built from the first batch of documents, so the store stays
   empty (``self._store is None``) until you call ``add_documents``.
2. Nothing is saved automatically. Call ``save()`` to write the index to a
   folder and ``FAISSStore.load()`` to read it back.
"""

from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from .base import BaseVectorStore


class FAISSStore(BaseVectorStore):
    """FAISS class for local similarity search and explicit local persistence."""

    def __init__(self, embeddings: Any) -> None:
        # Only stores the embeddings. The FAISS index itself is created in
        # ``add_documents`` once there is data.
        super().__init__(embeddings)

    def _ensure_store(self, documents: list[Document] | None = None) -> Any:
        """Return the FAISS index, building it from ``documents`` if needed."""
        if self._store is None:
            if not documents:
                # Searching before adding anything: there is no index yet.
                raise RuntimeError("Add documents before searching an empty FAISS index.")
            try:
                from langchain_community.vectorstores import FAISS
            except ImportError as exc:
                raise ImportError("Install FAISS with: pip install langchain-community faiss-cpu") from exc
            # Embed the documents and build a new index from them.
            self._store = FAISS.from_documents(documents, self.embeddings)
        return self._store

    def add_documents(self, documents, *, ids: list[str] | None = None) -> list[str]:
        """Add documents, creating the FAISS index on the first call."""
        docs = list(documents)
        if not docs:
            return []
        if self._store is None:
            # First call: make sure every document has an ID (random UUIDs if
            # none were given) so the method can return them.
            ids = ids or [str(uuid4()) for _ in docs]
            try:
                from langchain_community.vectorstores import FAISS
            except ImportError as exc:
                raise ImportError("Install FAISS with: pip install langchain-community faiss-cpu") from exc
            # Build the index from this first batch.
            self._store = FAISS.from_documents(docs, self.embeddings, ids=ids)
            return ids
        # Later calls: add to the existing index.
        return self._store.add_documents(docs, ids=ids)

    def save(self, directory: str | Path = "./faiss_data") -> None:
        """Write the index to ``directory``.

        This creates two files: ``index.faiss`` (the vectors) and ``index.pkl``
        (the documents' text and metadata, stored with Python pickle).
        """
        self._ensure_store().save_local(str(directory))

    # ``@classmethod`` means it's called on the class, not an instance:
    #     store = FAISSStore.load("./faiss_data", embeddings, allow_dangerous_deserialization=True)
    @classmethod
    def load(cls, directory: str | Path, embeddings: Any, *, allow_dangerous_deserialization: bool = False) -> "FAISSStore":
        """Load a user-created index. FAISS uses pickle for local metadata."""
        # Loading a pickle file can run arbitrary code, so only load folders
        # you created yourself. You have to opt in explicitly.
        if not allow_dangerous_deserialization:
            raise ValueError("FAISS loading uses pickle. Enable it only for an index you trust.")
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError("Install FAISS with: pip install langchain-community faiss-cpu") from exc
        instance = cls(embeddings)  # Create an empty FAISSStore...
        # ...and fill it with the index read from disk. Use the same embedding
        # model that built the index, or the query vectors won't match.
        instance._store = FAISS.load_local(
            str(directory), embeddings, allow_dangerous_deserialization=True
        )
        return instance
