"""LangChain in-memory vector store for quick experiments.

The simplest option: vectors are kept in a Python dict in RAM, and search
compares the query against every stored vector (cosine similarity). No extra
packages are needed, but nothing is saved to disk.
"""

from typing import Any

from .base import BaseVectorStore


class InMemoryStore(BaseVectorStore):
    """All indexed data disappears when the Python process exits."""

    def __init__(self, embeddings: Any) -> None:
        # Comes with langchain-core, so nothing extra to install.
        from langchain_core.vectorstores import InMemoryVectorStore

        super().__init__(embeddings)  # Saves ``self.embeddings``.
        # Create the empty store right away. add/search/delete are inherited
        # from BaseVectorStore and forwarded to this object.
        self._store = InMemoryVectorStore(embeddings)
