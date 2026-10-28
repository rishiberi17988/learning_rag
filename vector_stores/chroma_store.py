"""Chroma vector store with local persistence.

Chroma is an open-source vector database. Here it runs inside your Python
process and saves everything to a folder (``persist_directory``) as you go.
Creating a ChromaStore with the same folder and collection name later gives
you back the same data.
"""

from pathlib import Path
from typing import Any

from .base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Chroma class backed by a local directory; no hosted account required."""

    def __init__(self, embeddings: Any, *, collection_name: str = "learning_rag", persist_directory: str | Path = "./chroma_data") -> None:
        # ``collection_name`` works like a table name: one folder can hold
        # several separate collections.
        super().__init__(embeddings)
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ImportError("Install Chroma with: pip install langchain-chroma chromadb") from exc
        self.persist_directory = Path(persist_directory)
        # Create the folder (and any parent folders) if it doesn't exist yet.
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        # Opens the collection, creating it if needed. Existing data is
        # loaded, and new data is written to disk automatically.
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

    def save(self) -> None:
        """Chroma writes changes automatically; this method is for API symmetry."""
        return None
