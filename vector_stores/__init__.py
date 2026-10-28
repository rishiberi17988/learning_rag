"""Reusable classes for common vector store providers.

Importing from this package lets you write::

    from vector_stores import FAISSStore, build_gemini_embeddings

instead of importing from each module. Provider libraries (faiss, chromadb,
pinecone) are only imported when you *create* a store, so importing this
package works even if a provider isn't installed.
"""

from .chroma_store import ChromaStore  # Local, saves to disk automatically.
from .embeddings import build_gemini_embeddings  # Google Gemini embeddings (default).
from .embeddings import build_local_embeddings  # Free local Hugging Face model (offline option).
from .faiss_store import FAISSStore  # Local and fast; you save/load it yourself.
from .in_memory_store import InMemoryStore  # RAM only, for quick experiments.
from .pinecone_store import PineconeStore  # Hosted in the cloud; needs an API key.

# The names that ``from vector_stores import *`` exposes.
__all__ = [
    "ChromaStore",
    "FAISSStore",
    "InMemoryStore",
    "PineconeStore",
    "build_gemini_embeddings",
    "build_local_embeddings",
]
