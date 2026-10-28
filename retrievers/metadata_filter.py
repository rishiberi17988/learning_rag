"""Metadata filtering: similarity search limited to documents matching a filter.

The problem it solves
---------------------
Sometimes you know *where* the answer should come from: one file, one
topic, documents newer than some year, and so on. Filtering on metadata
restricts the search to those documents before ranking by similarity.

Example: ``{"topic": "chunking"}`` only searches documents whose metadata
contains ``topic == "chunking"``.

Provider differences (handled for you here)
-------------------------------------------
Each provider expects filters in its own format:
* FAISS and Pinecone accept ``{"key": value, ...}`` (all pairs must match).
* Chroma needs ``{"$and": [{"key": value}, ...]}`` when there is more than
  one condition.
* LangChain's InMemoryVectorStore needs a Python function that receives a
  Document and returns True/False.

``build_metadata_filter_retriever`` takes a simple ``{"key": value}`` dict
and converts it to the right format for the store you pass in. Only exact
matches are supported here; for ranges like "year > 2022", pass a
provider-specific filter yourself or use ``self_query.py``.
"""

from typing import Any

from ._utils import to_langchain_store


def _convert_filter(lc_store: Any, metadata_filter: dict[str, Any]) -> Any:
    """Translate a simple ``{"key": value}`` dict for the given store."""
    store_type = type(lc_store).__name__

    if store_type == "InMemoryVectorStore":
        # InMemory filters are functions: keep a document only if every
        # key/value pair matches its metadata.
        def matches(doc) -> bool:
            return all(doc.metadata.get(key) == value for key, value in metadata_filter.items())

        return matches

    if store_type == "Chroma" and len(metadata_filter) > 1:
        # Chroma accepts one condition directly, but several must be
        # combined explicitly with "$and".
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    # FAISS, Pinecone, and single-condition Chroma use the dict as-is.
    return metadata_filter


def build_metadata_filter_retriever(store: Any, metadata_filter: dict[str, Any], *, k: int = 4) -> Any:
    """Return a similarity retriever that only searches documents matching ``metadata_filter``."""
    lc_store = to_langchain_store(store)
    return lc_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": _convert_filter(lc_store, metadata_filter)},
    )

