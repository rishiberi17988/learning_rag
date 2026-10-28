"""BM25: classic keyword search (no embeddings, no vector store).

How it works
------------
BM25 is the ranking formula behind many traditional search engines. It
scores a document higher when:
* it contains the words from your question,
* those words are *rare* across all documents (so "FAISS" counts for more
  than "the"),
* and the document isn't padded with lots of unrelated text.

Strengths: great at exact terms such as names, IDs, error codes and jargon;
fast; free (runs locally, no API calls).
Weakness: no understanding of meaning. "split documents" won't match a
document that only says "chunking". Combine it with vector search in
``hybrid.py`` to get both.

Requires: ``pip install rank_bm25``
"""

from typing import Any, Iterable

from langchain_core.documents import Document


def build_bm25_retriever(documents: Iterable[Document], *, k: int = 4) -> Any:
    """Build a keyword index over ``documents`` and return a BM25 retriever.

    BM25 keeps its own small index in memory, so it needs the documents
    themselves rather than a vector store.
    """
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as exc:
        raise ImportError("Install BM25 support with: pip install langchain-community rank_bm25") from exc

    # ``from_documents`` splits each text into words and builds the index.
    retriever = BM25Retriever.from_documents(list(documents))
    retriever.k = k  # How many documents to return per query.
    return retriever

