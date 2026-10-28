"""Parent document retrieval: search small chunks, return the bigger parent.

The problem it solves
---------------------
Chunk size is a trade-off:
* Small chunks give precise embeddings (each vector is about one idea), so
  search finds the right spot.
* Large chunks give the LLM enough surrounding context to answer well.

How it works
------------
1. Each document (the *parent*) is split into small *child* chunks.
2. Only the child chunks are embedded and stored in the vector store.
3. The full parents are kept in a separate key-value *docstore*, and every
   child remembers its parent's ID.
4. At search time the best child chunks are found, and their parents are
   returned instead (each parent only once).

Optionally pass ``parent_chunk_size`` to split very long documents into
medium-sized parents first (e.g. pages -> 2000-character parents -> 300-
character children).
"""

from typing import Any, Iterable

from langchain_core.documents import Document

from ._utils import to_langchain_store


def build_parent_document_retriever(
    documents: Iterable[Document],
    embeddings: Any,
    *,
    vectorstore: Any = None,
    child_chunk_size: int = 200,
    parent_chunk_size: int | None = None,
    k: int = 4,
) -> Any:
    """Index ``documents`` as small children and return a retriever that yields parents.

    ``vectorstore`` holds the child chunks. It must support starting empty
    (InMemory, Chroma or Pinecone; not FAISS). By default a new in-memory
    store is created. The parents live in an in-memory docstore, so this
    retriever must be rebuilt each time the program starts.
    """
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_core.stores import InMemoryStore as InMemoryDocStore
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if vectorstore is None:
        vectorstore = InMemoryVectorStore(embeddings)

    retriever = ParentDocumentRetriever(
        vectorstore=to_langchain_store(vectorstore),  # Stores child-chunk vectors.
        docstore=InMemoryDocStore(),  # Stores full parents by ID.
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=20),
        parent_splitter=(
            RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=100)
            if parent_chunk_size
            else None  # None = use each original document as a parent.
        ),
        search_kwargs={"k": k},  # Child chunks searched (fewer parents may come back).
    )
    # Splits, embeds and stores the children; saves the parents.
    retriever.add_documents(list(documents))
    return retriever

