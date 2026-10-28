"""Self-query retrieval: the LLM turns a question into a search + metadata filter.

The problem it solves
---------------------
Questions often mix *meaning* with *conditions*, for example
"advanced articles about retrieval published after 2023". Similarity search
can't apply "advanced" or "after 2023" reliably, but a metadata filter can.

How it works
------------
1. You describe the documents and their metadata fields (name, type,
   meaning) using ``AttributeInfo``.
2. The LLM reads the question and produces a structured query:
       search text: "retrieval"
       filter:      level == "advanced" AND year > 2023
3. The filter is translated into the vector store's own filter format and
   the search runs.

Supported stores: ChromaStore and PineconeStore (LangChain has no filter
translator for FAISS or the in-memory store).

Requires: ``pip install lark``
"""

from typing import Any

from ._utils import build_gemini_llm, to_langchain_store


def build_self_query_retriever(
    store: Any,
    *,
    document_contents: str,
    metadata_fields: list[tuple[str, str, str]],
    llm: Any = None,
    k: int = 4,
) -> Any:
    """Return a retriever that lets the LLM write metadata filters from the question.

    ``document_contents``: a short description of what the documents are.
    ``metadata_fields``: (name, type, description) for each filterable field,
    e.g. ``("year", "integer", "Year the note was written")``.
    """
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

    # Tell the LLM which fields exist and what they mean.
    attribute_info = [
        AttributeInfo(name=name, type=type_, description=description)
        for name, type_, description in metadata_fields
    ]
    return SelfQueryRetriever.from_llm(
        llm=llm or build_gemini_llm(),  # Writes the structured query.
        vectorstore=to_langchain_store(store),  # Must be Chroma or Pinecone.
        document_contents=document_contents,
        metadata_field_info=attribute_info,
        search_kwargs={"k": k},
    )

