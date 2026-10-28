"""HyDE (Hypothetical Document Embeddings): embed a hypothetical answer.

The problem it solves
---------------------
Questions and documents are written in very different styles:
* A user asks a short question: "How do I split documents?"
* A stored passage is an explanation: "Chunking breaks long texts into
  200-1000 character passages with overlap..."

Because embedding models group texts by how they are written as well as what
they mean, a short question vector may not sit as close to the best passage
vector as another passage would.

How HyDE works
--------------
1. Ask an LLM to write a *hypothetical* passage that answers the question.
   (Even if the LLM gets a specific detail wrong, the vocabulary and style
   will look like a real explanatory passage.)
2. Embed that hypothetical passage instead of the raw question.
3. Search the vector store for real stored documents whose vectors are
   closest to the hypothetical passage's vector.
4. Return the real documents (the hypothetical one is discarded).
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from ._utils import build_gemini_llm, to_langchain_store

HYDE_PROMPT = ChatPromptTemplate.from_template(
    "Write a concise, factual 2-3 sentence technical passage that directly "
    "answers the following question. Write it as documentation, not as a "
    "conversation.\n\nQuestion: {question}\n\nPassage:"
)


class HyDERetriever(BaseRetriever):
    """LangChain retriever that searches using a hypothetical answer passage."""

    vectorstore: Any
    llm: Any
    k: int = 4
    last_hypothetical_document: str = ""

    def _generate_hypothetical_document(self, question: str) -> str:
        """Ask the LLM to draft a short hypothetical passage answering ``question``."""
        messages = HYDE_PROMPT.format_messages(question=question)
        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        passage = str(content).strip()
        self.last_hypothetical_document = passage
        return passage

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        # Step 1: generate a hypothetical passage that looks like a real doc.
        hypothetical_doc = self._generate_hypothetical_document(query)
        # Step 2: search the vector store using that hypothetical passage.
        return self.vectorstore.similarity_search(hypothetical_doc, k=self.k)


def build_hyde_retriever(store: Any, *, llm: Any = None, k: int = 4) -> HyDERetriever:
    """Return a HyDE retriever over ``store``.

    After calling ``retriever.invoke(query)``, inspect
    ``retriever.last_hypothetical_document`` to see what the LLM wrote.
    """
    return HyDERetriever(
        vectorstore=to_langchain_store(store),
        llm=llm or build_gemini_llm(),
        k=k,
    )

