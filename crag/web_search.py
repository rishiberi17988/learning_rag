"""Step 3 of CRAG: Query Rewriter + Fallback External Search.

When the Retrieval Evaluator grades local retrieval as ``"incorrect"`` or
``"ambiguous"``, CRAG:
1. **Rewrites** the user's question into a concise keyword search query.
2. **Searches an external knowledge source** (in production, a web search API;
   by default we also include a built-in fallback knowledge base so the demo
   works offline without paid search API keys, plus optional live Wikipedia /
   DuckDuckGo hooks).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .evaluator import _extract_text

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the following question into a concise 3-6 word web search query "
    "that captures the core technical concepts.\n"
    "Output ONLY the rewritten search query.\n\n"
    "Question: {question}\nSearch query:"
)

# Built-in fallback knowledge articles representing broader "web" facts outside
# our local vector store, so CRAG's corrective path works deterministically.
DEFAULT_WEB_CORPUS: list[Document] = [
    Document(
        page_content=(
            "Corrective RAG (CRAG, Yan et al. 2024) uses a lightweight retrieval evaluator "
            "to grade retrieved documents as Correct, Incorrect, or Ambiguous, applies "
            "decompose-then-recompose knowledge refinement, and falls back to web search "
            "when local retrieval fails."
        ),
        metadata={"id": "web-crag-paper", "source": "web_fallback"},
    ),
    Document(
        page_content=(
            "Self-RAG (Asai et al. 2023) trains a model to emit reflection tokens — "
            "Retrieve, ISREL (relevance), ISSUP (groundedness/support), and ISUSE (utility) — "
            "so it can decide when to retrieve and critique its own generated answers."
        ),
        metadata={"id": "web-self-rag-paper", "source": "web_fallback"},
    ),
    Document(
        page_content=(
            "GraphRAG builds a knowledge graph of entities and relationships from documents "
            "and summarizes communities so global, multi-hop questions can be answered."
        ),
        metadata={"id": "web-graphrag", "source": "web_fallback"},
    ),
]


class WebSearchFallback:
    """Rewrites the query and searches an external fallback source."""

    def __init__(
        self,
        llm: Any,
        *,
        search_fn: Callable[[str], list[Document]] | None = None,
        web_corpus: list[Document] | None = None,
    ) -> None:
        self.llm = llm
        self.search_fn = search_fn
        self.web_corpus = web_corpus if web_corpus is not None else list(DEFAULT_WEB_CORPUS)

    def rewrite_query(self, question: str) -> str:
        """Ask the LLM to turn ``question`` into a short web-search query."""
        rewritten = _extract_text(self.llm.invoke(REWRITE_PROMPT.format_messages(question=question)))
        return rewritten.splitlines()[0].strip(" \"'") if rewritten else question

    def search(self, query: str, *, k: int = 2) -> list[Document]:
        """Run ``search_fn(query)`` if provided, otherwise keyword-match ``web_corpus``."""
        if self.search_fn is not None:
            return self.search_fn(query)[:k]

        words = {w.lower() for w in query.replace("-", " ").split() if len(w) > 2}
        scored: list[tuple[int, Document]] = []
        for doc in self.web_corpus:
            text_lower = doc.page_content.lower()
            overlap = sum(1 for w in words if w in text_lower)
            scored.append((overlap, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for score, doc in scored[:k] if score > 0] or self.web_corpus[:1]

