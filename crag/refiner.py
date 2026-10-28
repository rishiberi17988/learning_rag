"""Step 2 of CRAG: Decompose-then-Recompose Knowledge Refinement.

Even when a retrieved document is relevant, it often contains unrelated
sentences that can distract the generator. The CRAG paper introduces a
**decompose-then-recompose** algorithm:

1. **Decompose:** Split each retrieved document into small knowledge strips
   (individual sentences or short clauses).
2. **Filter:** Score each strip against the question and drop irrelevant ones.
3. **Recompose:** Concatenate the surviving strips back together in order to
   form clean, high-signal context for the LLM.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .evaluator import _extract_text

STRIP_FILTER_PROMPT = ChatPromptTemplate.from_template(
    "You are filtering sentences from retrieved documents so only facts relevant "
    "to the question remain.\n\n"
    "Question: {question}\n\n"
    "Numbered sentences:\n{strips}\n\n"
    "Reply with a comma-separated list of the sentence numbers (e.g. 0, 2, 3) that "
    "directly help answer the question, or NONE if none apply."
)


class KnowledgeRefiner:
    """Splits documents into sentence strips, filters out noise, and recomposes them."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @staticmethod
    def decompose(documents: Iterable[Document]) -> list[tuple[str, dict[str, Any]]]:
        """Break documents into individual sentence strips while keeping metadata."""
        strips: list[tuple[str, dict[str, Any]]] = []
        for doc in documents:
            for sentence in re.split(r"(?<=[.!?])\s+", doc.page_content.strip()):
                cleaned = sentence.strip()
                if cleaned:
                    strips.append((cleaned, dict(doc.metadata)))
        return strips

    def refine(self, question: str, documents: Iterable[Document]) -> list[Document]:
        """Return refined ``Document`` strips containing only sentences relevant to ``question``."""
        strips = self.decompose(documents)
        if not strips:
            return []

        numbered = "\n".join(f"[{i}] {text}" for i, (text, _) in enumerate(strips))
        raw = _extract_text(
            self.llm.invoke(STRIP_FILTER_PROMPT.format_messages(question=question, strips=numbered))
        )
        if "NONE" in raw.upper():
            return []

        kept_indices = [
            int(match) for match in re.findall(r"\d+", raw) if 0 <= int(match) < len(strips)
        ]
        if not kept_indices:
            # Fallback: keep all strips if the LLM output wasn't parseable.
            kept_indices = list(range(len(strips)))

        return [
            Document(page_content=strips[i][0], metadata={**strips[i][1], "strip_index": i})
            for i in dict.fromkeys(kept_indices)
        ]

