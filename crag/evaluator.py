"""Step 1 of CRAG: Retrieval Evaluator (grades whether each document is relevant).

In Corrective RAG (CRAG), we never blindly trust retrieved passages. Instead,
an LLM acts as a lightweight **Retrieval Evaluator** that inspects every
``(question, document)`` pair and classifies the overall retrieval quality into
one of three verdicts:

* ``"correct"``: At least one retrieved passage clearly answers the question.
* ``"incorrect"``: None of the retrieved passages are relevant -> discard them
  and fall back to an external search.
* ``"ambiguous"``: The passages are only partially relevant -> refine what we
  have *and* supplement it with an external search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

GRADE_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict retrieval evaluator for a RAG system.\n"
    "Judge whether the retrieved document contains information that helps answer the question.\n\n"
    "Question: {question}\n"
    "Document: {document}\n\n"
    "Reply with a single word: RELEVANT, PARTIAL, or IRRELEVANT."
)


@dataclass
class EvaluationResult:
    """Output of the CRAG Retrieval Evaluator."""

    verdict: str  # "correct", "ambiguous", or "incorrect"
    relevant_docs: list[Document]
    grades: list[tuple[str, str]]  # [(doc_id, "RELEVANT"|"PARTIAL"|"IRRELEVANT"), ...]


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()


class RetrievalEvaluator:
    """Grades retrieved documents and decides the CRAG action ('correct', 'ambiguous', 'incorrect')."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def grade_document(self, question: str, document: Document) -> str:
        """Return 'RELEVANT', 'PARTIAL', or 'IRRELEVANT' for a single document."""
        raw = _extract_text(
            self.llm.invoke(GRADE_PROMPT.format_messages(question=question, document=document.page_content))
        ).upper()
        if "IRRELEVANT" in raw:
            return "IRRELEVANT"
        if "PARTIAL" in raw:
            return "PARTIAL"
        if "RELEVANT" in raw:
            return "RELEVANT"
        return "IRRELEVANT"

    def evaluate(self, question: str, documents: Iterable[Document]) -> EvaluationResult:
        """Grade every retrieved document and compute the overall CRAG verdict."""
        docs = list(documents)
        grades: list[tuple[str, str]] = []
        relevant: list[Document] = []
        has_strong = False
        has_partial = False

        for i, doc in enumerate(docs):
            doc_id = str(doc.metadata.get("id", f"doc-{i}"))
            label = self.grade_document(question, doc)
            grades.append((doc_id, label))
            if label == "RELEVANT":
                has_strong = True
                relevant.append(doc)
            elif label == "PARTIAL":
                has_partial = True
                relevant.append(doc)

        if has_strong:
            verdict = "correct"
        elif has_partial:
            verdict = "ambiguous"
        else:
            verdict = "incorrect"

        return EvaluationResult(verdict=verdict, relevant_docs=relevant, grades=grades)

