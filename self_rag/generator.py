"""Candidate generation & self-critique ranking for Self-RAG.

In Self-RAG, for each relevant passage $d$ the generator produces a candidate
answer, and the critic scores that candidate with:
* ``[ISSUP]`` (is the answer grounded in $d$?)
* ``[ISUSE]`` (how useful is the answer overall, 1-5?)

Candidates are then ranked so that **fully supported** answers always beat
partially supported or unsupported ones, with ``[ISUSE]`` breaking ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .critics import SelfRAGCritics, _extract_text

GROUNDED_CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    "Using ONLY facts from the passage, answer the question in 1-2 concise sentences, "
    "then self-critique your answer with the [ISSUP] and [ISUSE] reflection tokens.\n\n"
    "Passage: {passage}\n"
    "Question: {question}\n\n"
    "Reply in this exact 3-line format:\n"
    "ANSWER: <1-2 sentence grounded answer>\n"
    "ISSUP: <FULLY_SUPPORTED | PARTIALLY_SUPPORTED | NO_SUPPORT>\n"
    "ISUSE: <integer 1 to 5>"
)

DIRECT_CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    "Answer the following question concisely in 1-2 sentences, then rate its utility "
    "with the [ISUSE] reflection token (1 to 5).\n\n"
    "Question: {question}\n\n"
    "Reply in this exact 2-line format:\n"
    "ANSWER: <concise answer>\n"
    "ISUSE: <integer 1 to 5>"
)

SUPPORT_WEIGHT = {
    "fully_supported": 2,
    "partially_supported": 1,
    "no_support": 0,
}


@dataclass
class CandidateGeneration:
    """One candidate answer paired with its passage and reflection tokens."""

    doc_id: str
    answer: str
    isrel: str  # "relevant" | "irrelevant"
    issup: str  # "fully_supported" | "partially_supported" | "no_support"
    isuse: int  # 1..5

    @property
    def rank_key(self) -> tuple[int, int]:
        """Sort key: groundedness first, utility score second."""
        return (SUPPORT_WEIGHT.get(self.issup, 0), self.isuse)


class SelfRAGGenerator:
    """Generates candidate answers per passage and critiques each one."""

    def __init__(self, llm: Any, critics: SelfRAGCritics | None = None) -> None:
        self.llm = llm
        self.critics = critics or SelfRAGCritics(llm)

    def generate_direct(self, question: str) -> CandidateGeneration:
        """Generate an answer without retrieval (when ``[Retrieve] == no``)."""
        raw = _extract_text(self.llm.invoke(DIRECT_CRITIQUE_PROMPT.format_messages(question=question)))
        answer, _, isuse = _parse_generation_critique(raw)
        return CandidateGeneration(
            doc_id="no-retrieval",
            answer=answer,
            isrel="n/a",
            issup="n/a",
            isuse=isuse,
        )

    def generate_and_critique(self, question: str, document: Document, *, index: int = 0) -> CandidateGeneration:
        """Draft a candidate answer from ``document`` and score ``[ISSUP]`` + ``[ISUSE]``."""
        doc_id = str(document.metadata.get("id", f"doc-{index}"))
        raw = _extract_text(
            self.llm.invoke(GROUNDED_CRITIQUE_PROMPT.format_messages(passage=document.page_content, question=question))
        )
        answer, issup, isuse = _parse_generation_critique(raw)
        return CandidateGeneration(
            doc_id=doc_id,
            answer=answer,
            isrel="relevant",
            issup=issup,
            isuse=isuse,
        )


def _parse_generation_critique(raw: str) -> tuple[str, str, int]:
    """Extract (answer, issup, isuse) from the structured 3-line LLM output."""
    import re

    answer = raw.strip()
    issup = "fully_supported"
    isuse = 4

    for line in raw.splitlines():
        upper = line.strip().upper()
        if upper.startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
        elif upper.startswith("ISSUP:"):
            if "NO_SUPPORT" in upper or "NOT_SUPPORTED" in upper:
                issup = "no_support"
            elif "PARTIAL" in upper:
                issup = "partially_supported"
            else:
                issup = "fully_supported"
        elif upper.startswith("ISUSE:"):
            match = re.search(r"[1-5]", line)
            if match:
                isuse = int(match.group(0))

    return answer, issup, isuse

