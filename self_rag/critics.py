"""Reflection tokens & critics for Self-RAG (Asai et al., 2023).

What makes Self-RAG different from standard RAG?
------------------------------------------------
Standard RAG *always* retrieves a fixed number of passages and *never* checks
whether the generated answer is actually supported by those passages.

Self-RAG introduces four explicit **reflection steps** (called reflection
tokens in the paper):

1. ``[Retrieve]``  -> Does this question even need external retrieval?
                      (e.g. "What is 2 + 2?" -> ``no``; domain question -> ``yes``)
2. ``[ISREL]``     -> Is a retrieved passage actually **relevant** to the question?
                      (``relevant`` vs ``irrelevant``)
3. ``[ISSUP]``     -> Is a candidate answer **supported** by the passage (no
                      hallucinated claims)?
                      (``fully_supported``, ``partially_supported``, ``no_support``)
4. ``[ISUSE]``     -> How **useful** and complete is the answer for the user?
                      (scored ``1`` to ``5``)
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

RETRIEVE_PROMPT = ChatPromptTemplate.from_template(
    "Decide whether answering the question requires retrieving specialized or "
    "up-to-date documents, or if it is basic general knowledge / simple math.\n\n"
    "Question: {question}\n\n"
    "Reply with a single word: YES (retrieve documents) or NO (answer directly)."
)

ISREL_PROMPT = ChatPromptTemplate.from_template(
    "Evaluate the [ISREL] reflection token: does the passage contain information "
    "relevant to answering the question?\n\n"
    "Question: {question}\n"
    "Passage: {passage}\n\n"
    "Reply with a single word: RELEVANT or IRRELEVANT."
)

ISSUP_PROMPT = ChatPromptTemplate.from_template(
    "Evaluate the [ISSUP] reflection token: are the factual claims in the candidate "
    "answer supported by the evidence passage?\n\n"
    "Evidence passage: {passage}\n"
    "Candidate answer: {answer}\n\n"
    "Reply with a single token: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, or NO_SUPPORT."
)

ISUSE_PROMPT = ChatPromptTemplate.from_template(
    "Evaluate the [ISUSE] reflection token: rate how helpful, clear, and complete "
    "the answer is for the user's question on a scale from 1 (useless) to 5 (excellent).\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Reply with a single integer from 1 to 5."
)


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()


class SelfRAGCritics:
    """Evaluates the four Self-RAG reflection tokens using an LLM."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def should_retrieve(self, question: str) -> bool:
        """``[Retrieve]`` token: return True if external documents are needed."""
        raw = _extract_text(self.llm.invoke(RETRIEVE_PROMPT.format_messages(question=question))).upper()
        return "NO" not in raw

    def check_relevance(self, question: str, document: Document) -> str:
        """``[ISREL]`` token: return 'relevant' or 'irrelevant'."""
        raw = _extract_text(
            self.llm.invoke(ISREL_PROMPT.format_messages(question=question, passage=document.page_content))
        ).upper()
        return "irrelevant" if "IRRELEVANT" in raw else "relevant"

    def check_support(self, document: Document, answer: str) -> str:
        """``[ISSUP]`` token: return 'fully_supported', 'partially_supported', or 'no_support'."""
        raw = _extract_text(
            self.llm.invoke(ISSUP_PROMPT.format_messages(passage=document.page_content, answer=answer))
        ).upper()
        if "NO_SUPPORT" in raw or "NOT_SUPPORTED" in raw:
            return "no_support"
        if "PARTIAL" in raw:
            return "partially_supported"
        return "fully_supported"

    def score_utility(self, question: str, answer: str) -> int:
        """``[ISUSE]`` token: return an integer utility rating from 1 to 5."""
        raw = _extract_text(
            self.llm.invoke(ISUSE_PROMPT.format_messages(question=question, answer=answer))
        )
        match = re.search(r"[1-5]", raw)
        return int(match.group(0)) if match else 3

