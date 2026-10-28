"""End-to-end Self-RAG pipeline (Retrieve on demand -> Generate -> Self-Critique).

How Self-RAG works step by step
-------------------------------
1. **Decide whether to retrieve (``[Retrieve]``):**
   * If ``no`` (e.g. basic arithmetic or general chit-chat), generate the answer
     directly and score ``[ISUSE]``.
   * If ``yes``, retrieve candidate passages from the vector store.
2. **Filter by relevance (``[ISREL]``):**
   * Critique every retrieved passage; drop passages marked ``irrelevant``.
3. **Generate & critique per passage (``[ISSUP]`` + ``[ISUSE]``):**
   * For each relevant passage, generate a candidate answer grounded in that
     passage.
   * Check whether the answer's claims are supported by the passage
     (``fully_supported`` / ``partially_supported`` / ``no_support``) and score
     overall utility (``1..5``).
   * If the best candidate is unsupported or low-utility, optionally rewrite the
     query once and retry.
4. **Select the best candidate** ranked by ``([ISSUP], [ISUSE])``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from retrievers import build_gemini_llm, build_similarity_retriever

from .critics import SelfRAGCritics, _extract_text
from .generator import CandidateGeneration, SelfRAGGenerator

RETRY_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "The initial search for the question below did not find relevant supporting "
    "passages. Rewrite it into a clearer semantic search query.\n"
    "Output ONLY the rewritten query.\n\n"
    "Question: {question}\nRewritten query:"
)


@dataclass
class SelfRAGResponse:
    """Full reflection trace and winning answer from a Self-RAG run."""

    question: str
    retrieve_decision: bool
    isrel_grades: list[tuple[str, str]] = field(default_factory=list)
    candidates: list[CandidateGeneration] = field(default_factory=list)
    retried_with_query: str | None = None
    best_candidate: CandidateGeneration | None = None
    answer: str = ""


class SelfRAGPipeline:
    """Self-Reflective RAG pipeline implementing [Retrieve], [ISREL], [ISSUP], and [ISUSE]."""

    def __init__(self, retriever: Any, *, llm: Any = None, max_retries: int = 1) -> None:
        self.retriever = retriever
        self.llm = llm or build_gemini_llm()
        self.critics = SelfRAGCritics(self.llm)
        self.generator = SelfRAGGenerator(self.llm, self.critics)
        self.max_retries = max_retries

    @classmethod
    def from_store(cls, store: Any, *, k: int = 3, llm: Any = None) -> "SelfRAGPipeline":
        """Convenience constructor wrapping ``store`` in a similarity retriever."""
        return cls(build_similarity_retriever(store, k=k), llm=llm)

    def _retrieve_and_score(
        self,
        question: str,
        search_query: str,
    ) -> tuple[list[tuple[str, str]], list[CandidateGeneration]]:
        docs = self.retriever.invoke(search_query)
        isrel_grades: list[tuple[str, str]] = []
        candidates: list[CandidateGeneration] = []

        for i, doc in enumerate(docs):
            doc_id = str(doc.metadata.get("id", f"doc-{i}"))
            rel = self.critics.check_relevance(question, doc)
            isrel_grades.append((doc_id, rel))
            if rel == "relevant":
                candidates.append(self.generator.generate_and_critique(question, doc, index=i))

        return isrel_grades, candidates

    def run(self, question: str) -> SelfRAGResponse:
        """Run the full Self-RAG loop with reflection tokens."""
        # Step 1: [Retrieve] reflection token — do we even need retrieval?
        need_retrieval = self.critics.should_retrieve(question)
        if not need_retrieval:
            direct = self.generator.generate_direct(question)
            return SelfRAGResponse(
                question=question,
                retrieve_decision=False,
                candidates=[direct],
                best_candidate=direct,
                answer=direct.answer,
            )

        # Step 2 & 3: Retrieve -> [ISREL] -> Generate -> [ISSUP] + [ISUSE]
        isrel_grades, candidates = self._retrieve_and_score(question, question)
        retried_query: str | None = None

        # Adaptive retry if no relevant passage or no supported candidate was found
        has_supported = any(c.issup == "fully_supported" for c in candidates)
        if not has_supported and self.max_retries > 0:
            retried_query = _extract_text(
                self.llm.invoke(RETRY_REWRITE_PROMPT.format_messages(question=question))
            ).splitlines()[0].strip(" \"'")
            more_grades, more_candidates = self._retrieve_and_score(question, retried_query)
            isrel_grades.extend(more_grades)
            candidates.extend(more_candidates)

        if candidates:
            best = max(candidates, key=lambda c: c.rank_key)
            return SelfRAGResponse(
                question=question,
                retrieve_decision=True,
                isrel_grades=isrel_grades,
                candidates=candidates,
                retried_with_query=retried_query,
                best_candidate=best,
                answer=best.answer,
            )

        # Fallback if no passage passed [ISREL]
        fallback = self.generator.generate_direct(question)
        return SelfRAGResponse(
            question=question,
            retrieve_decision=True,
            isrel_grades=isrel_grades,
            candidates=[fallback],
            retried_with_query=retried_query,
            best_candidate=fallback,
            answer=fallback.answer,
        )

