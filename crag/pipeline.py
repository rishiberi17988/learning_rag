"""End-to-end Corrective RAG (CRAG) pipeline.

How CRAG works step by step
---------------------------
1. **Retrieve** candidate documents from the local vector store (or retriever).
2. **Evaluate** each document with ``RetrievalEvaluator`` -> verdict is one of:
   * ``"correct"``: refine the relevant local documents into clean strips.
   * ``"incorrect"``: discard local documents, rewrite the query, and search
     the fallback web source.
   * ``"ambiguous"``: combine refined local strips *and* fallback web results.
3. **Generate** a grounded answer using only the verified/refined knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from retrievers import build_gemini_llm, build_similarity_retriever

from .evaluator import RetrievalEvaluator, _extract_text
from .refiner import KnowledgeRefiner
from .web_search import WebSearchFallback

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question concisely using ONLY the verified knowledge below.\n"
    "If the knowledge is insufficient, say so clearly.\n\n"
    "Verified knowledge:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)


@dataclass
class CRAGResponse:
    """Full trace and final answer from a CRAG run."""

    question: str
    verdict: str  # "correct", "ambiguous", or "incorrect"
    grades: list[tuple[str, str]]
    rewritten_query: str | None
    knowledge_strips: list[Document] = field(default_factory=list)
    answer: str = ""


class CRAGPipeline:
    """Corrective RAG pipeline combining evaluation, refinement, and fallback search."""

    def __init__(
        self,
        retriever: Any,
        *,
        llm: Any = None,
        fallback: WebSearchFallback | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or build_gemini_llm()
        self.evaluator = RetrievalEvaluator(self.llm)
        self.refiner = KnowledgeRefiner(self.llm)
        self.fallback = fallback or WebSearchFallback(self.llm)

    @classmethod
    def from_store(cls, store: Any, *, k: int = 3, llm: Any = None) -> "CRAGPipeline":
        """Convenience constructor that wraps ``store`` in a similarity retriever."""
        return cls(build_similarity_retriever(store, k=k), llm=llm)

    def run(self, question: str) -> CRAGResponse:
        """Execute Retrieve -> Evaluate -> Correct/Refine -> Generate."""
        # 1. Retrieve from local store
        retrieved_docs = self.retriever.invoke(question)

        # 2. Grade retrieved documents
        eval_result = self.evaluator.evaluate(question, retrieved_docs)
        verdict = eval_result.verdict

        rewritten_query: str | None = None
        final_strips: list[Document] = []

        # 3. Take corrective action based on the verdict
        if verdict == "correct":
            final_strips = self.refiner.refine(question, eval_result.relevant_docs)
        elif verdict == "incorrect":
            rewritten_query = self.fallback.rewrite_query(question)
            web_docs = self.fallback.search(rewritten_query)
            final_strips = self.refiner.refine(question, web_docs) or web_docs
        else:  # "ambiguous"
            local_strips = self.refiner.refine(question, eval_result.relevant_docs)
            rewritten_query = self.fallback.rewrite_query(question)
            web_docs = self.fallback.search(rewritten_query)
            web_strips = self.refiner.refine(question, web_docs) or web_docs
            final_strips = local_strips + web_strips

        # 4. Generate the final answer from the cleaned knowledge strips
        context = "\n".join(f"- {doc.page_content}" for doc in final_strips) or "(no verified context)"
        answer = _extract_text(
            self.llm.invoke(ANSWER_PROMPT.format_messages(question=question, context=context))
        )

        return CRAGResponse(
            question=question,
            verdict=verdict,
            grades=eval_result.grades,
            rewritten_query=rewritten_query,
            knowledge_strips=final_strips,
            answer=answer,
        )

