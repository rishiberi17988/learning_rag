"""Corrective RAG (CRAG): evaluate retrieval, refine strips, and fall back to web search."""

from .evaluator import EvaluationResult, RetrievalEvaluator
from .pipeline import CRAGPipeline, CRAGResponse
from .refiner import KnowledgeRefiner
from .web_search import DEFAULT_WEB_CORPUS, WebSearchFallback

__all__ = [
    "CRAGPipeline",
    "CRAGResponse",
    "DEFAULT_WEB_CORPUS",
    "EvaluationResult",
    "KnowledgeRefiner",
    "RetrievalEvaluator",
    "WebSearchFallback",
]

