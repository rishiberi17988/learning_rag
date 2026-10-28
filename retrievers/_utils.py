"""Helpers shared by the retriever modules.

Two small jobs live here so every retriever file can stay short:

1. ``to_langchain_store`` accepts either one of *our* wrapper classes from
   ``vector_stores`` (FAISSStore, ChromaStore, ...) or a plain LangChain
   vector store, and always returns the plain LangChain object. LangChain's
   retriever classes only understand the plain object.
2. ``build_gemini_llm`` creates the chat model used by the "smart"
   retrievers (multi-query, LLM compression, self-query), which ask an LLM
   for help before or after searching.
"""

from typing import Any

# Fast, low-latency Gemini Flash model ideal for small helper tasks such as
# query rewriting, hypothetical answer generation, and filter extraction.
DEFAULT_GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"


def to_langchain_store(store: Any) -> Any:
    """Return the underlying LangChain vector store.

    Our wrappers keep the real store in ``_store`` and expose it through
    ``_ensure_store()`` (which also raises a clear error when FAISS is still
    empty). Anything else is assumed to be a LangChain store already.
    """
    if hasattr(store, "_ensure_store"):
        return store._ensure_store()
    return store


def build_gemini_llm(
    model: str = DEFAULT_GEMINI_CHAT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 6,
) -> Any:
    """Create a Gemini chat model. Reads ``GOOGLE_API_KEY`` from the environment.

    ``temperature=0`` makes answers as repeatable as possible, which is what
    you want when the LLM is producing search queries or filters.
    ``max_retries=6`` allows automatic exponential-backoff retries when the
    Gemini free-tier 15-RPM rate limit is reached.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError("Install Gemini chat support with: pip install langchain-google-genai") from exc
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=max_retries)

