"""Embedding helpers for the vector store examples.

An *embedding model* converts text into a fixed-length list of numbers (a
vector). Texts with similar meaning end up with similar vectors, which is what
makes similarity search work.

Two options are provided:

* ``build_gemini_embeddings()`` (default): Google's hosted
  ``gemini-embedding-001`` model. Needs ``GOOGLE_API_KEY`` and an internet
  connection.
* ``build_local_embeddings()``: a free Hugging Face model that runs on your
  CPU. Use it offline or to avoid API usage.

Important: vectors from different models (or different dimensions) can't be
mixed. A Chroma collection or Pinecone index built with one model must be
searched with that same model. Use a new collection/index when you switch.
"""

# Gemini's embedding model. It returns 3072 numbers by default, but it can
# shorten its output. 768 keeps most of the quality and uses 4x less storage.
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_EMBEDDING_DIMENSION = 768


def build_gemini_embeddings(
    model_name: str = GEMINI_EMBEDDING_MODEL,
    dimension: int | None = GEMINI_EMBEDDING_DIMENSION,
    api_key: str | None = None,
):
    """Create a Google Gemini embedding model.

    The API key comes from ``api_key`` if given, otherwise from the
    ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) environment variable.
    Pass ``dimension=None`` to use the model's full 3072 dimensions.
    """
    # Import inside the function so the package can be imported even when
    # this optional dependency isn't installed.
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise ImportError("Install Gemini embeddings with: pip install langchain-google-genai") from exc

    kwargs = {"model": model_name, "output_dimensionality": dimension}
    if api_key:
        kwargs["google_api_key"] = api_key
    # The returned object has ``embed_documents(texts)`` and
    # ``embed_query(text)``. Documents are embedded as "RETRIEVAL_DOCUMENT"
    # and queries as "RETRIEVAL_QUERY" (task types Gemini uses to improve
    # search quality). The vector stores call these methods for you.
    return GoogleGenerativeAIEmbeddings(**kwargs)


def build_local_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create a free local Hugging Face embedding model (384 dimensions).

    The model is downloaded on first use. Install ``langchain-huggingface`` and
    ``sentence-transformers`` before calling this function.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        # Replace the generic import error with one that says what to install.
        raise ImportError(
            "Install local embedding dependencies with: "
            "pip install langchain-huggingface sentence-transformers"
        ) from exc

    return HuggingFaceEmbeddings(model_name=model_name)
