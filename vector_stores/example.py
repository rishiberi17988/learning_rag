"""Run a small add-and-search demo with any (or every) vector store.

Usage (from the repository root, with your environment activated):

    python -m vector_stores.example              # FAISS (the default)
    python -m vector_stores.example memory       # InMemoryStore
    python -m vector_stores.example chroma       # ChromaStore
    python -m vector_stores.example pinecone     # PineconeStore (needs PINECONE_API_KEY)
    python -m vector_stores.example all          # every store, one after another

``python vector_stores/example.py chroma`` works too. If you use your IDE's
"Run File" button (no arguments), change ``DEFAULT_STORE`` below instead.

Embeddings come from Google Gemini, so ``GOOGLE_API_KEY`` must be set in
``.env`` (or in your shell).

Running it many times is safe:
* Documents use fixed IDs, so adding them again overwrites them instead of
  creating duplicates in Chroma or Pinecone, which keep their data.
* If a store fails (e.g. a package or API key is missing), the error is
  printed and the remaining stores still run.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# When run as ``python vector_stores/example.py``, Python starts with this
# folder on sys.path rather than the repository root. Add the package parent so
# the same example also works from an IDE's Run File action.
REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document

from vector_stores import ChromaStore, FAISSStore, InMemoryStore, PineconeStore, build_gemini_embeddings

# Store used when no command-line argument is given (e.g. IDE "Run File").
# One of: "memory", "faiss", "chroma", "pinecone", "all".
DEFAULT_STORE = "faiss"

# Folders for the local stores. They're anchored to the repository root so
# they end up in the same place whatever directory you run from.
FAISS_DIR = REPO_ROOT / "faiss_data"
CHROMA_DIR = REPO_ROOT / "chroma_data"
PINECONE_INDEX = "learning-rag-demo-gemini"

# Sample data. Fixed IDs make reruns overwrite instead of duplicate.
DOCUMENTS = [
    Document(page_content="RAG retrieves relevant passages before generating an answer.", metadata={"topic": "RAG"}),
    Document(page_content="Chunking splits long documents into smaller passages.", metadata={"topic": "chunking"}),
    Document(page_content="Embeddings represent text as vectors for similarity search.", metadata={"topic": "embeddings"}),
]
IDS = ["demo-rag", "demo-chunking", "demo-embeddings"]
QUERY = "How do I split documents?"


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ (no extra package needed).

    Variables already set in your shell take priority over the file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue  # Skip blank lines, comments and malformed lines.
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# One small function per store. Each returns a ready-to-use store object.
# ---------------------------------------------------------------------------

def make_memory(embeddings):
    return InMemoryStore(embeddings)


def make_faiss(embeddings):
    return FAISSStore(embeddings)  # The index is built on the first add_documents().


def make_chroma(embeddings):
    # A collection only holds vectors from one model. The "_gemini" suffix
    # keeps it apart from collections built with other embedding models.
    return ChromaStore(embeddings, collection_name="learning_rag_demo_gemini", persist_directory=CHROMA_DIR)


def make_pinecone(embeddings):
    if not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY is not set. Add it to .env or export it in your shell.")
    # Same idea: a separate index for the Gemini (768-dimension) vectors.
    return PineconeStore(embeddings, index_name=PINECONE_INDEX)


STORES = {
    "memory": make_memory,
    "faiss": make_faiss,
    "chroma": make_chroma,
    "pinecone": make_pinecone,
}


def wait_until_searchable(store, expected: int, timeout: float = 30.0) -> None:
    """Pinecone takes a few seconds to make new vectors searchable.

    Search repeatedly until our demo documents show up (or time runs out).
    For the local stores the first check passes immediately.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = store.similarity_search(QUERY, k=expected)
        if len(found) >= expected:
            return
        time.sleep(1)
    print(f"  (warning: results still incomplete after {timeout:.0f}s)")


def run_demo(name: str, embeddings) -> None:
    """Create one store, add the sample documents and run a search."""
    print(f"\n=== {name} ===")
    store = STORES[name](embeddings)

    # 1. Add (or overwrite) the documents. Each one is embedded and stored.
    ids = store.add_documents(DOCUMENTS, ids=IDS)
    print(f"Added {len(ids)} documents: {ids}")
    wait_until_searchable(store, expected=len(DOCUMENTS))

    # 2. Plain search: the documents closest in meaning to the question.
    print(f"Query: {QUERY!r}")
    for doc in store.similarity_search(QUERY, k=2):
        print(f"  -> [{doc.metadata['topic']}] {doc.page_content}")

    # 3. Search with scores. FAISS/Chroma: distance (lower = closer).
    #    InMemory/Pinecone: cosine similarity (higher = closer).
    for doc, score in store.similarity_search_with_score(QUERY, k=2):
        print(f"  score={score:.3f} [{doc.metadata['topic']}]")

    # 4. Retriever: the object a RAG chain uses to fetch context.
    retriever = store.as_retriever(search_kwargs={"k": 1})
    print(f"Retriever top hit: {retriever.invoke(QUERY)[0].metadata['topic']}")

    # 5. Persistence, which differs per store.
    if isinstance(store, FAISSStore):
        store.save(FAISS_DIR)  # FAISS: you save and load it yourself.
        reloaded = FAISSStore.load(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        print(f"Saved to {FAISS_DIR} and reloaded ({len(reloaded.similarity_search(QUERY, k=3))} docs found)")
    elif isinstance(store, ChromaStore):
        print(f"Chroma saved automatically to {CHROMA_DIR}")
    elif isinstance(store, PineconeStore):
        print(f"Pinecone stores data in the cloud (index '{PINECONE_INDEX}')")
    else:
        print("InMemoryStore: data is lost when the program exits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector store demo")
    parser.add_argument(
        "store",
        nargs="?",  # The argument is optional.
        default=DEFAULT_STORE,
        choices=[*STORES, "all"],
        help="which store to run (default: %(default)s)",
    )
    args = parser.parse_args()

    load_dotenv()  # Makes GOOGLE_API_KEY and PINECONE_API_KEY from .env available.

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")
    print("Using Gemini embeddings (gemini-embedding-001, 768 dimensions)...")
    embeddings = build_gemini_embeddings()

    names = list(STORES) if args.store == "all" else [args.store]
    failures = []
    for name in names:
        try:
            run_demo(name, embeddings)
        except Exception as exc:  # noqa: BLE001 - report the error and keep going.
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append(name)

    print("\nDone." if not failures else f"\nDone, but these failed: {', '.join(failures)}")
    sys.exit(1 if failures else 0)  # A non-zero exit code tells scripts/CI it failed.


if __name__ == "__main__":
    main()
