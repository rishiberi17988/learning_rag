"""Run Self-RAG and inspect all four reflection tokens ([Retrieve], [ISREL], [ISSUP], [ISUSE]).

Usage (from the repository root):

    python -m self_rag.example
    python self_rag/example.py

Demonstrates two cases:
1. A domain question where ``[Retrieve] = YES`` -> retrieves passages, filters
   with ``[ISREL]``, generates candidate answers per relevant passage, critiques
   each with ``[ISSUP]`` (groundedness) and ``[ISUSE]`` (1-5 utility), and picks
   the best-supported candidate.
2. A simple arithmetic / general question where ``[Retrieve] = NO`` -> skips
   vector search altogether and answers directly.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document

from self_rag import SelfRAGPipeline
from vector_stores import FAISSStore, build_gemini_embeddings

DOCS = [
    Document(
        page_content=(
            "Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrievers "
            "by scoring each document as sum(1 / (rank + c)), where c is typically 60."
        ),
        metadata={"id": "doc-rrf"},
    ),
    Document(
        page_content=(
            "Hybrid retrieval combines BM25 keyword search (which excels at exact terms "
            "like error codes) with dense vector search (which captures semantic meaning)."
        ),
        metadata={"id": "doc-hybrid"},
    ),
    Document(
        page_content=(
            "Chunking splits long documents into 200 to 1000 character passages with overlap "
            "so each chunk fits within the embedding model's context window."
        ),
        metadata={"id": "doc-chunking"},
    ),
]


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    load_dotenv()
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")

    embeddings = build_gemini_embeddings()
    store = FAISSStore(embeddings)
    store.add_documents(DOCS, ids=[d.metadata["id"] for d in DOCS])

    pipeline = SelfRAGPipeline.from_store(store, k=3)

    questions = [
        # 1. Domain question -> [Retrieve]=YES, evaluates [ISREL], [ISSUP], [ISUSE]
        "How does Reciprocal Rank Fusion combine results in hybrid search?",
        # 2. Trivial math question -> [Retrieve]=NO, answers without touching the vector store
        "What is 15 multiplied by 4?",
    ]

    for question in questions:
        print("\n" + "=" * 72)
        print(f"Question: {question}")
        res = pipeline.run(question)
        print(f"  [Retrieve] needed? : {'YES' if res.retrieve_decision else 'NO (answered directly)'}")
        if res.isrel_grades:
            print(f"  [ISREL] grades     : {res.isrel_grades}")
        if res.retried_with_query:
            print(f"  Retried query      : {res.retried_with_query!r}")
        print(f"  Candidate critiques ({len(res.candidates)}):")
        for c in res.candidates:
            print(f"    * [{c.doc_id}] ISSUP={c.issup} | ISUSE={c.isuse}/5 -> {c.answer}")
        print(f"  Selected Answer    : {res.answer}")


if __name__ == "__main__":
    main()

