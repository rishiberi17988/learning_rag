"""Run Corrective RAG (CRAG) across all three corrective paths.

Usage (from the repository root):

    python -m crag.example
    python crag/example.py

Demonstrates:
1. **CORRECT** path   -> local vector store has the answer; CRAG refines it into
   clean sentence strips and generates the answer.
2. **INCORRECT** path -> local vector store does NOT have the answer (e.g. a
   question about the CRAG paper itself); CRAG discards the irrelevant local
   docs, rewrites the query, and pulls from the fallback web source.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document

from crag import CRAGPipeline
from vector_stores import FAISSStore, build_gemini_embeddings

LOCAL_DOCS = [
    Document(
        page_content=(
            "Chunking splits long documents into smaller passages of 200 to 1000 characters "
            "with overlap so each chunk fits the embedding model. Unrelated note: office "
            "lunch is served at 12:30 PM on Fridays."
        ),
        metadata={"id": "local-chunking"},
    ),
    Document(
        page_content=(
            "Hybrid search combines BM25 keyword search with vector similarity search "
            "using Reciprocal Rank Fusion (RRF)."
        ),
        metadata={"id": "local-hybrid"},
    ),
    Document(
        page_content=(
            "Embeddings convert text into fixed-length numerical vectors so semantically "
            "similar texts sit close together in vector space."
        ),
        metadata={"id": "local-embeddings"},
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
    store.add_documents(LOCAL_DOCS, ids=[d.metadata["id"] for d in LOCAL_DOCS])

    crag = CRAGPipeline.from_store(store, k=2)

    questions = [
        # 1. Answered by local store (note also contains an unrelated lunch sentence
        #    that KnowledgeRefiner strips out).
        "How does chunking split long documents?",
        # 2. Not in local store -> triggers INCORRECT verdict + query rewrite + web fallback.
        "What are the three verdicts used by Corrective RAG (CRAG)?",
    ]

    for question in questions:
        print("\n" + "=" * 72)
        print(f"Question: {question}")
        result = crag.run(question)
        print(f"  Evaluator grades : {result.grades}")
        print(f"  CRAG verdict     : {result.verdict.upper()}")
        if result.rewritten_query:
            print(f"  Rewritten query  : {result.rewritten_query!r} (used for fallback search)")
        print(f"  Refined strips ({len(result.knowledge_strips)}):")
        for strip in result.knowledge_strips:
            print(f"    - [{strip.metadata.get('id')}] {strip.page_content}")
        print(f"  Final Answer     : {result.answer}")


if __name__ == "__main__":
    main()

