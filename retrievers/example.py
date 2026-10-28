"""Try every retrieval strategy on the same small set of notes.

Usage (from the repository root, with your environment activated):

    python -m retrievers.example                      # all retrievers, FAISS store
    python -m retrievers.example fusion hyde rerank   # only selected retrievers
    python -m retrievers.example all --store chroma   # pick the vector store
    python -m retrievers.example --list               # show retriever names

``python retrievers/example.py ...`` works too. For your IDE's "Run File"
button (no arguments), change ``DEFAULT_RETRIEVERS`` / ``DEFAULT_STORE``.

Needs ``GOOGLE_API_KEY`` in ``.env`` (Gemini embeddings and LLM), plus
``PINECONE_API_KEY`` if you choose ``--store pinecone``. Each demo runs on
its own: if one fails, the error is printed and the rest still run.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Make ``import vector_stores`` / ``import retrievers`` work when this file is
# run directly (``python retrievers/example.py``) or from an IDE.
REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document

from retrievers import (
    build_bm25_retriever,
    build_compression_retriever,
    build_ensemble_retriever,
    build_fusion_retriever,
    build_hybrid_retriever,
    build_hyde_retriever,
    build_metadata_filter_retriever,
    build_mmr_retriever,
    build_multi_query_retriever,
    build_multi_strategy_ensemble_retriever,
    build_parent_document_retriever,
    build_reranking_retriever,
    build_score_threshold_retriever,
    build_self_query_retriever,
    build_similarity_retriever,
)
from retrievers._utils import to_langchain_store
from vector_stores import ChromaStore, FAISSStore, InMemoryStore, PineconeStore, build_gemini_embeddings

# Used when no command-line arguments are given (e.g. IDE "Run File").
DEFAULT_RETRIEVERS = ["all"]
DEFAULT_STORE = "faiss"  # One of: memory, faiss, chroma, pinecone.

CHROMA_DIR = REPO_ROOT / "chroma_data"
CHROMA_COLLECTION = "learning_rag_retrievers"
PINECONE_INDEX = "learning-rag-retrievers"

# ---------------------------------------------------------------------------
# Sample notes. They're chosen so each strategy has something to show:
#   * two near-duplicate chunking notes   -> MMR picks only one of them
#   * jargon such as "HNSW" and "E1042"   -> BM25 / hybrid / ensemble shine
#   * level / year metadata               -> metadata filter, self-query
#   * one long note                       -> parent document retrieval
# ---------------------------------------------------------------------------
NOTES = [
    ("chunking-basics", "chunking", "beginner", 2022,
     "Chunking splits long documents into smaller passages so each piece fits the embedding model and the prompt."),
    ("chunking-sizes", "chunking", "beginner", 2023,
     "Splitting long documents into smaller chunks of 200 to 1000 characters, with some overlap, is a common starting point."),
    ("chunking-semantic", "chunking", "advanced", 2024,
     "Semantic chunking breaks text where the topic changes, using embeddings to detect boundaries between passages."),
    ("embeddings", "embeddings", "beginner", 2022,
     "Embeddings represent text as vectors so that texts with similar meaning end up close together."),
    ("hnsw-index", "vector_db", "advanced", 2023,
     "HNSW is a graph-based index used by FAISS, Chroma and Pinecone for fast approximate nearest neighbour search."),
    ("bm25", "retrieval", "beginner", 2021,
     "BM25 ranks documents by keyword overlap and gives rare terms, like error code E1042, extra weight."),
    ("hybrid-rerank", "retrieval", "advanced", 2024,
     "Hybrid retrieval merges keyword and vector results with reciprocal rank fusion, then a reranker reorders the top hits."),
    ("rag-pipeline", "rag", "beginner", 2023,
     "A RAG pipeline has two stages. First, the retriever finds passages related to the question in a vector store. "
     "Second, those passages are inserted into the prompt and the LLM writes an answer grounded in them. "
     "After retrieval, many systems also rerank or compress the passages so only the most useful text reaches the LLM. "
     "Citing the source passages lets users check the answer."),
    ("rag-evaluation", "rag", "advanced", 2025,
     "RAG evaluation measures context precision, context recall and answer faithfulness, for example with RAGAS or DeepEval."),
]
DOCUMENTS = [
    Document(page_content=text, metadata={"id": doc_id, "topic": topic, "level": level, "year": year})
    for doc_id, topic, level, year, text in NOTES
]
IDS = [doc.metadata["id"] for doc in DOCUMENTS]

# Description used by the self-query retriever.
DOCUMENT_CONTENTS = "Short study notes about retrieval-augmented generation (RAG)"
METADATA_FIELDS = [
    ("topic", "string", "Subject of the note: chunking, embeddings, vector_db, retrieval or rag"),
    ("level", "string", "Difficulty: 'beginner' or 'advanced'"),
    ("year", "integer", "Year the note was written"),
]


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ (shell variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def show(docs, *, preview: int = 90) -> None:
    """Print one line per retrieved document."""
    if not docs:
        print("    (no documents returned)")
    for rank, doc in enumerate(docs, 1):
        meta = doc.metadata
        score_str = f" | score={meta['rerank_score']:.1f}" if "rerank_score" in meta else ""
        text = doc.page_content.replace("\n", " ")
        text = text if len(text) <= preview else text[: preview - 3] + "..."
        print(f"    {rank}. [{meta.get('id')} | {meta.get('level')} | {meta.get('year')}{score_str}] {text}")


# ---------------------------------------------------------------------------
# Vector store setup
# ---------------------------------------------------------------------------

def build_store(name: str, embeddings):
    """Create the chosen vector store and load the sample notes into it."""
    if name == "memory":
        store = InMemoryStore(embeddings)
    elif name == "faiss":
        store = FAISSStore(embeddings)
    elif name == "chroma":
        store = ChromaStore(embeddings, collection_name=CHROMA_COLLECTION, persist_directory=CHROMA_DIR)
    elif name == "pinecone":
        if not os.environ.get("PINECONE_API_KEY"):
            raise RuntimeError("PINECONE_API_KEY is not set. Add it to .env.")
        store = PineconeStore(embeddings, index_name=PINECONE_INDEX)
    else:
        raise ValueError(f"Unknown store {name!r}")

    # Fixed IDs: rerunning overwrites the notes instead of duplicating them.
    store.add_documents(DOCUMENTS, ids=IDS)
    if name == "pinecone":
        # Pinecone needs a moment before new vectors are searchable.
        deadline = time.time() + 30
        while time.time() < deadline and len(store.similarity_search("RAG", k=len(IDS))) < len(IDS):
            time.sleep(1)
    return store


# ---------------------------------------------------------------------------
# One demo per retriever. Each receives a context dict (store, embeddings,
# store name) and prints what it retrieves.
# ---------------------------------------------------------------------------

def demo_similarity(ctx):
    query = "How do I split documents?"
    print(f"  Query: {query!r}  (top 3 by meaning)")
    show(build_similarity_retriever(ctx["store"], k=3).invoke(query))


def demo_mmr(ctx):
    query = "How do I split documents?"
    print(f"  Query: {query!r}  (top 3, relevant but varied)")
    print("  Compare with 'similarity': the near-duplicate chunking note is swapped for something different.")
    show(build_mmr_retriever(ctx["store"], k=3, fetch_k=9, lambda_mult=0.5).invoke(query))


def demo_score_threshold(ctx):
    threshold = 0.6
    lc_store = to_langchain_store(ctx["store"])
    retriever = build_score_threshold_retriever(ctx["store"], score_threshold=threshold, k=4)
    for query in ["How do I split documents?", "What is the capital of France?"]:
        scored = lc_store.similarity_search_with_relevance_scores(query, k=4)
        scores = ", ".join(f"{d.metadata['id']}={s:.2f}" for d, s in scored)
        print(f"  Query: {query!r}  (threshold {threshold})")
        print(f"    relevance scores: {scores}")
        show(retriever.invoke(query))


def demo_metadata_filter(ctx):
    query = "How do I split documents?"
    metadata_filter = {"level": "advanced"}
    print(f"  Query: {query!r}  filter={metadata_filter}")
    show(build_metadata_filter_retriever(ctx["store"], metadata_filter, k=3).invoke(query))


def demo_bm25(ctx):
    for query in ["What does error E1042 mean?", "How do I split documents?"]:
        print(f"  Query: {query!r}  (keyword match only)")
        show(build_bm25_retriever(DOCUMENTS, k=2).invoke(query))
    print("  Note: the exact code 'E1042' is found instantly; wording matters for the second query.")


def demo_hybrid(ctx):
    query = "fast HNSW nearest neighbour search"
    print(f"  Query: {query!r}  (BM25 + vector similarity merged with RRF)")
    show(build_hybrid_retriever(ctx["store"], DOCUMENTS, k=3, weights=(0.5, 0.5)).invoke(query))


def demo_ensemble(ctx):
    query = "How do I split documents and evaluate RAG?"
    print(f"  Query: {query!r}  (3-way ensemble: BM25 + Similarity + MMR)")
    retriever = build_multi_strategy_ensemble_retriever(ctx["store"], DOCUMENTS, k=3, weights=(0.4, 0.3, 0.3))
    show(retriever.invoke(query)[:3])


def demo_multi_query(ctx):
    query = "ways to break up big texts"
    print(f"  Query: {query!r}  (LLM writes extra versions of the question, returns union)")
    show(build_multi_query_retriever(ctx["store"], k=2).invoke(query))


def demo_fusion(ctx):
    query = "ways to break up big texts"
    print(f"  Query: {query!r}  (RAG-Fusion: LLM query variations + Reciprocal Rank Fusion)")
    retriever = build_fusion_retriever(ctx["store"], k=3, fetch_k=5, num_queries=3)
    docs = retriever.invoke(query)
    print(f"  Generated queries: {retriever.last_generated_queries}")
    show(docs)


def demo_hyde(ctx):
    query = "How do we check if a RAG system's answers are hallucinated?"
    print(f"  Query: {query!r}  (HyDE: LLM writes a hypothetical passage, then embeds it)")
    retriever = build_hyde_retriever(ctx["store"], k=2)
    docs = retriever.invoke(query)
    print(f"  Hypothetical passage: {retriever.last_hypothetical_document!r}")
    show(docs)


def demo_reranking(ctx):
    query = "How do we combine keyword and vector search and reorder results?"
    print(f"  Query: {query!r}  (fetch 5 candidates, Gemini LLM reranker keeps top 3)")
    retriever = build_reranking_retriever(ctx["store"], mode="llm", top_n=3, fetch_k=5)
    show(retriever.invoke(query))


def demo_compression_embeddings(ctx):
    query = "How do I split documents?"
    print(f"  Query: {query!r}  (fetch 4, keep those with similarity >= 0.75)")
    retriever = build_compression_retriever(
        ctx["store"], mode="embeddings", embeddings=ctx["embeddings"], similarity_threshold=0.75, k=4
    )
    show(retriever.invoke(query))


def demo_compression_llm(ctx):
    query = "What happens after retrieval in a RAG pipeline?"
    print(f"  Query: {query!r}  (LLM keeps only the relevant sentences)")
    show(build_compression_retriever(ctx["store"], mode="llm", k=2).invoke(query), preview=200)


def demo_parent_document(ctx):
    query = "What happens after retrieval in a RAG pipeline?"
    retriever = build_parent_document_retriever(DOCUMENTS, ctx["embeddings"], child_chunk_size=120, k=2)
    child = retriever.vectorstore.similarity_search(query, k=1)[0]
    print(f"  Query: {query!r}")
    print(f"  Best child chunk ({len(child.page_content)} chars): {child.page_content!r}")
    parents = retriever.invoke(query)
    print(f"  Returned parent ({len(parents[0].page_content)} chars):")
    show(parents, preview=400)


def demo_self_query(ctx):
    store = ctx["store"]
    if ctx["store_name"] not in {"chroma", "pinecone"}:
        print("  (self-query needs Chroma or Pinecone; using a local Chroma store for this demo)")
        store = build_store("chroma", ctx["embeddings"])
    query = "advanced notes about chunking or retrieval written after 2023"
    print(f"  Query: {query!r}  (LLM extracts the filter: level='advanced' AND year > 2023)")
    retriever = build_self_query_retriever(
        store, document_contents=DOCUMENT_CONTENTS, metadata_fields=METADATA_FIELDS, k=4
    )
    show(retriever.invoke(query))


DEMOS = {
    "similarity": demo_similarity,
    "mmr": demo_mmr,
    "score_threshold": demo_score_threshold,
    "metadata_filter": demo_metadata_filter,
    "bm25": demo_bm25,
    "hybrid": demo_hybrid,
    "ensemble": demo_ensemble,
    "multi_query": demo_multi_query,
    "fusion": demo_fusion,
    "hyde": demo_hyde,
    "reranking": demo_reranking,
    "compression_embeddings": demo_compression_embeddings,
    "compression_llm": demo_compression_llm,
    "parent_document": demo_parent_document,
    "self_query": demo_self_query,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retriever demos")
    parser.add_argument("retrievers", nargs="*", default=DEFAULT_RETRIEVERS,
                        help="retriever names to run, or 'all' (default: all)")
    parser.add_argument("--store", default=DEFAULT_STORE, choices=["memory", "faiss", "chroma", "pinecone"],
                        help="vector store to search (default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="list retriever names and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(DEMOS))
        return
    names = list(DEMOS) if "all" in args.retrievers else args.retrievers
    unknown = [n for n in names if n not in DEMOS]
    if unknown:
        parser.error(f"unknown retriever(s): {', '.join(unknown)}. Use --list to see the options.")

    load_dotenv()
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")

    embeddings = build_gemini_embeddings()
    print(f"Loading {len(DOCUMENTS)} notes into the '{args.store}' vector store (Gemini embeddings)...")
    ctx = {"store": build_store(args.store, embeddings), "embeddings": embeddings, "store_name": args.store}

    failures = []
    for name in names:
        print(f"\n=== {name} ===")
        try:
            DEMOS[name](ctx)
        except Exception as exc:  # noqa: BLE001 - report the error and keep going.
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append(name)

    print("\nDone." if not failures else f"\nDone, but these failed: {', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
