# Retrievers

One module per retrieval strategy. Every builder returns a standard LangChain retriever, so they all share the same `.invoke(query)` interface and work with any of the vector stores in [`vector_stores/`](../vector_stores/).

## Strategies included

| Module | Builder function | What it does | Needs LLM? |
| --- | --- | --- | --- |
| `similarity.py` | `build_similarity_retriever` | Top-$k$ nearest vectors by embedding similarity | No |
| `mmr.py` | `build_mmr_retriever` | Maximal Marginal Relevance: balances relevance with diversity | No |
| `score_threshold.py` | `build_score_threshold_retriever` | Only returns documents whose relevance score $\ge$ threshold | No |
| `metadata_filter.py` | `build_metadata_filter_retriever` | Restricts similarity search to documents matching metadata | No |
| `bm25.py` | `build_bm25_retriever` | Classic BM25 keyword search (no embeddings) | No |
| `hybrid.py` | `build_hybrid_retriever` | BM25 keyword + vector similarity merged with Reciprocal Rank Fusion | No |
| `ensemble.py` | `build_ensemble_retriever`, `build_multi_strategy_ensemble_retriever` | Combines *any* list of retrievers (e.g. BM25 + Similarity + MMR) via RRF | No |
| `multi_query.py` | `build_multi_query_retriever` | LLM generates alternative phrasings and takes the union of results | Yes |
| `fusion.py` | `build_fusion_retriever` | **RAG-Fusion**: LLM query expansion + Reciprocal Rank Fusion re-ranking | Yes |
| `hyde.py` | `build_hyde_retriever` | **HyDE**: LLM writes a hypothetical answer passage and embeds *that* | Yes |
| `reranking.py` | `build_reranking_retriever` | Two-stage retrieval: fast recall followed by LLM or Cross-Encoder reranking | Yes (or local Cross-Encoder) |
| `contextual_compression.py` | `build_compression_retriever` | Filters (`mode="embeddings"`) or extracts relevant sentences (`mode="llm"`) | Optional |
| `parent_document.py` | `build_parent_document_retriever` | Indexes small child chunks for precision, returns full parent documents | No |
| `self_query.py` | `build_self_query_retriever` | LLM parses natural-language questions into a search query + metadata filter | Yes |

## Install

```bash
pip install rank_bm25 lark
```

Make sure `GOOGLE_API_KEY` is set in `.env` at the repository root (used for Gemini embeddings and the LLM-powered retrievers).

## Run the examples

```bash
python -m retrievers.example                          # run all 15 demos (FAISS default)
python -m retrievers.example fusion hyde reranking    # run specific retrievers
python -m retrievers.example all --store chroma       # run against Chroma / memory / pinecone
python -m retrievers.example --list                   # list all demo names
```

