# 🧠 Learning RAG

<p align="center">
  <strong>A hands-on playground for understanding Retrieval-Augmented Generation (RAG), from the building blocks to advanced retrieval and self-correcting pipelines.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Focus-Learning%20by%20Building-blueviolet" alt="Learning by building" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
</p>

Explore RAG by implementing its core components yourself: document loading, chunking, embeddings, vector stores, retrieval strategies, and advanced pipelines such as **Corrective RAG (CRAG)** and **Self-RAG**.

The goal is to make each concept easy to explore, run, and extend—not hide everything behind one large chain.

---

## 🗺️ Learning roadmap

| Stage | What you'll learn | Where to look |
| --- | --- | --- |
| 1. Load documents | Bring text and structured files into your pipeline | [`document_loaders/`](./document_loaders/) |
| 2. Split & chunk | Turn long documents into useful passages | [`text_splitters/`](./text_splitters/) |
| 3. Store knowledge | Work with embeddings and vector databases | [`vector_stores/`](./vector_stores/) |
| 4. Retrieve context | Compare similarity, BM25, hybrid search, MMR, HyDE, and more | [`retrievers/`](./retrievers/) |
| 5. Correct retrieval | Evaluate retrieved context and use refinement or web fallback | [`crag/`](./crag/) |
| 6. Reflect on answers | Score relevance, support, and answer usefulness | [`self_rag/`](./self_rag/) |

## ✨ What's inside?

### 📄 Document loading & chunking
- Explore document-loader examples and sample files.
- Learn how text splitting and chunk size affect retrieval.

### 🧮 Vector stores
A shared interface for experimenting with different storage backends:

- **FAISS** — local vector search.
- **Chroma** — persistent local collections.
- **In-memory** — lightweight experiments.
- **Pinecone** — hosted vector search.

Embedding helpers include Google Gemini embeddings and an optional local Hugging Face embedding path. Keep embedding models consistent within a collection or index.

### 🔎 Retrieval strategies
The `retrievers/` package includes examples of:
- Similarity search, MMR, and score thresholds
- Metadata filtering and BM25 keyword retrieval
- Hybrid search and ensemble retrieval with Reciprocal Rank Fusion
- Multi-query retrieval, RAG-Fusion, and HyDE
- Reranking, contextual compression, parent-document retrieval, and self-query retrieval

See the [retrievers guide](./retrievers/README.md) for the strategy table and runnable commands.

### 🛠️ Corrective RAG (CRAG)
A retrieval evaluator checks whether the retrieved context is useful. The pipeline can refine relevant content or use query rewriting and a fallback search path when retrieval is weak.

➡️ [Explore CRAG](./crag/README.md)

### 🔁 Self-RAG
A self-reflective pipeline evaluates whether retrieval is needed, checks passage relevance, and scores answer support and usefulness before choosing a response.

➡️ [Explore Self-RAG](./self_rag/README.md)

---

## 🚀 Get started

### 1. Clone the repository

```bash
git clone https://github.com/rishiberi17988/learning_rag.git
cd learning_rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API keys

Some examples use Google Gemini for embeddings or LLM calls. Create a `.env` file in the repository root and add your key:

```dotenv
GOOGLE_API_KEY=your-gemini-api-key
```

Some integrations may require additional provider credentials, such as `PINECONE_API_KEY`. Only configure the services used by the example you're running. **Never commit real API keys or secrets.**

### 5. Run an example

Run commands from the repository root:

```bash
# Retrieval strategies
python -m retrievers.example --list
python -m retrievers.example similarity bm25 hybrid
python -m retrievers.example fusion hyde reranking

# Vector stores
python -m vector_stores.example memory
python -m vector_stores.example faiss

# Advanced RAG pipelines
python -m crag.example
python -m self_rag.example
```

For provider-specific installation and available options, see the README inside each module.

---

## 🧱 Repository structure

```text
learning_rag/
├── document_loaders/     # Loading documents + sample inputs
├── text_splitters/       # Text splitting and chunking notebook
├── vector_stores/        # Embeddings and vector-store implementations
├── retrievers/            # Retrieval strategies and runnable demos
├── crag/                  # Corrective RAG pipeline
├── self_rag/              # Self-reflective RAG pipeline
├── requirements.txt
└── LICENSE
```

## 🎯 Suggested study order

1. Start with document loading and text chunking.
2. Explore embeddings and compare local vector stores.
3. Run basic similarity search, then compare BM25 and hybrid retrieval.
4. Try MMR, reranking, query expansion, and contextual compression.
5. Trace the CRAG pipeline to understand retrieval evaluation and fallback.
6. Trace Self-RAG to see how relevance, support, and usefulness influence answer selection.

**Tip:** Change one variable at a time—such as chunk size, top-`k`, or retrieval strategy—and compare the retrieved passages. This makes the trade-offs much easier to understand.

## 🧰 Tech stack

- Python
- LangChain
- Google Gemini embeddings / LLM integrations
- FAISS, Chroma, and Pinecone integrations
- Sentence Transformers (optional local embeddings)
- Jupyter notebooks

## 🔐 Notes

- Keep API keys in environment variables or a local `.env` file; never commit secrets.
- Don't mix vectors from different embedding models in the same collection or index.
- Hosted services may have quotas or costs. Check provider terms before use.
- These modules are learning implementations; review configuration and provider requirements before using them in production.

## 📚 References

- [LangChain documentation](https://python.langchain.com/docs/)
- [Corrective RAG paper](https://arxiv.org/abs/2401.15884)
- [Self-RAG paper](https://arxiv.org/abs/2310.11511)

---

<p align="center">
  <i>Learn the pieces. Understand the trade-offs. Build better RAG systems.</i>
</p>
