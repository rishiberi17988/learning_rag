# Provider-specific vector store classes

Each provider has its own class. They share familiar methods (`add_documents`, `similarity_search`, `similarity_search_with_score`, `as_retriever`, and `delete`) while keeping provider setup separate.

| Class | Where data lives | Free to try? | Packages |
| --- | --- | --- | --- |
| `FAISSStore` | Local files / memory | Yes | `langchain-community faiss-cpu` |
| `ChromaStore` | Local persistent directory | Yes | `langchain-chroma chromadb` |
| `InMemoryStore` | RAM until Python exits | Yes | Existing LangChain dependencies |
| `PineconeStore` | Pinecone cloud | Starter plan has a free allowance; account and API key required | `pinecone langchain-pinecone` |

Pinecone is a hosted option with a free Starter plan and usage limits. Check the current [Pinecone pricing page](https://www.pinecone.io/pricing/) before using it; plan limits and availability can change. Local FAISS and Chroma need no hosted account.

## Install

Embeddings come from Google Gemini (`gemini-embedding-001`, shortened to 768 dimensions). The project already includes `langchain-google-genai`. Put your key in `.env` at the repository root:

```bash
GOOGLE_API_KEY="your-gemini-api-key"
```

`build_local_embeddings()` (a free Hugging Face model that runs on your CPU) is still available as an offline alternative; it needs `pip install sentence-transformers`. Don't mix models in one Chroma collection or Pinecone index: vectors from different models aren't compatible.

Install only the vector store you want to try:

```bash
pip install langchain-community faiss-cpu       # FAISS
pip install langchain-chroma chromadb            # Chroma
pip install pinecone langchain-pinecone          # Pinecone
```

## Shared usage pattern

```python
from langchain_core.documents import Document
from vector_stores import FAISSStore, build_gemini_embeddings

embeddings = build_gemini_embeddings()  # Reads GOOGLE_API_KEY from the environment.
store = FAISSStore(embeddings)  # Or ChromaStore(embeddings), etc.
store.add_documents([
    Document(page_content="Chunking splits text into passages.", metadata={"source": "notes"}),
])
matches = store.similarity_search("How can I split a long text?", k=3)
for doc in matches:
    print(doc.page_content, doc.metadata)
```

## Provider-specific setup

```python
from vector_stores import ChromaStore, FAISSStore, PineconeStore, build_gemini_embeddings

embeddings = build_gemini_embeddings()

faiss = FAISSStore(embeddings)
faiss.add_documents(documents)
faiss.save("./faiss_data")
# Reload only your own trusted files: FAISSStore.load(path, embeddings, allow_dangerous_deserialization=True)

chroma = ChromaStore(embeddings, collection_name="my_notes", persist_directory="./chroma_data")
chroma.add_documents(documents)  # Chroma persists automatically.

# Set PINECONE_API_KEY in the environment before constructing this hosted option.
pinecone = PineconeStore(embeddings, index_name="my-notes", cloud="aws", region="us-east-1")
pinecone.add_documents(documents)
```

Pinecone creates an index if it does not exist, using the embedding dimension and the selected cloud and region. If the index already exists with a different dimension, `PineconeStore` raises a clear error. Gemini embedding calls count against your Google API quota. Run `python -m vector_stores.example [memory|faiss|chroma|pinecone|all]` from the repository root to try the example. Don't commit API keys or vector data folders.
