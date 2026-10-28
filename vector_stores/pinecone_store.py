"""Pinecone managed vector store. Requires an account and API key.

Pinecone runs in the cloud, so your vectors live on Pinecone's servers rather
than your computer. Things to keep in mind:

* An *index* is Pinecone's name for a database of vectors. Each index has a
  fixed *dimension* that must match your embedding model (768 for the
  default Gemini setup). The class checks this and raises a clear error.
* Pinecone is *eventually consistent*: right after you add or delete
  vectors, a search may take a few seconds to reflect the change.
"""

import os
from typing import Any

from .base import BaseVectorStore


class PineconeStore(BaseVectorStore):
    """Pinecone class that creates a serverless index if it does not exist."""

    def __init__(self, embeddings: Any, *, index_name: str = "learning-rag", cloud: str = "aws", region: str = "us-east-1", metric: str = "cosine", api_key: str | None = None) -> None:
        # index_name: lowercase letters, numbers and hyphens only.
        # cloud/region: where Pinecone hosts a newly created index.
        # metric: how vector closeness is measured ("cosine" is common for text).
        super().__init__(embeddings)
        try:
            from pinecone import Pinecone, ServerlessSpec
            from langchain_pinecone import PineconeVectorStore
        except ImportError as exc:
            raise ImportError("Install Pinecone integration with: pip install pinecone langchain-pinecone") from exc

        # Use the key that was passed in, otherwise read it from the
        # environment, e.g. from PINECONE_API_KEY in your .env file.
        key = api_key or os.environ.get("PINECONE_API_KEY")
        if not key:
            raise ValueError("Set PINECONE_API_KEY or pass api_key to connect to Pinecone.")
        client = Pinecone(api_key=key)  # Talks to Pinecone's API.

        # Embed a sample text to find out how many numbers the model outputs
        # (e.g. 768 for the default Gemini setup, 384 for MiniLM).
        dimension = len(embeddings.embed_query("dimension check"))
        if not client.has_index(index_name):
            # The new index must use exactly that dimension. "Serverless"
            # means Pinecone manages the servers and you pay per use. This
            # call waits until the index is ready.
            client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        else:
            # The index already exists: make sure it was built for the same
            # vector size. Otherwise every add/search would fail with a
            # confusing error, e.g. after switching embedding models.
            existing = client.describe_index(index_name).dimension
            if existing != dimension:
                raise ValueError(
                    f"Pinecone index '{index_name}' has dimension {existing}, but the embedding "
                    f"model produces {dimension}. Use a different index_name or delete the old index."
                )
        # Handle for one specific index; useful for stats such as
        # ``self._index.describe_index_stats()``.
        self._index = client.Index(index_name)
        # LangChain wrapper that handles embedding and upserting for us.
        self._store = PineconeVectorStore(index=self._index, embedding=embeddings)

    def save(self) -> None:
        """Pinecone persists remotely as documents are written."""
        return None
