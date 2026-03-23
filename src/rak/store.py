from __future__ import annotations

from pathlib import Path

import chromadb

from rak.errors import DimensionMismatchError


class VectorStore:
    COLLECTION_NAME = "rak_papers"

    def __init__(self, persist_dir: Path, dimension: int = 384) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._dimension = dimension
        self._validate_dimension()

    def _validate_dimension(self) -> None:
        """Check that stored embeddings match the expected dimension."""
        if self._collection.count() == 0:
            return
        sample = self._collection.peek(limit=1, include=["embeddings"])
        if sample["embeddings"]:
            stored_dim = len(sample["embeddings"][0])
            if stored_dim != self._dimension:
                raise DimensionMismatchError(expected=stored_dim, got=self._dimension)

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def search(self, query_embedding: list[float], limit: int = 10, where: dict | None = None) -> list[dict]:
        total = self._collection.count()
        if total == 0:
            return []
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(limit, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": max(0.0, 1.0 - results["distances"][0][i]),
            })
        return items

    def count(self) -> int:
        return self._collection.count()

    def get(self, ids: list[str], include: list[str] | None = None) -> dict:
        """Retrieve documents/metadatas by IDs."""
        return self._collection.get(ids=ids, include=include or ["documents"])

    def get_ids_by_metadata(self, where: dict) -> dict:
        """Query document IDs by metadata filter."""
        return self._collection.get(where=where, include=[])

    def get_embedding(self, doc_id: str) -> list[float] | None:
        """Retrieve the embedding vector for a document ID."""
        result = self._collection.get(ids=[doc_id], include=["embeddings"])
        if not result["ids"]:
            return None
        return result["embeddings"][0]

    def has(self, doc_id: str) -> bool:
        result = self._collection.get(ids=[doc_id], include=[])
        return len(result["ids"]) > 0

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
        )
