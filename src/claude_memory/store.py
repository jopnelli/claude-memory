"""ChromaDB vector store for semantic search."""

from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, ensure_dirs
from .chunker import load_all_chunks


def get_indexed_count() -> int:
    """Get ChromaDB collection count without loading the embedding model.

    This is a lightweight check that avoids the 3.5s embedding model load.
    Returns 0 if the collection doesn't exist or on any error.
    """
    if not CHROMA_DIR.exists():
        return 0
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


@dataclass
class SearchResult:
    """A single search result."""

    text: str
    session_id: str
    timestamp: str
    distance: float
    chunk_type: str = "turn"  # "turn" or "summary"
    turn_index: int = 0  # Position in conversation (-1 for summaries)


class Store:
    """Vector store for episodic memory."""

    def __init__(self):
        ensure_dirs()
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )

    def count(self) -> int:
        """Get the number of chunks in the collection."""
        return self._collection.count()

    def rebuild_index(self) -> int:
        """Rebuild the index from chunks.jsonl. Returns count of indexed chunks."""
        chunks = load_all_chunks()

        if not chunks:
            return 0

        # Get existing IDs
        existing = set(self._collection.get()["ids"])

        # Filter to new chunks only
        new_chunks = [c for c in chunks if c.id not in existing]

        if not new_chunks:
            return 0

        # Add new chunks
        self._collection.add(
            ids=[c.id for c in new_chunks],
            documents=[c.text for c in new_chunks],
            metadatas=[
                {
                    "session_id": c.session_id,
                    "timestamp": c.timestamp,
                    "chunk_type": c.chunk_type,
                    "turn_index": c.turn_index,
                }
                for c in new_chunks
            ],
        )

        return len(new_chunks)

    def search(self, query: str, n: int = 5) -> list[SearchResult]:
        """Search for chunks similar to the query."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self._collection.count()),
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = (
                    results["distances"][0][i] if results["distances"] else 0.0
                )
                search_results.append(
                    SearchResult(
                        text=doc,
                        session_id=metadata.get("session_id", ""),
                        timestamp=metadata.get("timestamp", ""),
                        distance=distance,
                        chunk_type=metadata.get("chunk_type", "turn"),
                        turn_index=metadata.get("turn_index", 0),
                    )
                )

        return search_results

    def clear(self) -> None:
        """Clear all data from the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
