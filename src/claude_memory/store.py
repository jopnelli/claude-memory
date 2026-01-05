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
    # Fields for tracking split chunks
    parent_turn_id: str = ""  # Original turn UUID (empty if not split)
    chunk_index: int = 0  # Position within split (0, 1, 2...)
    total_chunks: int = 1  # How many chunks this turn produced


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

    def rebuild_index(self, batch_size: int = 5000) -> int:
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

        # Add chunks in batches (ChromaDB has a max batch size ~5461)
        total_added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "session_id": c.session_id,
                        "timestamp": c.timestamp,
                        "chunk_type": c.chunk_type,
                        "turn_index": c.turn_index,
                        # New fields for split chunk tracking
                        "parent_turn_id": c.parent_turn_id,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                    }
                    for c in batch
                ],
            )
            total_added += len(batch)

        return total_added

    def search(self, query: str, n: int = 5, dedupe_splits: bool = True) -> list[SearchResult]:
        """Search for chunks similar to the query.

        Args:
            query: The search query text.
            n: Number of results to return.
            dedupe_splits: If True, deduplicate results from split chunks
                          (keeps best match from each parent turn).
        """
        if self._collection.count() == 0:
            return []

        # Fetch extra results to account for deduplication
        fetch_n = min(n * 3, self._collection.count()) if dedupe_splits else min(n, self._collection.count())

        results = self._collection.query(
            query_texts=[query],
            n_results=fetch_n,
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
                        parent_turn_id=metadata.get("parent_turn_id", ""),
                        chunk_index=metadata.get("chunk_index", 0),
                        total_chunks=metadata.get("total_chunks", 1),
                    )
                )

        # Deduplicate by parent_turn_id (keep best match from each split turn)
        if dedupe_splits:
            seen_parents: set[str] = set()
            deduped = []
            for result in search_results:
                # Use parent_turn_id if this is a split chunk, otherwise treat as unique
                key = result.parent_turn_id if result.parent_turn_id else f"__unique_{len(deduped)}"
                if key not in seen_parents:
                    seen_parents.add(key)
                    deduped.append(result)
                    if len(deduped) >= n:
                        break
            return deduped

        return search_results[:n]

    def clear(self) -> None:
        """Clear all data from the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
