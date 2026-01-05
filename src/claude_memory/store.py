"""ChromaDB vector store for semantic search."""

from dataclasses import dataclass
from typing import Callable

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, ensure_dirs
from .chunker import load_all_chunks


# Batch size for embedding operations (balances memory usage and speed)
BATCH_SIZE = 100

# Hybrid search weights (70% vector similarity, 30% keyword matching)
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


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
    """Vector store for episodic memory with hybrid search."""

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
        # BM25 index (lazy-loaded)
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[str] | None = None
        self._bm25_ids: list[str] | None = None

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the collection."""
        if self._collection.count() == 0:
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_ids = []
            return

        # Get all documents from the collection
        results = self._collection.get()
        self._bm25_ids = results["ids"]
        self._bm25_docs = results["documents"]

        # Tokenize documents for BM25 (simple whitespace tokenization, lowercase)
        tokenized = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    def _invalidate_bm25_cache(self) -> None:
        """Invalidate the BM25 cache (call after adding new documents)."""
        self._bm25 = None
        self._bm25_docs = None
        self._bm25_ids = None

    def count(self) -> int:
        """Get the number of chunks in the collection."""
        return self._collection.count()

    def rebuild_index(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Rebuild the index from chunks.jsonl. Returns count of indexed chunks.

        Args:
            progress_callback: Optional callback(current, total) for progress updates.
        """
        chunks = load_all_chunks()

        if not chunks:
            return 0

        # Get existing IDs
        existing = set(self._collection.get()["ids"])

        # Filter to new chunks only
        new_chunks = [c for c in chunks if c.id not in existing]

        if not new_chunks:
            return 0

        total = len(new_chunks)

        # Process in batches for memory efficiency
        for i in range(0, total, BATCH_SIZE):
            batch = new_chunks[i : i + BATCH_SIZE]

            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "session_id": c.session_id,
                        "timestamp": c.timestamp,
                        "chunk_type": c.chunk_type,
                        "turn_index": c.turn_index,
                    }
                    for c in batch
                ],
            )

            if progress_callback:
                progress_callback(min(i + len(batch), total), total)

        # Invalidate BM25 cache since we added new documents
        self._invalidate_bm25_cache()

        return total

    def search(self, query: str, n: int = 5, hybrid: bool = True) -> list[SearchResult]:
        """Search for chunks using hybrid search (vector + BM25).

        Args:
            query: The search query.
            n: Number of results to return.
            hybrid: If True, combine vector and BM25 scores. If False, use vector only.

        Returns:
            List of SearchResult objects sorted by combined relevance.
        """
        if self._collection.count() == 0:
            return []

        # Get more candidates for reranking when using hybrid search
        candidates = n * 2 if hybrid else n

        # Vector search
        vector_results = self._collection.query(
            query_texts=[query],
            n_results=min(candidates, self._collection.count()),
        )

        if not vector_results["documents"] or not vector_results["documents"][0]:
            return []

        # Build map of id -> (doc, metadata, vector_distance)
        result_map: dict[str, dict] = {}
        for i, doc in enumerate(vector_results["documents"][0]):
            doc_id = vector_results["ids"][0][i]
            metadata = vector_results["metadatas"][0][i] if vector_results["metadatas"] else {}
            distance = vector_results["distances"][0][i] if vector_results["distances"] else 0.0

            # Convert distance to similarity score (0-1, higher is better)
            # ChromaDB uses L2 distance by default, convert to similarity
            vector_score = 1.0 / (1.0 + distance)

            result_map[doc_id] = {
                "text": doc,
                "metadata": metadata,
                "vector_score": vector_score,
                "bm25_score": 0.0,
            }

        if hybrid and self._collection.count() > 0:
            # Build BM25 index if not already built
            if self._bm25 is None:
                self._build_bm25_index()

            if self._bm25 is not None and self._bm25_ids:
                # Get BM25 scores for query
                tokenized_query = query.lower().split()
                bm25_scores = self._bm25.get_scores(tokenized_query)

                # Normalize BM25 scores to 0-1 range
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

                # Add BM25 scores to result_map for candidates we already have
                for i, doc_id in enumerate(self._bm25_ids):
                    if doc_id in result_map:
                        result_map[doc_id]["bm25_score"] = bm25_scores[i] / max_bm25

                # Also check if any high BM25 scores are missing from vector results
                for i, doc_id in enumerate(self._bm25_ids):
                    if doc_id not in result_map and bm25_scores[i] > 0:
                        normalized_bm25 = bm25_scores[i] / max_bm25
                        # Only add if BM25 score is significant
                        if normalized_bm25 > 0.3:
                            # Fetch the full document info from collection
                            doc_result = self._collection.get(ids=[doc_id])
                            if doc_result["documents"]:
                                result_map[doc_id] = {
                                    "text": doc_result["documents"][0],
                                    "metadata": doc_result["metadatas"][0] if doc_result["metadatas"] else {},
                                    "vector_score": 0.0,  # Not in vector top-k
                                    "bm25_score": normalized_bm25,
                                }

        # Calculate combined scores and sort
        scored_results = []
        for doc_id, data in result_map.items():
            if hybrid:
                combined_score = (
                    VECTOR_WEIGHT * data["vector_score"]
                    + BM25_WEIGHT * data["bm25_score"]
                )
            else:
                combined_score = data["vector_score"]

            # Convert combined score back to distance-like metric (lower is better for display)
            display_distance = 1.0 / combined_score - 1.0 if combined_score > 0 else float("inf")

            scored_results.append(
                (
                    combined_score,
                    SearchResult(
                        text=data["text"],
                        session_id=data["metadata"].get("session_id", ""),
                        timestamp=data["metadata"].get("timestamp", ""),
                        distance=display_distance,
                        chunk_type=data["metadata"].get("chunk_type", "turn"),
                        turn_index=data["metadata"].get("turn_index", 0),
                    ),
                )
            )

        # Sort by combined score (descending) and take top n
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in scored_results[:n]]

    def clear(self) -> None:
        """Clear all data from the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
        self._invalidate_bm25_cache()
