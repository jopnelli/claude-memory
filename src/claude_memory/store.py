"""ChromaDB vector store for semantic search with hybrid BM25 support."""

from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, ensure_dirs
from .chunker import load_all_chunks


# Hybrid search weight: higher = more weight on vector search
# 0.7 means 70% vector, 30% BM25 keyword
HYBRID_VECTOR_WEIGHT = 0.7


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
    # Tool metadata
    tools_used: str = ""  # Comma-separated tool names
    files_touched: str = ""  # Comma-separated file paths
    commands_run: str = ""  # Comma-separated commands


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
                        # Split chunk tracking
                        "parent_turn_id": c.parent_turn_id,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                        # Tool metadata
                        "tools_used": c.tools_used,
                        "files_touched": c.files_touched,
                        "commands_run": c.commands_run,
                    }
                    for c in batch
                ],
            )
            total_added += len(batch)

        return total_added

    def _vector_search(
        self,
        query: str,
        n: int,
    ) -> list[tuple[str, float, dict]]:
        """Internal vector search returning (chunk_id, distance, metadata) tuples."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                metadata["text"] = doc  # Include text in metadata for convenience
                output.append((chunk_id, distance, metadata))

        return output

    def search(
        self,
        query: str,
        n: int = 5,
        dedupe_splits: bool = True,
        hybrid: bool = True,
    ) -> list[SearchResult]:
        """Search for chunks similar to the query using hybrid vector + keyword search.

        Args:
            query: The search query text.
            n: Number of results to return.
            dedupe_splits: If True, deduplicate results from split chunks
                          (keeps best match from each parent turn).
            hybrid: If True, combine vector search with BM25 keyword search (default).
        """
        if self._collection.count() == 0:
            return []

        # Fetch extra results to account for deduplication and filtering
        fetch_n = min(n * 5, self._collection.count()) if dedupe_splits else min(n * 2, self._collection.count())

        # Get vector search results
        vector_results = self._vector_search(query, fetch_n)

        # Build scores dict: chunk_id -> (combined_score, metadata)
        # For vector distance: lower is better, normalize to [0, 1]
        scores: dict[str, tuple[float, dict]] = {}

        if vector_results:
            # Normalize vector distances to [0, 1] where 0 is best
            max_dist = max(r[1] for r in vector_results) or 1.0
            for chunk_id, distance, metadata in vector_results:
                norm_vector = distance / max_dist  # 0 = perfect match, 1 = worst
                scores[chunk_id] = (norm_vector, metadata)

        # Add BM25 results if hybrid search is enabled
        if hybrid:
            try:
                from .text_index import TextIndex
                text_index = TextIndex()
                bm25_results = text_index.search(query, n=fetch_n)

                if bm25_results:
                    # BM25 scores are negative (more negative = better)
                    # Normalize to [0, 1] where 0 is best
                    min_score = min(r.bm25_score for r in bm25_results)
                    max_score = max(r.bm25_score for r in bm25_results)
                    score_range = max_score - min_score if max_score != min_score else 1.0

                    for result in bm25_results:
                        # Normalize: 0 = best match, 1 = worst
                        norm_bm25 = (result.bm25_score - min_score) / score_range

                        if result.chunk_id in scores:
                            # Combine scores: weighted average
                            old_score, metadata = scores[result.chunk_id]
                            combined = (HYBRID_VECTOR_WEIGHT * old_score +
                                       (1 - HYBRID_VECTOR_WEIGHT) * norm_bm25)
                            scores[result.chunk_id] = (combined, metadata)
                        else:
                            # BM25-only result: use BM25 score with penalty for no vector match
                            # This ensures vector matches still rank higher when available
                            scores[result.chunk_id] = (
                                0.5 + (1 - HYBRID_VECTOR_WEIGHT) * norm_bm25,
                                {
                                    "text": result.text,
                                    "session_id": result.session_id,
                                    "timestamp": result.timestamp,
                                    "chunk_type": "turn",
                                    "turn_index": 0,
                                    "parent_turn_id": "",
                                    "chunk_index": 0,
                                    "total_chunks": 1,
                                    "tools_used": "",
                                    "files_touched": "",
                                    "commands_run": "",
                                }
                            )
            except Exception:
                # Text index not available, fall back to vector-only
                pass

        # Sort by combined score (lower is better)
        sorted_results = sorted(scores.items(), key=lambda x: x[0][0])

        # Convert to SearchResult objects
        search_results = []
        for chunk_id, (score, metadata) in sorted_results:
            search_results.append(
                SearchResult(
                    text=metadata.get("text", ""),
                    session_id=metadata.get("session_id", ""),
                    timestamp=metadata.get("timestamp", ""),
                    distance=score,  # Combined score
                    chunk_type=metadata.get("chunk_type", "turn"),
                    turn_index=metadata.get("turn_index", 0),
                    parent_turn_id=metadata.get("parent_turn_id", ""),
                    chunk_index=metadata.get("chunk_index", 0),
                    total_chunks=metadata.get("total_chunks", 1),
                    tools_used=metadata.get("tools_used", ""),
                    files_touched=metadata.get("files_touched", ""),
                    commands_run=metadata.get("commands_run", ""),
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
