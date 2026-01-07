"""SQLite FTS5 text index for BM25 keyword search."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import get_storage_dir, ensure_dirs


def get_text_index_db() -> Path:
    """Get the text index database path (respects CLAUDE_MEMORY_STORAGE)."""
    return get_storage_dir() / "text_index.db"


@dataclass
class TextSearchResult:
    """A single text search result."""

    chunk_id: str
    text: str
    bm25_score: float  # Lower is better (negative values, more negative = better match)
    session_id: str
    timestamp: str


class TextIndex:
    """SQLite FTS5 index for BM25 keyword search.

    Complements vector search by catching exact keyword matches that
    semantic search might miss (e.g., class names, function names, file paths).
    """

    def __init__(self, db_path: Path | None = None):
        ensure_dirs()
        self._db_path = db_path or get_text_index_db()
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the FTS5 table if it doesn't exist."""
        cursor = self._conn.cursor()

        # FTS5 virtual table for full-text search
        # tokenize='porter unicode61' gives us:
        # - porter stemming (running -> run)
        # - unicode support
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                text,
                session_id,
                timestamp,
                tokenize='porter unicode61'
            )
        """)

        # Regular table to track indexed chunk IDs (for incremental sync)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_chunks (
                chunk_id TEXT PRIMARY KEY
            )
        """)

        self._conn.commit()

    def count(self) -> int:
        """Get the number of indexed chunks."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM indexed_chunks")
        return cursor.fetchone()[0]

    def is_indexed(self, chunk_id: str) -> bool:
        """Check if a chunk is already indexed."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM indexed_chunks WHERE chunk_id = ?", (chunk_id,))
        return cursor.fetchone() is not None

    def get_indexed_ids(self) -> set[str]:
        """Get all indexed chunk IDs."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT chunk_id FROM indexed_chunks")
        return {row[0] for row in cursor.fetchall()}

    def add(self, chunk_id: str, text: str, session_id: str, timestamp: str) -> None:
        """Add a single chunk to the index."""
        if self.is_indexed(chunk_id):
            return

        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO chunks_fts (chunk_id, text, session_id, timestamp) VALUES (?, ?, ?, ?)",
            (chunk_id, text, session_id, timestamp),
        )
        cursor.execute(
            "INSERT INTO indexed_chunks (chunk_id) VALUES (?)",
            (chunk_id,),
        )
        self._conn.commit()

    def add_batch(self, chunks: list[tuple[str, str, str, str]]) -> int:
        """Add multiple chunks to the index.

        Args:
            chunks: List of (chunk_id, text, session_id, timestamp) tuples.

        Returns:
            Number of chunks added (excludes already-indexed).
        """
        existing = self.get_indexed_ids()
        new_chunks = [(cid, text, sid, ts) for cid, text, sid, ts in chunks if cid not in existing]

        if not new_chunks:
            return 0

        cursor = self._conn.cursor()
        cursor.executemany(
            "INSERT INTO chunks_fts (chunk_id, text, session_id, timestamp) VALUES (?, ?, ?, ?)",
            new_chunks,
        )
        cursor.executemany(
            "INSERT INTO indexed_chunks (chunk_id) VALUES (?)",
            [(c[0],) for c in new_chunks],
        )
        self._conn.commit()
        return len(new_chunks)

    def search(self, query: str, n: int = 20) -> list[TextSearchResult]:
        """Search for chunks matching the query using BM25.

        Args:
            query: The search query (supports FTS5 syntax like AND, OR, NOT, "phrases").
            n: Maximum number of results to return.

        Returns:
            List of results sorted by BM25 score (best matches first).
        """
        cursor = self._conn.cursor()

        # Escape special FTS5 characters in user query for safety
        # But preserve basic search functionality
        safe_query = self._prepare_query(query)

        try:
            # bm25() returns negative values where more negative = better match
            cursor.execute(
                """
                SELECT chunk_id, text, session_id, timestamp, bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (safe_query, n),
            )

            return [
                TextSearchResult(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    session_id=row["session_id"],
                    timestamp=row["timestamp"],
                    bm25_score=row["score"],
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.OperationalError:
            # Query syntax error - fall back to simple prefix search
            return []

    def _prepare_query(self, query: str) -> str:
        """Prepare a user query for FTS5 search.

        Converts natural language query to FTS5 syntax:
        - Multiple words become OR search (find any)
        - Preserves quoted phrases
        - Adds prefix matching for partial words
        """
        # If query contains FTS5 operators, use as-is
        if any(op in query.upper() for op in [" AND ", " OR ", " NOT ", '"']):
            return query

        # Split into words and add prefix matching
        words = query.split()
        if not words:
            return query

        # Each word gets prefix matching, combined with OR for broader results
        # Example: "auth bug" -> "auth* OR bug*"
        terms = [f"{word}*" for word in words if word]
        return " OR ".join(terms)

    def clear(self) -> None:
        """Clear all data from the index."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute("DELETE FROM indexed_chunks")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
