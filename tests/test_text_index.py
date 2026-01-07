"""Tests for the text_index module (BM25 keyword search)."""

import pytest
from pathlib import Path

from claude_memory.text_index import TextIndex


@pytest.fixture
def text_index(temp_dir):
    """Create a temporary text index for testing."""
    db_path = temp_dir / "test_text_index.db"
    index = TextIndex(db_path=db_path)
    yield index
    index.close()


class TestTextIndex:
    """Tests for TextIndex class."""

    def test_empty_index_count(self, text_index):
        """Empty index has zero count."""
        assert text_index.count() == 0

    def test_add_single_chunk(self, text_index):
        """Adding a chunk increases count."""
        text_index.add(
            chunk_id="chunk-1",
            text="This is a test document about authentication",
            session_id="session-1",
            timestamp="2025-01-15T10:00:00Z",
        )
        assert text_index.count() == 1

    def test_add_duplicate_chunk_ignored(self, text_index):
        """Adding same chunk ID twice doesn't duplicate."""
        text_index.add("chunk-1", "First text", "session-1", "2025-01-15T10:00:00Z")
        text_index.add("chunk-1", "Different text", "session-1", "2025-01-15T10:00:00Z")
        assert text_index.count() == 1

    def test_add_batch(self, text_index):
        """Batch add works correctly."""
        chunks = [
            ("chunk-1", "Authentication with JWT tokens", "session-1", "2025-01-15T10:00:00Z"),
            ("chunk-2", "Database connection pooling", "session-2", "2025-01-15T11:00:00Z"),
            ("chunk-3", "API rate limiting implementation", "session-3", "2025-01-15T12:00:00Z"),
        ]
        added = text_index.add_batch(chunks)
        assert added == 3
        assert text_index.count() == 3

    def test_add_batch_skips_existing(self, text_index):
        """Batch add skips already indexed chunks."""
        text_index.add("chunk-1", "First text", "session-1", "2025-01-15T10:00:00Z")

        chunks = [
            ("chunk-1", "First text again", "session-1", "2025-01-15T10:00:00Z"),
            ("chunk-2", "Second text", "session-2", "2025-01-15T11:00:00Z"),
        ]
        added = text_index.add_batch(chunks)
        assert added == 1  # Only chunk-2 was added
        assert text_index.count() == 2

    def test_search_finds_exact_match(self, text_index):
        """Search finds documents with exact keyword match."""
        text_index.add("chunk-1", "JWT authentication tokens", "session-1", "2025-01-15T10:00:00Z")
        text_index.add("chunk-2", "Database migrations", "session-2", "2025-01-15T11:00:00Z")

        results = text_index.search("JWT")
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"
        assert "JWT" in results[0].text

    def test_search_finds_partial_match(self, text_index):
        """Search with prefix matching finds partial matches."""
        text_index.add("chunk-1", "UserService class implementation", "session-1", "2025-01-15T10:00:00Z")

        results = text_index.search("User")
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"

    def test_search_returns_bm25_scores(self, text_index):
        """Search results include BM25 scores."""
        text_index.add("chunk-1", "authentication", "session-1", "2025-01-15T10:00:00Z")
        text_index.add("chunk-2", "authentication authentication authentication", "session-2", "2025-01-15T11:00:00Z")

        results = text_index.search("authentication")
        assert len(results) == 2
        # All results have scores (more negative = better match)
        for r in results:
            assert r.bm25_score < 0

    def test_search_empty_query(self, text_index):
        """Empty query returns empty results."""
        text_index.add("chunk-1", "some text", "session-1", "2025-01-15T10:00:00Z")
        results = text_index.search("")
        assert len(results) == 0

    def test_search_no_matches(self, text_index):
        """Search with no matches returns empty list."""
        text_index.add("chunk-1", "apple banana cherry", "session-1", "2025-01-15T10:00:00Z")
        results = text_index.search("zebra")
        assert len(results) == 0

    def test_search_limits_results(self, text_index):
        """Search respects n parameter."""
        chunks = [
            (f"chunk-{i}", f"authentication document {i}", f"session-{i}", "2025-01-15T10:00:00Z")
            for i in range(10)
        ]
        text_index.add_batch(chunks)

        results = text_index.search("authentication", n=3)
        assert len(results) == 3

    def test_clear(self, text_index):
        """Clear removes all data."""
        text_index.add("chunk-1", "test text", "session-1", "2025-01-15T10:00:00Z")
        assert text_index.count() == 1

        text_index.clear()
        assert text_index.count() == 0

    def test_is_indexed(self, text_index):
        """is_indexed correctly reports indexing status."""
        assert text_index.is_indexed("chunk-1") is False

        text_index.add("chunk-1", "text", "session-1", "2025-01-15T10:00:00Z")
        assert text_index.is_indexed("chunk-1") is True
        assert text_index.is_indexed("chunk-2") is False

    def test_get_indexed_ids(self, text_index):
        """get_indexed_ids returns all indexed chunk IDs."""
        text_index.add_batch([
            ("chunk-1", "text 1", "session-1", "2025-01-15T10:00:00Z"),
            ("chunk-2", "text 2", "session-2", "2025-01-15T11:00:00Z"),
        ])

        ids = text_index.get_indexed_ids()
        assert ids == {"chunk-1", "chunk-2"}

    def test_context_manager(self, temp_dir):
        """TextIndex works as context manager."""
        db_path = temp_dir / "context_test.db"

        with TextIndex(db_path=db_path) as index:
            index.add("chunk-1", "test", "session-1", "2025-01-15T10:00:00Z")
            assert index.count() == 1


class TestTextIndexQueryParsing:
    """Tests for query parsing and FTS5 syntax."""

    def test_multi_word_query_uses_or(self, text_index):
        """Multi-word queries use OR by default (find any word)."""
        text_index.add("chunk-1", "authentication system", "session-1", "2025-01-15T10:00:00Z")
        text_index.add("chunk-2", "database connection", "session-2", "2025-01-15T11:00:00Z")

        # "auth database" should find both (OR logic with prefix)
        results = text_index.search("auth database")
        assert len(results) == 2

    def test_quoted_phrase_search(self, text_index):
        """Quoted phrases search for exact phrase."""
        text_index.add("chunk-1", "user authentication flow", "session-1", "2025-01-15T10:00:00Z")
        text_index.add("chunk-2", "authentication for user", "session-2", "2025-01-15T11:00:00Z")

        # Exact phrase match
        results = text_index.search('"user authentication"')
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"
