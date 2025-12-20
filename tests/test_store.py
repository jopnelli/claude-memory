"""Tests for the store module."""

import json

import pytest


def reload_modules(monkeypatch, storage_dir):
    """Helper to reload config and store modules with new paths."""
    monkeypatch.setenv("CLAUDE_MEMORY_STORAGE", str(storage_dir))

    import importlib
    import claude_memory.config
    importlib.reload(claude_memory.config)
    import claude_memory.chunker
    importlib.reload(claude_memory.chunker)
    import claude_memory.store
    importlib.reload(claude_memory.store)

    from claude_memory.store import Store
    from claude_memory.config import CHUNKS_FILE, ensure_dirs

    ensure_dirs()
    return Store, CHUNKS_FILE


class TestStore:
    """Tests for the Store class."""

    def test_empty_store_count(self, temp_dir, monkeypatch):
        """New store should have zero chunks."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, _ = reload_modules(monkeypatch, storage_dir)
        store = Store()
        assert store.count() == 0

    def test_rebuild_index(self, temp_dir, monkeypatch):
        """Rebuilding index should add all chunks."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, CHUNKS_FILE = reload_modules(monkeypatch, storage_dir)

        # Write test chunks
        chunks = [
            {"id": "chunk-1", "text": "User: How do I use Python?\n\nAssistant: Python is great.",
             "timestamp": "2025-01-15T10:00:00Z", "session_id": "session-1"},
            {"id": "chunk-2", "text": "User: What about JavaScript?\n\nAssistant: JavaScript runs in browsers.",
             "timestamp": "2025-01-15T11:00:00Z", "session_id": "session-2"},
            {"id": "chunk-3", "text": "User: Database recommendations?\n\nAssistant: PostgreSQL is reliable.",
             "timestamp": "2025-01-15T12:00:00Z", "session_id": "session-3"},
        ]

        with open(CHUNKS_FILE, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        store = Store()
        indexed = store.rebuild_index()
        assert indexed == 3
        assert store.count() == 3

    def test_rebuild_index_incremental(self, temp_dir, monkeypatch):
        """Rebuilding should only add new chunks."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, CHUNKS_FILE = reload_modules(monkeypatch, storage_dir)

        # Write initial chunk
        with open(CHUNKS_FILE, "w") as f:
            f.write(json.dumps({
                "id": "chunk-1",
                "text": "User: Hi\n\nAssistant: Hello",
                "timestamp": "2025-01-15T10:00:00Z",
                "session_id": "test",
            }) + "\n")

        store = Store()
        indexed = store.rebuild_index()
        assert indexed == 1
        assert store.count() == 1

        # Add another chunk
        with open(CHUNKS_FILE, "a") as f:
            f.write(json.dumps({
                "id": "chunk-2",
                "text": "User: Bye\n\nAssistant: Goodbye",
                "timestamp": "2025-01-15T11:00:00Z",
                "session_id": "test",
            }) + "\n")

        # Rebuild should only add the new one
        indexed = store.rebuild_index()
        assert indexed == 1
        assert store.count() == 2

    def test_search_returns_results(self, temp_dir, monkeypatch):
        """Search should return relevant results."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, CHUNKS_FILE = reload_modules(monkeypatch, storage_dir)

        # Write test chunks
        chunks = [
            {"id": "chunk-1", "text": "User: How do I use Python?\n\nAssistant: Python is great for scripting.",
             "timestamp": "2025-01-15T10:00:00Z", "session_id": "session-1"},
            {"id": "chunk-2", "text": "User: What about JavaScript?\n\nAssistant: JavaScript runs in browsers.",
             "timestamp": "2025-01-15T11:00:00Z", "session_id": "session-2"},
        ]

        with open(CHUNKS_FILE, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        store = Store()
        store.rebuild_index()

        results = store.search("Python programming", n=3)

        assert len(results) > 0
        texts = [r.text for r in results]
        assert any("Python" in t for t in texts)

    def test_search_result_fields(self, temp_dir, monkeypatch):
        """Search results should have all required fields."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, CHUNKS_FILE = reload_modules(monkeypatch, storage_dir)

        with open(CHUNKS_FILE, "w") as f:
            f.write(json.dumps({
                "id": "chunk-1",
                "text": "User: Database recommendations?\n\nAssistant: PostgreSQL is reliable.",
                "timestamp": "2025-01-15T12:00:00Z",
                "session_id": "session-3",
            }) + "\n")

        store = Store()
        store.rebuild_index()

        from claude_memory.store import SearchResult

        results = store.search("database", n=1)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.text
        assert result.session_id
        assert result.timestamp
        assert isinstance(result.distance, float)

    def test_search_empty_store(self, temp_dir, monkeypatch):
        """Searching empty store returns empty list."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, _ = reload_modules(monkeypatch, storage_dir)
        store = Store()

        results = store.search("anything")
        assert results == []

    def test_search_limits_results(self, temp_dir, monkeypatch):
        """Search should respect n parameter."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, CHUNKS_FILE = reload_modules(monkeypatch, storage_dir)

        # Write 3 chunks
        for i in range(3):
            with open(CHUNKS_FILE, "a") as f:
                f.write(json.dumps({
                    "id": f"chunk-{i}",
                    "text": f"User: Question {i}?\n\nAssistant: Answer {i}.",
                    "timestamp": f"2025-01-15T{10+i}:00:00Z",
                    "session_id": "test",
                }) + "\n")

        store = Store()
        store.rebuild_index()

        results = store.search("question", n=1)
        assert len(results) == 1

        results = store.search("question", n=10)
        assert len(results) == 3  # Only 3 chunks exist

    def test_clear(self, temp_dir, monkeypatch):
        """Clear should remove all data."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, CHUNKS_FILE = reload_modules(monkeypatch, storage_dir)

        with open(CHUNKS_FILE, "w") as f:
            f.write(json.dumps({
                "id": "chunk-1",
                "text": "User: Hi\n\nAssistant: Hello",
                "timestamp": "2025-01-15T10:00:00Z",
                "session_id": "test",
            }) + "\n")

        store = Store()
        store.rebuild_index()
        assert store.count() == 1

        store.clear()
        assert store.count() == 0

    def test_search_after_clear(self, temp_dir, monkeypatch):
        """Search after clear should return empty."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        Store, CHUNKS_FILE = reload_modules(monkeypatch, storage_dir)

        with open(CHUNKS_FILE, "w") as f:
            f.write(json.dumps({
                "id": "chunk-1",
                "text": "User: Python question\n\nAssistant: Python answer",
                "timestamp": "2025-01-15T10:00:00Z",
                "session_id": "test",
            }) + "\n")

        store = Store()
        store.rebuild_index()
        store.clear()

        results = store.search("Python")
        assert results == []
