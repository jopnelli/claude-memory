"""Tests for the chunker module."""

import json
from pathlib import Path

import pytest


class TestChunkConversation:
    """Tests for chunk_conversation function."""

    def test_chunk_short_conversation(self, short_conversation):
        """Single exchange produces one chunk."""
        # Import here to avoid config issues before monkeypatch
        from claude_memory.chunker import chunk_conversation

        chunks = list(chunk_conversation(short_conversation))

        assert len(chunks) == 1
        chunk = chunks[0]
        assert "User:" in chunk.text
        assert "Assistant:" in chunk.text
        assert "authentication" in chunk.text.lower()
        assert chunk.session_id == "short-conversation"

    def test_chunk_multi_exchange(self, multi_exchange):
        """Multiple exchanges produce multiple chunks."""
        from claude_memory.chunker import chunk_conversation

        chunks = list(chunk_conversation(multi_exchange))

        assert len(chunks) == 3

        # Each chunk should have both User and Assistant parts
        for chunk in chunks:
            assert "User:" in chunk.text
            assert "Assistant:" in chunk.text

    def test_chunk_file_history_only(self, file_history_only):
        """File with no exchanges produces no chunks."""
        from claude_memory.chunker import chunk_conversation

        chunks = list(chunk_conversation(file_history_only))
        assert len(chunks) == 0

    def test_chunk_has_required_fields(self, short_conversation):
        """Each chunk should have all required fields."""
        from claude_memory.chunker import Chunk, chunk_conversation

        chunks = list(chunk_conversation(short_conversation))

        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.id  # UUID
            assert chunk.text  # Combined text
            assert chunk.timestamp
            assert chunk.session_id

    def test_chunk_long_conversation(self, long_conversation):
        """Long conversation should produce multiple substantial chunks."""
        from claude_memory.chunker import chunk_conversation

        chunks = list(chunk_conversation(long_conversation))

        # 6 exchanges = 6 chunks
        assert len(chunks) == 6

        # Each chunk should have substantial content
        for chunk in chunks:
            assert len(chunk.text) > 200  # Meaningful content
            assert "User:" in chunk.text
            assert "Assistant:" in chunk.text

        # Total text should be substantial
        total_text = sum(len(c.text) for c in chunks)
        assert total_text > 5000  # Long conversation has lots of content


class TestCreateChunk:
    """Tests for create_chunk function."""

    def test_create_chunk_format(self):
        """Chunk text should have correct format."""
        from claude_memory.chunker import create_chunk
        from claude_memory.parser import Message

        user = Message(
            role="user",
            content="What is Python?",
            uuid="user-uuid",
            timestamp="2025-01-15T10:00:00Z",
            session_id="test",
        )
        assistant = Message(
            role="assistant",
            content="Python is a programming language.",
            uuid="asst-uuid",
            timestamp="2025-01-15T10:00:01Z",
            session_id="test",
        )

        chunk = create_chunk(user, assistant)

        assert chunk.id == "asst-uuid"  # Uses assistant UUID
        assert chunk.text == "User: What is Python?\n\nAssistant: Python is a programming language."
        assert chunk.timestamp == "2025-01-15T10:00:01Z"
        assert chunk.session_id == "test"


class TestSyncChunks:
    """Tests for sync_chunks and related functions.

    Note: These tests require careful module reloading because config.py
    has module-level constants that are evaluated at import time.
    """

    def test_sync_new_conversation(self, temp_dir, sample_conversation_data, monkeypatch):
        """Syncing a new conversation should create chunks."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        monkeypatch.setenv("CLAUDE_MEMORY_STORAGE", str(storage_dir))
        monkeypatch.setenv("CLAUDE_MEMORY_PROJECT", str(project_dir))

        # Reimport everything with fresh config
        import importlib
        import claude_memory.config
        importlib.reload(claude_memory.config)
        import claude_memory.parser
        importlib.reload(claude_memory.parser)
        import claude_memory.chunker
        importlib.reload(claude_memory.chunker)

        from claude_memory.chunker import sync_chunks, load_all_chunks
        from claude_memory.config import CHUNKS_FILE, ensure_dirs

        ensure_dirs()

        # Create a test conversation
        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        # Sync
        new_chunks, new_files = sync_chunks()

        assert new_chunks == 1
        assert new_files == 1
        assert CHUNKS_FILE.exists()

        # Verify chunk content
        chunks = load_all_chunks()
        assert len(chunks) == 1
        assert "Hello" in chunks[0].text
        assert "Hi there!" in chunks[0].text

    def test_sync_skips_already_processed(self, temp_dir, sample_conversation_data, monkeypatch):
        """Second sync should skip already-processed files."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        monkeypatch.setenv("CLAUDE_MEMORY_STORAGE", str(storage_dir))
        monkeypatch.setenv("CLAUDE_MEMORY_PROJECT", str(project_dir))

        import importlib
        import claude_memory.config
        importlib.reload(claude_memory.config)
        import claude_memory.parser
        importlib.reload(claude_memory.parser)
        import claude_memory.chunker
        importlib.reload(claude_memory.chunker)
        from claude_memory.chunker import sync_chunks
        from claude_memory.config import ensure_dirs

        ensure_dirs()

        # Create a test conversation
        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        # First sync
        new_chunks1, new_files1 = sync_chunks()
        assert new_chunks1 == 1

        # Second sync - should skip
        new_chunks2, new_files2 = sync_chunks()
        assert new_chunks2 == 0
        assert new_files2 == 0

    def test_sync_detects_modified_file(self, temp_dir, sample_conversation_data, monkeypatch):
        """Sync should detect and reprocess modified files."""
        from conftest import write_jsonl
        import time

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        monkeypatch.setenv("CLAUDE_MEMORY_STORAGE", str(storage_dir))
        monkeypatch.setenv("CLAUDE_MEMORY_PROJECT", str(project_dir))

        import importlib
        import claude_memory.config
        importlib.reload(claude_memory.config)
        import claude_memory.parser
        importlib.reload(claude_memory.parser)
        import claude_memory.chunker
        importlib.reload(claude_memory.chunker)
        from claude_memory.chunker import sync_chunks, load_all_chunks
        from claude_memory.config import ensure_dirs

        ensure_dirs()

        # Create a test conversation
        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        # First sync
        sync_chunks()

        # Modify the file (add another exchange)
        time.sleep(0.01)  # Ensure mtime changes
        extended_data = sample_conversation_data + [
            {"type": "user", "uuid": "u2", "timestamp": "2025-01-15T10:01:00Z",
             "sessionId": "test-session", "message": {"role": "user", "content": "Follow up"}},
            {"type": "assistant", "uuid": "a2", "timestamp": "2025-01-15T10:01:01Z",
             "sessionId": "test-session", "message": {"role": "assistant",
             "content": [{"type": "text", "text": "Response"}]}},
        ]
        write_jsonl(conv_file, extended_data)

        # Second sync - should detect new chunk
        new_chunks, _ = sync_chunks()
        assert new_chunks == 1

        # Should now have 2 chunks total
        chunks = load_all_chunks()
        assert len(chunks) == 2


class TestLoadAllChunks:
    """Tests for load_all_chunks function."""

    def test_load_empty(self, temp_dir, monkeypatch):
        """Loading from nonexistent file returns empty list."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        monkeypatch.setenv("CLAUDE_MEMORY_STORAGE", str(storage_dir))

        import importlib
        import claude_memory.config
        importlib.reload(claude_memory.config)
        import claude_memory.chunker
        importlib.reload(claude_memory.chunker)
        from claude_memory.chunker import load_all_chunks

        chunks = load_all_chunks()
        assert chunks == []

    def test_load_existing_chunks(self, temp_dir, monkeypatch):
        """Load chunks from existing file."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        monkeypatch.setenv("CLAUDE_MEMORY_STORAGE", str(storage_dir))

        import importlib
        import claude_memory.config
        importlib.reload(claude_memory.config)
        import claude_memory.chunker
        importlib.reload(claude_memory.chunker)
        from claude_memory.chunker import load_all_chunks
        from claude_memory.config import CHUNKS_FILE, ensure_dirs

        ensure_dirs()

        # Write some chunks
        with open(CHUNKS_FILE, "w") as f:
            f.write(json.dumps({
                "id": "chunk-1",
                "text": "User: Hi\n\nAssistant: Hello",
                "timestamp": "2025-01-15T10:00:00Z",
                "session_id": "test",
            }) + "\n")

        chunks = load_all_chunks()
        assert len(chunks) == 1
        assert chunks[0].id == "chunk-1"
