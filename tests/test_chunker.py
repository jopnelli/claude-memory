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
        """Long conversation should produce multiple substantial chunks.

        Note: With chunking improvements, long exchanges may be split into
        multiple smaller chunks to fit within the embedding model's token limit.
        """
        from claude_memory.chunker import chunk_conversation, MAX_CHUNK_CHARS

        chunks = list(chunk_conversation(long_conversation))

        # 6 exchanges, but long ones get split - expect >= 6 chunks
        assert len(chunks) >= 6

        # Each chunk should have meaningful content but stay within limits
        for chunk in chunks:
            assert len(chunk.text) > 50  # Meaningful content
            assert len(chunk.text) <= MAX_CHUNK_CHARS + 300  # Allow some buffer for overlap
            # Note: Split chunks in the middle of an exchange may not have User:/Assistant: labels

        # Verify split chunks have correct metadata
        split_chunks = [c for c in chunks if c.total_chunks > 1]
        for chunk in split_chunks:
            assert chunk.parent_turn_id  # Should have parent reference
            assert chunk.chunk_index >= 0
            assert chunk.chunk_index < chunk.total_chunks

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

    def test_load_deduplicates_by_id(self, temp_dir, monkeypatch):
        """Load deduplicates chunks with same ID, keeping last occurrence."""
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

        # Write chunks with duplicate IDs (simulating git merge conflict)
        with open(CHUNKS_FILE, "w") as f:
            # First occurrence
            f.write(json.dumps({
                "id": "chunk-1",
                "text": "User: Hi\n\nAssistant: Hello v1",
                "timestamp": "2025-01-15T10:00:00Z",
                "session_id": "test",
            }) + "\n")
            # Unique chunk
            f.write(json.dumps({
                "id": "chunk-2",
                "text": "User: Bye\n\nAssistant: Goodbye",
                "timestamp": "2025-01-15T10:01:00Z",
                "session_id": "test",
            }) + "\n")
            # Duplicate of chunk-1 (should replace first)
            f.write(json.dumps({
                "id": "chunk-1",
                "text": "User: Hi\n\nAssistant: Hello v2",
                "timestamp": "2025-01-15T10:00:00Z",
                "session_id": "test",
            }) + "\n")

        chunks = load_all_chunks()

        # Should have 2 unique chunks
        assert len(chunks) == 2

        # chunk-1 should have the last version (v2)
        chunk_1 = next(c for c in chunks if c.id == "chunk-1")
        assert "Hello v2" in chunk_1.text


class TestRecursiveSplit:
    """Tests for recursive_split function."""

    def test_short_text_not_split(self):
        """Text shorter than MAX_CHUNK_CHARS should not be split."""
        from claude_memory.chunker import recursive_split, MAX_CHUNK_CHARS

        short_text = "This is a short text."
        assert len(short_text) < MAX_CHUNK_CHARS

        parts = recursive_split(short_text)
        assert len(parts) == 1
        assert parts[0] == short_text

    def test_long_text_split_at_paragraphs(self):
        """Long text should be split at paragraph boundaries first."""
        from claude_memory.chunker import recursive_split, MAX_CHUNK_CHARS

        # Create text with paragraphs that exceeds MAX_CHUNK_CHARS
        paragraph = "A" * 500  # 500 char paragraph
        long_text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"
        assert len(long_text) > MAX_CHUNK_CHARS

        parts = recursive_split(long_text)
        assert len(parts) > 1
        # Each part should be under the limit
        for part in parts:
            assert len(part) <= MAX_CHUNK_CHARS

    def test_long_text_split_at_lines(self):
        """Text without paragraphs should split at line boundaries."""
        from claude_memory.chunker import recursive_split, MAX_CHUNK_CHARS

        # Create text with lines (no paragraph breaks)
        line = "B" * 200  # 200 char line
        long_text = "\n".join([line] * 10)  # 10 lines
        assert len(long_text) > MAX_CHUNK_CHARS

        parts = recursive_split(long_text)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= MAX_CHUNK_CHARS

    def test_long_text_split_at_sentences(self):
        """Text without line breaks should split at sentences."""
        from claude_memory.chunker import recursive_split, MAX_CHUNK_CHARS

        # Create text with sentences (no line breaks)
        sentence = "C" * 150  # 150 char sentence
        long_text = ". ".join([sentence] * 15)  # Many sentences
        assert len(long_text) > MAX_CHUNK_CHARS

        parts = recursive_split(long_text)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= MAX_CHUNK_CHARS

    def test_very_long_word_hard_split(self):
        """Single very long word should be hard split."""
        from claude_memory.chunker import recursive_split, MAX_CHUNK_CHARS

        # Create a single "word" longer than MAX_CHUNK_CHARS
        very_long = "X" * (MAX_CHUNK_CHARS + 500)

        parts = recursive_split(very_long)
        assert len(parts) > 1
        # All parts except possibly the last should be MAX_CHUNK_CHARS or less
        for part in parts[:-1]:
            assert len(part) <= MAX_CHUNK_CHARS


class TestAddOverlap:
    """Tests for add_overlap function."""

    def test_single_chunk_no_overlap(self):
        """Single chunk should not get overlap."""
        from claude_memory.chunker import add_overlap

        chunks = ["Only one chunk"]
        result = add_overlap(chunks)
        assert result == chunks

    def test_two_chunks_get_overlap(self):
        """Second chunk should have overlap from first."""
        from claude_memory.chunker import add_overlap, OVERLAP_CHARS

        chunks = ["First chunk with some content" * 50, "Second chunk starts here"]
        result = add_overlap(chunks)

        assert len(result) == 2
        assert result[0] == chunks[0]  # First unchanged
        # Second should start with end of first
        assert result[1].startswith(chunks[0][-OVERLAP_CHARS:])

    def test_empty_list(self):
        """Empty list should return empty."""
        from claude_memory.chunker import add_overlap

        result = add_overlap([])
        assert result == []


class TestCreateChunksWithContext:
    """Tests for create_chunks_with_context function."""

    def test_short_exchange_single_chunk(self):
        """Short exchange should produce single chunk."""
        from claude_memory.chunker import create_chunks_with_context
        from claude_memory.parser import Message

        user = Message(
            role="user",
            content="Short question",
            uuid="user-1",
            timestamp="2025-01-15T10:00:00Z",
            session_id="test",
        )
        assistant = Message(
            role="assistant",
            content="Short answer",
            uuid="asst-1",
            timestamp="2025-01-15T10:00:01Z",
            session_id="test",
        )

        chunks = create_chunks_with_context([(user, assistant)], 0)

        assert len(chunks) == 1
        assert chunks[0].total_chunks == 1
        assert chunks[0].parent_turn_id == ""  # Not split
        assert chunks[0].id == "asst-1"

    def test_long_exchange_multiple_chunks(self):
        """Long exchange should produce multiple chunks with metadata."""
        from claude_memory.chunker import create_chunks_with_context, MAX_CHUNK_CHARS
        from claude_memory.parser import Message

        # Create a very long response
        long_response = "This is a very detailed response. " * 200  # ~6400 chars
        assert len(long_response) > MAX_CHUNK_CHARS

        user = Message(
            role="user",
            content="Tell me everything about this topic",
            uuid="user-1",
            timestamp="2025-01-15T10:00:00Z",
            session_id="test",
        )
        assistant = Message(
            role="assistant",
            content=long_response,
            uuid="asst-1",
            timestamp="2025-01-15T10:00:01Z",
            session_id="test",
        )

        chunks = create_chunks_with_context([(user, assistant)], 0)

        assert len(chunks) > 1  # Should be split
        for i, chunk in enumerate(chunks):
            assert chunk.parent_turn_id == "asst-1"  # Track parent
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)
            assert chunk.id == f"asst-1-{i}"  # Unique ID for each part
