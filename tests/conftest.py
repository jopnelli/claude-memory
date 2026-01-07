"""Shared test fixtures and configuration."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    """Path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def short_conversation(fixtures_dir):
    """Path to short conversation fixture."""
    return fixtures_dir / "short-conversation.jsonl"


@pytest.fixture
def multi_exchange(fixtures_dir):
    """Path to multi-exchange conversation fixture."""
    return fixtures_dir / "multi-exchange.jsonl"


@pytest.fixture
def with_tool_calls(fixtures_dir):
    """Path to conversation with tool calls fixture."""
    return fixtures_dir / "with-tool-calls.jsonl"


@pytest.fixture
def file_history_only(fixtures_dir):
    """Path to file-history-only fixture."""
    return fixtures_dir / "file-history-only.jsonl"


@pytest.fixture
def empty_file(fixtures_dir):
    """Path to empty file fixture."""
    return fixtures_dir / "empty.jsonl"


@pytest.fixture
def long_conversation(fixtures_dir):
    """Path to long conversation fixture (6 exchanges)."""
    return fixtures_dir / "long-conversation.jsonl"


@pytest.fixture
def with_rich_tool_calls(fixtures_dir):
    """Path to conversation with rich tool usage fixture."""
    return fixtures_dir / "with-rich-tool-calls.jsonl"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_conversation_data():
    """Return sample conversation data for creating test files."""
    return [
        {"type": "user", "uuid": "u1", "timestamp": "2025-01-15T10:00:00Z",
         "sessionId": "test-session", "message": {"role": "user", "content": "Hello"}},
        {"type": "assistant", "uuid": "a1", "timestamp": "2025-01-15T10:00:01Z",
         "sessionId": "test-session", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "Hi there!"}]}},
    ]


def write_jsonl(path: Path, data: list[dict]) -> None:
    """Helper to write JSONL data to a file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
