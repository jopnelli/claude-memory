"""Tests for the parser module."""

import pytest
from claude_memory.parser import (
    Message,
    extract_text_content,
    parse_conversation,
)


class TestExtractTextContent:
    """Tests for extract_text_content function."""

    def test_user_message_string_content(self):
        """User messages have string content."""
        data = {"message": {"content": "Hello world"}}
        assert extract_text_content(data) == "Hello world"

    def test_assistant_message_text_blocks(self):
        """Assistant messages have array of blocks."""
        data = {
            "message": {
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ]
            }
        }
        assert extract_text_content(data) == "First part.\nSecond part."

    def test_assistant_message_with_tool_use(self):
        """Tool use blocks should be skipped, only text extracted."""
        data = {
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {}},
                    {"type": "text", "text": "Here's what I found."},
                ]
            }
        }
        assert extract_text_content(data) == "Here's what I found."

    def test_empty_content(self):
        """Empty content returns None."""
        assert extract_text_content({"message": {"content": None}}) is None
        assert extract_text_content({"message": {}}) is None

    def test_empty_text_blocks(self):
        """Empty text blocks return None."""
        data = {"message": {"content": [{"type": "tool_use", "name": "Bash"}]}}
        assert extract_text_content(data) is None


class TestParseConversation:
    """Tests for parse_conversation function."""

    def test_parse_short_conversation(self, short_conversation):
        """Parse a simple user+assistant exchange."""
        messages = list(parse_conversation(short_conversation))

        assert len(messages) == 2

        user_msg = messages[0]
        assert user_msg.role == "user"
        assert "authentication" in user_msg.content.lower()
        assert user_msg.session_id == "short-conversation"

        asst_msg = messages[1]
        assert asst_msg.role == "assistant"
        assert "JWT" in asst_msg.content

    def test_parse_multi_exchange(self, multi_exchange):
        """Parse conversation with multiple exchanges."""
        messages = list(parse_conversation(multi_exchange))

        # 3 user + 3 assistant = 6 messages
        assert len(messages) == 6

        user_messages = [m for m in messages if m.role == "user"]
        asst_messages = [m for m in messages if m.role == "assistant"]

        assert len(user_messages) == 3
        assert len(asst_messages) == 3

    def test_skip_meta_messages(self, with_tool_calls):
        """Meta messages (tool results) should be skipped."""
        messages = list(parse_conversation(with_tool_calls))

        # Should have 2 user messages (1 skipped due to isMeta) and 2 assistant
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 1  # Only the non-meta user message

    def test_skip_file_history_snapshots(self, file_history_only):
        """File history snapshots should not produce messages."""
        messages = list(parse_conversation(file_history_only))
        assert len(messages) == 0

    def test_empty_file(self, empty_file):
        """Empty file should return no messages."""
        messages = list(parse_conversation(empty_file))
        assert len(messages) == 0

    def test_message_has_required_fields(self, short_conversation):
        """Each message should have all required fields."""
        messages = list(parse_conversation(short_conversation))

        for msg in messages:
            # Check type by name to avoid issues with module reloading
            assert type(msg).__name__ == "Message"
            assert msg.role in ("user", "assistant")
            assert msg.content
            assert msg.uuid
            assert msg.timestamp
            assert msg.session_id

    def test_nonexistent_file_raises(self):
        """Parsing nonexistent file should raise."""
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            list(parse_conversation(Path("/nonexistent/file.jsonl")))

    def test_malformed_json_skipped(self, temp_dir):
        """Malformed JSON lines should be skipped gracefully."""
        filepath = temp_dir / "malformed.jsonl"
        filepath.write_text(
            'not valid json\n'
            '{"type":"user","uuid":"u1","timestamp":"2025-01-15T10:00:00Z",'
            '"message":{"role":"user","content":"Valid line"}}\n'
            '{incomplete\n'
        )

        messages = list(parse_conversation(filepath))
        # Only the valid line should be parsed
        assert len(messages) == 1
        assert messages[0].content == "Valid line"
