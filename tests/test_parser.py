"""Tests for the parser module."""

import pytest
from claude_memory.parser import (
    Message,
    ToolCall,
    extract_text_content,
    extract_tool_calls,
    extract_tool_results,
    extract_files_from_tool_calls,
    extract_commands_from_tool_calls,
    parse_conversation,
    find_conversation_file,
    get_context_around,
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

    def test_parse_long_conversation(self, long_conversation):
        """Parse a longer conversation with multiple exchanges."""
        messages = list(parse_conversation(long_conversation))

        # 6 user + 6 assistant = 12 messages
        assert len(messages) == 12

        user_messages = [m for m in messages if m.role == "user"]
        asst_messages = [m for m in messages if m.role == "assistant"]

        assert len(user_messages) == 6
        assert len(asst_messages) == 6

        # Check content is substantial (not truncated)
        # The assistant messages have long technical content
        total_content_length = sum(len(m.content) for m in asst_messages)
        assert total_content_length > 3000  # Should have substantial content

    def test_long_conversation_preserves_formatting(self, long_conversation):
        """Long messages with code blocks should preserve formatting."""
        messages = list(parse_conversation(long_conversation))

        # Find a message with code
        code_messages = [m for m in messages if "```" in m.content]
        assert len(code_messages) > 0  # Should have code blocks

        # Verify code blocks are intact
        for msg in code_messages:
            # Every opening ``` should have a closing ```
            assert msg.content.count("```") % 2 == 0


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""

    def test_extract_single_tool_call(self):
        """Extract a single tool call from assistant message."""
        data = {
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/foo.py"}},
                    {"type": "text", "text": "Reading file."},
                ]
            }
        }
        tool_calls = extract_tool_calls(data)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "Read"
        assert tool_calls[0].input == {"file_path": "/foo.py"}
        assert tool_calls[0].id == "t1"

    def test_extract_multiple_tool_calls(self):
        """Extract multiple tool calls from a single message."""
        data = {
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a.py"}},
                    {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls -la"}},
                    {"type": "tool_use", "id": "t3", "name": "Edit", "input": {"file_path": "/b.py"}},
                    {"type": "text", "text": "Done."},
                ]
            }
        }
        tool_calls = extract_tool_calls(data)
        assert len(tool_calls) == 3
        assert [tc.name for tc in tool_calls] == ["Read", "Bash", "Edit"]

    def test_no_tool_calls(self):
        """Message with only text has no tool calls."""
        data = {"message": {"content": [{"type": "text", "text": "Just text."}]}}
        assert extract_tool_calls(data) == []

    def test_string_content_no_tool_calls(self):
        """User message with string content has no tool calls."""
        data = {"message": {"content": "Hello"}}
        assert extract_tool_calls(data) == []


class TestExtractToolResults:
    """Tests for extract_tool_results function."""

    def test_extract_tool_result(self):
        """Extract tool result from user message."""
        data = {
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "File contents here"}
                ]
            }
        }
        results = extract_tool_results(data)
        assert len(results) == 1
        assert results[0].tool_use_id == "t1"
        assert results[0].content == "File contents here"
        assert results[0].is_error is False

    def test_extract_error_result(self):
        """Extract error tool result."""
        data = {
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Error!", "is_error": True}
                ]
            }
        }
        results = extract_tool_results(data)
        assert len(results) == 1
        assert results[0].is_error is True

    def test_extract_nested_content_result(self):
        """Extract tool result with nested content blocks."""
        data = {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "First part"},
                            {"type": "text", "text": "Second part"},
                        ]
                    }
                ]
            }
        }
        results = extract_tool_results(data)
        assert len(results) == 1
        assert results[0].content == "First part\nSecond part"


class TestExtractFilesFromToolCalls:
    """Tests for extract_files_from_tool_calls function."""

    def test_extract_file_path(self):
        """Extract file_path from Read/Write/Edit tools."""
        tool_calls = [
            ToolCall(name="Read", input={"file_path": "/src/main.py"}, id="t1"),
            ToolCall(name="Edit", input={"file_path": "/src/utils.py"}, id="t2"),
        ]
        files = extract_files_from_tool_calls(tool_calls)
        assert files == {"/src/main.py", "/src/utils.py"}

    def test_extract_path_from_grep(self):
        """Extract path from Grep tool."""
        tool_calls = [
            ToolCall(name="Grep", input={"pattern": "TODO", "path": "/project"}, id="t1"),
        ]
        files = extract_files_from_tool_calls(tool_calls)
        assert "/project" in files

    def test_extract_quoted_paths_from_bash(self):
        """Extract quoted file paths from Bash commands."""
        tool_calls = [
            ToolCall(name="Bash", input={"command": 'cat "/path/to/file.txt"'}, id="t1"),
            ToolCall(name="Bash", input={"command": "python 'script.py'"}, id="t2"),
        ]
        files = extract_files_from_tool_calls(tool_calls)
        assert "/path/to/file.txt" in files
        assert "script.py" in files

    def test_empty_tool_calls(self):
        """Empty tool calls list returns empty set."""
        assert extract_files_from_tool_calls([]) == set()


class TestExtractCommandsFromToolCalls:
    """Tests for extract_commands_from_tool_calls function."""

    def test_extract_bash_commands(self):
        """Extract commands from Bash tool calls."""
        tool_calls = [
            ToolCall(name="Bash", input={"command": "ls -la"}, id="t1"),
            ToolCall(name="Bash", input={"command": "git status"}, id="t2"),
        ]
        commands = extract_commands_from_tool_calls(tool_calls)
        assert commands == ["ls -la", "git status"]

    def test_truncate_long_commands(self):
        """Long commands are truncated."""
        long_cmd = "x" * 300  # Exceeds 200 char limit
        tool_calls = [ToolCall(name="Bash", input={"command": long_cmd}, id="t1")]
        commands = extract_commands_from_tool_calls(tool_calls)
        assert len(commands[0]) == 203  # 200 + "..."
        assert commands[0].endswith("...")

    def test_skip_non_bash(self):
        """Non-Bash tools are skipped."""
        tool_calls = [
            ToolCall(name="Read", input={"file_path": "/foo.py"}, id="t1"),
            ToolCall(name="Bash", input={"command": "echo hi"}, id="t2"),
        ]
        commands = extract_commands_from_tool_calls(tool_calls)
        assert commands == ["echo hi"]


class TestParseConversationWithTools:
    """Tests for parsing conversations with tool metadata."""

    def test_parse_extracts_tool_calls(self, with_rich_tool_calls):
        """Parsing should extract tool calls from assistant messages."""
        messages = list(parse_conversation(with_rich_tool_calls))
        assistant_msgs = [m for m in messages if m.role == "assistant"]

        # First assistant message has Read and Bash tool calls
        first_asst = assistant_msgs[0]
        assert len(first_asst.tool_calls) == 2
        tool_names = [tc.name for tc in first_asst.tool_calls]
        assert "Read" in tool_names
        assert "Bash" in tool_names

    def test_tool_calls_have_correct_input(self, with_rich_tool_calls):
        """Tool calls should have correct input data."""
        messages = list(parse_conversation(with_rich_tool_calls))
        assistant_msgs = [m for m in messages if m.role == "assistant"]

        first_asst = assistant_msgs[0]
        read_call = next(tc for tc in first_asst.tool_calls if tc.name == "Read")
        assert read_call.input["file_path"] == "/project/config.json"


class TestFindConversationFile:
    """Tests for find_conversation_file function."""

    def test_find_nonexistent_session(self):
        """Nonexistent session returns None."""
        result = find_conversation_file("nonexistent-session-id-12345")
        assert result is None


class TestGetContextAround:
    """Tests for get_context_around function."""

    def test_context_for_nonexistent_session(self):
        """Nonexistent session returns empty list."""
        result = get_context_around("nonexistent-session", "2025-01-01T00:00:00Z", n=2)
        assert result == []
