"""Parse Claude Code conversation JSONL files."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import get_project_dirs, PROJECT_DIR


@dataclass
class ToolCall:
    """A tool call from an assistant message."""

    name: str  # Tool name: Bash, Read, Edit, Write, Grep, Glob, etc.
    input: dict  # Tool input parameters
    id: str = ""  # Tool use ID for matching with results


@dataclass
class ToolResult:
    """A tool result from a user message."""

    tool_use_id: str
    content: str  # The result content (may be truncated for storage)
    is_error: bool = False


@dataclass
class Message:
    """A single message from a conversation."""

    role: str  # "user" or "assistant"
    content: str  # Text content only
    uuid: str
    timestamp: str
    session_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


def extract_text_content(message_data: dict) -> str | None:
    """Extract text content from a message, handling both user and assistant formats."""
    message = message_data.get("message", {})
    content = message.get("content")

    if content is None:
        return None

    # User messages: content is a string
    if isinstance(content, str):
        return content

    # Assistant messages: content is an array of blocks
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts) if text_parts else None

    return None


def extract_tool_calls(message_data: dict) -> list[ToolCall]:
    """Extract tool calls from an assistant message."""
    message = message_data.get("message", {})
    content = message.get("content")

    if not isinstance(content, list):
        return []

    tool_calls = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                    id=block.get("id", ""),
                )
            )
    return tool_calls


def extract_tool_results(message_data: dict) -> list[ToolResult]:
    """Extract tool results from a user message (tool_result blocks)."""
    message = message_data.get("message", {})
    content = message.get("content")

    if not isinstance(content, list):
        return []

    results = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            # Content can be string or list of content blocks
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                # Extract text from content blocks
                text_parts = []
                for part in result_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                result_content = "\n".join(text_parts)

            results.append(
                ToolResult(
                    tool_use_id=block.get("tool_use_id", ""),
                    content=str(result_content) if result_content else "",
                    is_error=block.get("is_error", False),
                )
            )
    return results


def extract_files_from_tool_calls(tool_calls: list[ToolCall]) -> set[str]:
    """Extract file paths from tool calls."""
    files = set()
    for tc in tool_calls:
        # Read, Write, Edit, Glob all have file_path
        if "file_path" in tc.input:
            files.add(tc.input["file_path"])
        # Grep and some others have path
        if "path" in tc.input:
            files.add(tc.input["path"])
        # Bash commands might reference files - extract common patterns
        if tc.name == "Bash" and "command" in tc.input:
            cmd = tc.input["command"]
            # Simple heuristic: find quoted paths or common file extensions
            # This is imperfect but catches many cases
            path_patterns = re.findall(r'["\']([^"\']+\.[a-z]{1,4})["\']', cmd)
            files.update(path_patterns)
    return files


def extract_commands_from_tool_calls(tool_calls: list[ToolCall]) -> list[str]:
    """Extract command strings from Bash tool calls."""
    commands = []
    for tc in tool_calls:
        if tc.name == "Bash" and "command" in tc.input:
            # Truncate long commands
            cmd = tc.input["command"]
            if len(cmd) > 200:
                cmd = cmd[:200] + "..."
            commands.append(cmd)
    return commands


def parse_conversation(filepath: Path, include_tool_only: bool = False) -> Iterator[Message]:
    """Parse a single conversation JSONL file, yielding messages.

    Args:
        filepath: Path to the .jsonl conversation file
        include_tool_only: If True, include messages that only have tool calls/results
                          (no text content). Default False for backwards compatibility.
    """
    session_id = filepath.stem

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip non-message types
            msg_type = data.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            # Skip meta messages
            if data.get("isMeta"):
                continue

            # Get role from message object
            role = data.get("message", {}).get("role")
            if role not in ("user", "assistant"):
                continue

            # Extract content and tools
            content = extract_text_content(data)
            tool_calls = extract_tool_calls(data) if role == "assistant" else []
            tool_results = extract_tool_results(data) if role == "user" else []

            # Skip if no content and no tools (unless include_tool_only)
            has_content = content and content.strip()
            has_tools = tool_calls or tool_results

            if not has_content and not has_tools:
                continue
            if not has_content and not include_tool_only:
                continue

            yield Message(
                role=role,
                content=content.strip() if content else "",
                uuid=data.get("uuid", ""),
                timestamp=data.get("timestamp", ""),
                session_id=session_id,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )


def get_conversation_files() -> list[Path]:
    """Get all JSONL conversation files from all project directories."""
    files = []
    for project_dir in get_project_dirs():
        if project_dir.exists():
            files.extend(project_dir.glob("*.jsonl"))
    return sorted(files)


def parse_all_conversations() -> Iterator[Message]:
    """Parse all conversations from the project directory."""
    for filepath in get_conversation_files():
        yield from parse_conversation(filepath)


def find_conversation_file(session_id: str) -> Path | None:
    """Find the .jsonl file for a given session ID."""
    for filepath in get_conversation_files():
        if filepath.stem == session_id:
            return filepath
    return None


def get_context_around(session_id: str, timestamp: str, n: int = 2) -> list[Message]:
    """Get N messages before and after a specific timestamp in a conversation.

    Args:
        session_id: The session ID (filename stem) of the conversation
        timestamp: The timestamp of the target message
        n: Number of messages to include before and after

    Returns:
        List of messages around the target, including tool calls
    """
    filepath = find_conversation_file(session_id)
    if not filepath:
        return []

    messages = list(parse_conversation(filepath, include_tool_only=False))
    if not messages:
        return []

    # Find the message matching the timestamp
    target_idx = None
    for i, msg in enumerate(messages):
        if msg.timestamp == timestamp:
            target_idx = i
            break

    if target_idx is None:
        # Fallback: return first few messages if timestamp not found
        return messages[:n * 2 + 1]

    # Calculate window bounds
    start = max(0, target_idx - n)
    end = min(len(messages), target_idx + n + 1)

    return messages[start:end]
