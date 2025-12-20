"""Parse Claude Code conversation JSONL files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import PROJECT_DIR


@dataclass
class Message:
    """A single message from a conversation."""

    role: str  # "user" or "assistant"
    content: str
    uuid: str
    timestamp: str
    session_id: str


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


def parse_conversation(filepath: Path) -> Iterator[Message]:
    """Parse a single conversation JSONL file, yielding messages."""
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

            # Extract content
            content = extract_text_content(data)
            if not content or not content.strip():
                continue

            # Get role from message object
            role = data.get("message", {}).get("role")
            if role not in ("user", "assistant"):
                continue

            yield Message(
                role=role,
                content=content.strip(),
                uuid=data.get("uuid", ""),
                timestamp=data.get("timestamp", ""),
                session_id=session_id,
            )


def get_conversation_files() -> list[Path]:
    """Get all JSONL conversation files from the project directory."""
    if not PROJECT_DIR.exists():
        return []

    return sorted(PROJECT_DIR.glob("*.jsonl"))


def parse_all_conversations() -> Iterator[Message]:
    """Parse all conversations from the project directory."""
    for filepath in get_conversation_files():
        yield from parse_conversation(filepath)
