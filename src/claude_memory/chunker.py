"""Chunk conversations into user+assistant exchanges for embedding."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Literal

from .config import CHUNKS_FILE, PROCESSED_FILE, ensure_dirs, get_all_chunk_files
from .parser import Message, get_conversation_files, parse_conversation


# Context window: 1 turn before + 1 turn after for balanced bidirectional context
CONTEXT_BEFORE = 1
CONTEXT_AFTER = 1

# Messages to exclude (Claude's automatic initialization, not useful for search)
EXCLUDED_USER_MESSAGES = {"warmup"}


@dataclass
class Chunk:
    """A single chunk representing a user+assistant exchange or conversation summary."""

    id: str  # UUID of the assistant message, or "summary-{session_id}"
    text: str  # Combined user + assistant text (with context), or summary
    timestamp: str  # Timestamp of last message in chunk
    session_id: str  # Links all chunks from same conversation
    chunk_type: Literal["turn", "summary"] = "turn"  # Type of chunk
    turn_index: int = 0  # Position in conversation (for turns)


def create_chunk_with_context(
    exchanges: list[tuple[Message, Message]],
    current_index: int,
) -> Chunk:
    """Create a chunk from a user+assistant pair with bidirectional context."""
    user_msg, assistant_msg = exchanges[current_index]
    num_exchanges = len(exchanges)

    # Build context from previous exchanges
    before_parts = []
    start = max(0, current_index - CONTEXT_BEFORE)
    for j in range(start, current_index):
        prev_user, prev_asst = exchanges[j]
        before_parts.append(f"User: {prev_user.content}\n\nAssistant: {prev_asst.content}")

    # Current exchange
    current = f"User: {user_msg.content}\n\nAssistant: {assistant_msg.content}"

    # Build context from following exchanges
    after_parts = []
    end = min(num_exchanges, current_index + CONTEXT_AFTER + 1)
    for j in range(current_index + 1, end):
        next_user, next_asst = exchanges[j]
        after_parts.append(f"User: {next_user.content}\n\nAssistant: {next_asst.content}")

    # Combine: [before] --- [current] --- [after]
    all_parts = before_parts + [current] + after_parts
    text = "\n\n---\n\n".join(all_parts)

    return Chunk(
        id=assistant_msg.uuid,
        text=text,
        timestamp=assistant_msg.timestamp,
        session_id=assistant_msg.session_id,
        chunk_type="turn",
        turn_index=current_index,
    )


def is_excluded_message(user_content: str) -> bool:
    """Check if a user message should be excluded from indexing."""
    normalized = user_content.strip().lower()
    return normalized in EXCLUDED_USER_MESSAGES


def chunk_conversation(filepath: Path) -> Iterator[Chunk]:
    """Chunk a conversation into user+assistant exchanges with context."""
    messages = list(parse_conversation(filepath))

    # First pass: collect all exchanges (excluding warmup/init messages)
    exchanges: list[tuple[Message, Message]] = []
    user_msg = None
    for msg in messages:
        if msg.role == "user":
            user_msg = msg
        elif msg.role == "assistant" and user_msg is not None:
            # Skip excluded messages like "Warmup"
            if not is_excluded_message(user_msg.content):
                exchanges.append((user_msg, msg))
            user_msg = None

    # Second pass: create chunks with context
    for i in range(len(exchanges)):
        yield create_chunk_with_context(exchanges, i)


def load_processed() -> dict[str, str]:
    """Load the set of processed conversation files with their mtimes."""
    if not PROCESSED_FILE.exists():
        return {}

    try:
        with open(PROCESSED_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_processed(processed: dict[str, str]) -> None:
    """Save the set of processed conversation files."""
    ensure_dirs()
    with open(PROCESSED_FILE, "w") as f:
        json.dump(processed, f, indent=2)


def load_existing_chunk_ids() -> set[str]:
    """Load IDs of chunks already in any chunks file (all machines)."""
    ids = set()
    for chunk_file in get_all_chunk_files():
        with open(chunk_file, "r") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    ids.add(chunk.get("id", ""))
                except json.JSONDecodeError:
                    continue
    return ids


def sync_chunks() -> tuple[int, int]:
    """
    Sync new conversation chunks to chunks.jsonl.

    Returns (new_chunks, new_files) counts.
    """
    ensure_dirs()

    processed = load_processed()
    existing_ids = load_existing_chunk_ids()

    new_chunks = 0
    new_files = 0

    conversation_files = get_conversation_files()

    for filepath in conversation_files:
        # Check if file has been modified since last processing
        mtime = str(filepath.stat().st_mtime)
        file_key = filepath.name

        if file_key in processed and processed[file_key] == mtime:
            continue

        # Process this conversation
        file_had_new = False
        for chunk in chunk_conversation(filepath):
            if chunk.id not in existing_ids:
                # Append to chunks file
                with open(CHUNKS_FILE, "a") as f:
                    f.write(json.dumps(asdict(chunk)) + "\n")
                existing_ids.add(chunk.id)
                new_chunks += 1
                file_had_new = True

        if file_had_new:
            new_files += 1

        # Mark as processed
        processed[file_key] = mtime

    save_processed(processed)
    return new_chunks, new_files


def load_all_chunks() -> list[Chunk]:
    """Load all chunks from all chunk files (all machines), deduplicating by ID.

    Reads from both legacy chunks.jsonl and machine-specific chunks-*.jsonl files.
    If the same chunk ID appears multiple times (e.g., from a git merge),
    the last occurrence is kept. This makes the system robust to duplicate
    entries from multi-machine sync conflicts.

    Handles both old format (without chunk_type/turn_index) and new format.
    """
    chunk_files = get_all_chunk_files()
    if not chunk_files:
        return []

    # Use dict to deduplicate by ID, keeping last occurrence
    chunks_by_id: dict[str, Chunk] = {}
    for chunk_file in chunk_files:
        with open(chunk_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    chunk = Chunk(
                        id=data["id"],
                        text=data["text"],
                        timestamp=data["timestamp"],
                        session_id=data["session_id"],
                        # New fields with defaults for backwards compatibility
                        chunk_type=data.get("chunk_type", "turn"),
                        turn_index=data.get("turn_index", 0),
                    )
                    chunks_by_id[chunk.id] = chunk
                except (json.JSONDecodeError, KeyError):
                    continue
    return list(chunks_by_id.values())
