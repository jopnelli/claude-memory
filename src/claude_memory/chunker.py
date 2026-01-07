"""Chunk conversations into user+assistant exchanges for embedding."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Literal

from .config import CHUNKS_FILE, PROCESSED_FILE, ensure_dirs, get_all_chunk_files
from .parser import (
    Message,
    get_conversation_files,
    parse_conversation,
    extract_files_from_tool_calls,
    extract_commands_from_tool_calls,
)


# Context window: 1 turn before + 1 turn after for balanced bidirectional context
CONTEXT_BEFORE = 1
CONTEXT_AFTER = 1

# Chunk size limits for embedding model compatibility
# all-mpnet-base-v2 has a 384 token limit (~1,500 chars)
# We use 1,400 chars as a safe margin (~310 tokens)
MAX_CHUNK_CHARS = 1400
OVERLAP_CHARS = 280  # 20% overlap for context continuity

# Separators for recursive splitting, in order of preference
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]

# Messages to exclude (Claude's automatic initialization, not useful for search)
EXCLUDED_USER_MESSAGES = {"warmup"}


def recursive_split(text: str, separators: list[str] | None = None) -> list[str]:
    """Split text recursively at natural boundaries to fit within MAX_CHUNK_CHARS.

    Tries separators in order of preference (paragraphs -> lines -> sentences -> words).
    Falls back to hard character split if no separator works.
    """
    if separators is None:
        separators = SEPARATORS

    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    # Try each separator in order
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= MAX_CHUNK_CHARS:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If single part exceeds limit, try finer separator
                    if len(part) > MAX_CHUNK_CHARS:
                        remaining_seps = separators[separators.index(sep) + 1 :]
                        if remaining_seps:
                            chunks.extend(recursive_split(part, remaining_seps))
                        else:
                            # Hard split as last resort
                            chunks.extend(
                                part[i : i + MAX_CHUNK_CHARS]
                                for i in range(
                                    0, len(part), MAX_CHUNK_CHARS - OVERLAP_CHARS
                                )
                            )
                    else:
                        current = part

            if current:
                chunks.append(current)

            return chunks if chunks else [text]

    # Last resort: hard split by characters
    return [
        text[i : i + MAX_CHUNK_CHARS]
        for i in range(0, len(text), MAX_CHUNK_CHARS - OVERLAP_CHARS)
    ]


def add_overlap(chunks: list[str], overlap: int = OVERLAP_CHARS) -> list[str]:
    """Add overlap between chunks for context continuity.

    Prepends the end of the previous chunk to each subsequent chunk,
    helping preserve context across chunk boundaries.
    """
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        # Prepend end of previous chunk
        prev_overlap = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
        result.append(prev_overlap + " " + chunks[i])

    return result


@dataclass
class Chunk:
    """A single chunk representing a user+assistant exchange or conversation summary."""

    id: str  # UUID of the assistant message, or "summary-{session_id}", or "{uuid}-{index}" for splits
    text: str  # Combined user + assistant text (with context), or summary
    timestamp: str  # Timestamp of last message in chunk
    session_id: str  # Links all chunks from same conversation
    chunk_type: Literal["turn", "summary"] = "turn"  # Type of chunk
    turn_index: int = 0  # Position in conversation (for turns)
    # Fields for tracking split chunks (when a turn exceeds MAX_CHUNK_CHARS)
    parent_turn_id: str = ""  # Original turn UUID (empty if not split)
    chunk_index: int = 0  # Position within split (0, 1, 2...)
    total_chunks: int = 1  # How many chunks this turn produced
    # Tool metadata - enables filtering/search by tool usage
    tools_used: str = ""  # Comma-separated tool names (e.g., "Read,Bash,Edit")
    files_touched: str = ""  # Comma-separated file paths
    commands_run: str = ""  # Comma-separated commands (truncated)


def create_chunks_with_context(
    exchanges: list[tuple[Message, Message]],
    current_index: int,
) -> list[Chunk]:
    """Create chunk(s) from a user+assistant pair with bidirectional context.

    If the combined text exceeds MAX_CHUNK_CHARS, splits into multiple chunks
    while preserving natural boundaries and adding overlap for context continuity.

    Returns a list of Chunk objects (usually 1, but may be more for long exchanges).
    """
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

    base_id = assistant_msg.uuid

    # Extract tool metadata from the current turn's assistant message
    tool_calls = assistant_msg.tool_calls
    tools_used = ",".join(sorted(set(tc.name for tc in tool_calls))) if tool_calls else ""
    files_touched = ",".join(sorted(extract_files_from_tool_calls(tool_calls))) if tool_calls else ""
    commands = extract_commands_from_tool_calls(tool_calls) if tool_calls else []
    commands_run = ",".join(commands[:5])  # Limit to 5 commands to avoid bloat

    # If text fits in one chunk, return single chunk
    if len(text) <= MAX_CHUNK_CHARS:
        return [
            Chunk(
                id=base_id,
                text=text,
                timestamp=assistant_msg.timestamp,
                session_id=assistant_msg.session_id,
                chunk_type="turn",
                turn_index=current_index,
                parent_turn_id="",
                chunk_index=0,
                total_chunks=1,
                tools_used=tools_used,
                files_touched=files_touched,
                commands_run=commands_run,
            )
        ]

    # Split and create multiple chunks
    parts = add_overlap(recursive_split(text))
    return [
        Chunk(
            id=f"{base_id}-{i}" if len(parts) > 1 else base_id,
            text=part,
            timestamp=assistant_msg.timestamp,
            session_id=assistant_msg.session_id,
            chunk_type="turn",
            turn_index=current_index,
            parent_turn_id=base_id,  # Track original turn
            chunk_index=i,
            total_chunks=len(parts),
            tools_used=tools_used,
            files_touched=files_touched,
            commands_run=commands_run,
        )
        for i, part in enumerate(parts)
    ]


# Backwards compatibility alias
def create_chunk_with_context(
    exchanges: list[tuple[Message, Message]],
    current_index: int,
) -> Chunk:
    """Create a single chunk (legacy interface). Use create_chunks_with_context instead."""
    chunks = create_chunks_with_context(exchanges, current_index)
    return chunks[0]  # Return first chunk for backwards compatibility


def create_chunk(user_msg: Message, assistant_msg: Message) -> Chunk:
    """Create a simple chunk from a user+assistant pair without context.

    This is a simpler interface for creating chunks without surrounding context.
    Useful for tests and simple use cases.
    """
    text = f"User: {user_msg.content}\n\nAssistant: {assistant_msg.content}"
    return Chunk(
        id=assistant_msg.uuid,
        text=text,
        timestamp=assistant_msg.timestamp,
        session_id=assistant_msg.session_id,
        chunk_type="turn",
        turn_index=0,
    )


def is_excluded_message(user_content: str) -> bool:
    """Check if a user message should be excluded from indexing."""
    normalized = user_content.strip().lower()
    return normalized in EXCLUDED_USER_MESSAGES


def chunk_conversation(filepath: Path) -> Iterator[Chunk]:
    """Chunk a conversation into user+assistant exchanges with context.

    Long exchanges that exceed MAX_CHUNK_CHARS are automatically split into
    multiple chunks with overlap for context continuity.
    """
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

    # Second pass: create chunks with context (may produce multiple chunks per exchange)
    for i in range(len(exchanges)):
        yield from create_chunks_with_context(exchanges, i)


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

    Handles both old format (without chunk_type/turn_index/split fields) and new format.
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
                        # Fields with defaults for backwards compatibility
                        chunk_type=data.get("chunk_type", "turn"),
                        turn_index=data.get("turn_index", 0),
                        # Split-tracking fields
                        parent_turn_id=data.get("parent_turn_id", ""),
                        chunk_index=data.get("chunk_index", 0),
                        total_chunks=data.get("total_chunks", 1),
                        # Tool metadata fields (new)
                        tools_used=data.get("tools_used", ""),
                        files_touched=data.get("files_touched", ""),
                        commands_run=data.get("commands_run", ""),
                    )
                    chunks_by_id[chunk.id] = chunk
                except (json.JSONDecodeError, KeyError):
                    continue
    return list(chunks_by_id.values())
