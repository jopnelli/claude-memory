"""Chunk conversations into user+assistant exchanges for embedding."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from .config import CHUNKS_FILE, PROCESSED_FILE, ensure_dirs
from .parser import Message, get_conversation_files, parse_conversation


@dataclass
class Chunk:
    """A single chunk representing a user+assistant exchange."""

    id: str  # UUID of the assistant message
    text: str  # Combined user + assistant text
    timestamp: str
    session_id: str


def create_chunk(user_msg: Message, assistant_msg: Message) -> Chunk:
    """Create a chunk from a user+assistant pair."""
    text = f"User: {user_msg.content}\n\nAssistant: {assistant_msg.content}"
    return Chunk(
        id=assistant_msg.uuid,
        text=text,
        timestamp=assistant_msg.timestamp,
        session_id=assistant_msg.session_id,
    )


def chunk_conversation(filepath: Path) -> Iterator[Chunk]:
    """Chunk a conversation into user+assistant exchanges."""
    messages = list(parse_conversation(filepath))

    user_msg = None
    for msg in messages:
        if msg.role == "user":
            user_msg = msg
        elif msg.role == "assistant" and user_msg is not None:
            yield create_chunk(user_msg, msg)
            user_msg = None


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
    """Load IDs of chunks already in chunks.jsonl."""
    if not CHUNKS_FILE.exists():
        return set()

    ids = set()
    with open(CHUNKS_FILE, "r") as f:
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
    """Load all chunks from chunks.jsonl, deduplicating by ID.

    If the same chunk ID appears multiple times (e.g., from a git merge),
    the last occurrence is kept. This makes the system robust to duplicate
    entries from multi-machine sync conflicts.
    """
    if not CHUNKS_FILE.exists():
        return []

    # Use dict to deduplicate by ID, keeping last occurrence
    chunks_by_id: dict[str, Chunk] = {}
    with open(CHUNKS_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                chunk = Chunk(
                    id=data["id"],
                    text=data["text"],
                    timestamp=data["timestamp"],
                    session_id=data["session_id"],
                )
                chunks_by_id[chunk.id] = chunk
            except (json.JSONDecodeError, KeyError):
                continue
    return list(chunks_by_id.values())
