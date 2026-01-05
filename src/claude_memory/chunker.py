"""Chunk conversations into user+assistant exchanges for embedding."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Literal

from .config import CHUNKS_FILE, PROCESSED_FILE, ensure_dirs, get_all_chunk_files
from .parser import Message, get_conversation_files, parse_conversation


# Token limits for embedding model (all-mpnet-base-v2 has 384 token limit)
MAX_TOKENS = 384
TARGET_TOKENS = 300  # Leave headroom for safety
CONTEXT_TOKEN_BUDGET = 150  # Budget for context (before + after)

# Messages to exclude (Claude's automatic initialization, not useful for search)
EXCLUDED_USER_MESSAGES = {"warmup"}


def estimate_tokens(text: str) -> int:
    """Estimate token count using word-based heuristic.

    Most tokenizers produce ~1.3 tokens per word on average for English text.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def split_at_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks that fit within token limit.

    Splits at sentence boundaries when possible.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = split_at_sentences(text)
    if not sentences:
        # Fallback: split by words if no sentences found
        words = text.split()
        chunks = []
        current = []
        for word in words:
            current.append(word)
            if estimate_tokens(" ".join(current)) > max_tokens:
                if len(current) > 1:
                    current.pop()
                    chunks.append(" ".join(current))
                    current = [word]
                else:
                    # Single word exceeds limit, include it anyway
                    chunks.append(word)
                    current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    # Build chunks from sentences
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if sentence_tokens > max_tokens:
            # Sentence itself is too long, split it by words
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Recursively split the long sentence
            sub_chunks = split_long_text(sentence, max_tokens)
            chunks.extend(sub_chunks[:-1])
            if sub_chunks:
                current_chunk = [sub_chunks[-1]]
                current_tokens = estimate_tokens(sub_chunks[-1])
        elif current_tokens + sentence_tokens > max_tokens:
            # Adding this sentence would exceed limit
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@dataclass
class Chunk:
    """A single chunk representing a user+assistant exchange or conversation summary."""

    id: str  # UUID of the assistant message, or "summary-{session_id}"
    text: str  # Combined user + assistant text (with context), or summary
    timestamp: str  # Timestamp of last message in chunk
    session_id: str  # Links all chunks from same conversation
    chunk_type: Literal["turn", "summary"] = "turn"  # Type of chunk
    turn_index: int = 0  # Position in conversation (for turns)


def format_exchange(user_msg: Message, assistant_msg: Message) -> str:
    """Format a user+assistant exchange as text."""
    return f"User: {user_msg.content}\n\nAssistant: {assistant_msg.content}"


def calculate_adaptive_context(
    exchanges: list[tuple[Message, Message]],
    current_index: int,
    current_tokens: int,
    token_budget: int = CONTEXT_TOKEN_BUDGET,
) -> tuple[list[str], list[str]]:
    """Calculate how much context fits within the token budget.

    Prioritizes recent context (before) over future context (after).
    Uses a 70/30 split of the budget between before and after context.

    Returns (before_parts, after_parts).
    """
    before_budget = int(token_budget * 0.7)
    after_budget = token_budget - before_budget

    before_parts = []
    before_tokens = 0

    # Add context from most recent to oldest (greedy)
    for j in range(current_index - 1, -1, -1):
        prev_user, prev_asst = exchanges[j]
        part = format_exchange(prev_user, prev_asst)
        part_tokens = estimate_tokens(part)

        if before_tokens + part_tokens > before_budget:
            break
        before_parts.insert(0, part)  # Insert at beginning to maintain order
        before_tokens += part_tokens

    after_parts = []
    after_tokens = 0

    # Add context from nearest to furthest
    num_exchanges = len(exchanges)
    for j in range(current_index + 1, num_exchanges):
        next_user, next_asst = exchanges[j]
        part = format_exchange(next_user, next_asst)
        part_tokens = estimate_tokens(part)

        if after_tokens + part_tokens > after_budget:
            break
        after_parts.append(part)
        after_tokens += part_tokens

    return before_parts, after_parts


def create_chunk_with_context(
    exchanges: list[tuple[Message, Message]],
    current_index: int,
) -> Iterator[Chunk]:
    """Create chunk(s) from a user+assistant pair with adaptive context.

    Yields one or more chunks. If the current exchange exceeds the token limit,
    it will be split at sentence boundaries into multiple chunks.
    """
    user_msg, assistant_msg = exchanges[current_index]

    # Format the current exchange
    current = format_exchange(user_msg, assistant_msg)
    current_tokens = estimate_tokens(current)

    # Check if current exchange itself needs splitting
    if current_tokens > TARGET_TOKENS:
        # Split the assistant's response at sentence boundaries
        user_prefix = f"User: {user_msg.content}\n\nAssistant: "
        prefix_tokens = estimate_tokens(user_prefix)
        available_tokens = TARGET_TOKENS - prefix_tokens

        if available_tokens <= 0:
            # User message alone is too long, truncate it
            user_content_tokens = TARGET_TOKENS - estimate_tokens("User: \n\nAssistant: ")
            # Rough truncation by words
            words = user_msg.content.split()
            truncated_words = []
            for word in words:
                truncated_words.append(word)
                if estimate_tokens(" ".join(truncated_words)) > user_content_tokens:
                    truncated_words.pop()
                    break
            user_prefix = f"User: {' '.join(truncated_words)}...\n\nAssistant: "
            prefix_tokens = estimate_tokens(user_prefix)
            available_tokens = TARGET_TOKENS - prefix_tokens

        # Split assistant response
        assistant_parts = split_long_text(assistant_msg.content, max(available_tokens, 50))

        for part_idx, part in enumerate(assistant_parts):
            chunk_text = user_prefix + part
            chunk_id = (
                assistant_msg.uuid if len(assistant_parts) == 1
                else f"{assistant_msg.uuid}-part-{part_idx}"
            )

            yield Chunk(
                id=chunk_id,
                text=chunk_text,
                timestamp=assistant_msg.timestamp,
                session_id=assistant_msg.session_id,
                chunk_type="turn",
                turn_index=current_index,
            )
        return

    # Current exchange fits, calculate adaptive context
    remaining_budget = TARGET_TOKENS - current_tokens
    before_parts, after_parts = calculate_adaptive_context(
        exchanges, current_index, current_tokens, remaining_budget
    )

    # Combine: [before] --- [current] --- [after]
    all_parts = before_parts + [current] + after_parts
    text = "\n\n---\n\n".join(all_parts)

    yield Chunk(
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
    """Chunk a conversation into user+assistant exchanges with adaptive context.

    Handles long exchanges by splitting them into multiple token-limited chunks.
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

    # Second pass: create chunks with adaptive context
    for i in range(len(exchanges)):
        yield from create_chunk_with_context(exchanges, i)


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
