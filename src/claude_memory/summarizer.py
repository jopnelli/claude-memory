"""Generate conversation summaries using Ollama."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .chunker import Chunk, load_all_chunks
from .config import CHUNKS_FILE, ensure_dirs
from .parser import get_conversation_files, parse_conversation


# Default model for summarization (qwen2.5:1.5b is fast and capable)
DEFAULT_MODEL = "qwen2.5:1.5b"


def is_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is installed and the model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        # Check if model is in the list
        return model.split(":")[0] in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

# Prompt template for generating conversation summaries
SUMMARY_PROMPT = """Summarize this conversation between a user and an AI assistant in 2-3 sentences. Focus on:
- What the user was trying to accomplish
- Key decisions or solutions discussed
- Any important technical details or outcomes

Conversation:
{conversation}

Summary:"""


def get_conversation_text(filepath: Path, max_turns: int = 50) -> tuple[str, str, int]:
    """Extract conversation text for summarization.

    Returns (text, last_timestamp, turn_count).
    """
    from .parser import parse_conversation

    messages = list(parse_conversation(filepath))

    # Build conversation text
    parts = []
    turn_count = 0
    last_timestamp = ""

    user_msg = None
    for msg in messages:
        if msg.role == "user":
            user_msg = msg
        elif msg.role == "assistant" and user_msg is not None:
            # Truncate very long messages
            user_content = user_msg.content[:1000] + "..." if len(user_msg.content) > 1000 else user_msg.content
            asst_content = msg.content[:1000] + "..." if len(msg.content) > 1000 else msg.content

            parts.append(f"User: {user_content}\n\nAssistant: {asst_content}")
            last_timestamp = msg.timestamp
            turn_count += 1
            user_msg = None

            if turn_count >= max_turns:
                parts.append("... (conversation truncated)")
                break

    return "\n\n---\n\n".join(parts), last_timestamp, turn_count


def generate_summary_ollama(conversation_text: str, model: str = DEFAULT_MODEL) -> str | None:
    """Generate a summary using Ollama.

    Returns None if Ollama is not available or fails.
    """
    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_existing_summary_ids() -> set[str]:
    """Get IDs of existing summary chunks."""
    chunks = load_all_chunks()
    return {c.id for c in chunks if c.chunk_type == "summary"}


def generate_summaries(model: str = DEFAULT_MODEL, force: bool = False) -> Iterator[tuple[str, Chunk | None]]:
    """Generate summary chunks for all conversations.

    Yields (session_id, chunk) tuples. chunk is None if generation failed.
    Skips conversations that already have summaries unless force=True.
    """
    existing_ids = set() if force else get_existing_summary_ids()

    for filepath in get_conversation_files():
        session_id = filepath.stem
        summary_id = f"summary-{session_id}"

        if summary_id in existing_ids:
            continue

        # Get conversation text
        conv_text, last_timestamp, turn_count = get_conversation_text(filepath)

        # Skip very short conversations (< 2 turns)
        if turn_count < 2:
            continue

        # Generate summary
        summary = generate_summary_ollama(conv_text, model)

        if summary:
            chunk = Chunk(
                id=summary_id,
                text=summary,
                timestamp=last_timestamp,
                session_id=session_id,
                chunk_type="summary",
                turn_index=-1,  # -1 indicates this is a summary, not a turn
            )
            yield session_id, chunk
        else:
            yield session_id, None


def sync_summaries(model: str = DEFAULT_MODEL, quiet: bool = False) -> tuple[int, int]:
    """Generate and save summary chunks for all conversations.

    Returns (generated_count, failed_count).
    """
    ensure_dirs()

    generated = 0
    failed = 0

    for session_id, chunk in generate_summaries(model):
        if chunk:
            # Append to chunks file
            with open(CHUNKS_FILE, "a") as f:
                f.write(json.dumps(asdict(chunk)) + "\n")
            generated += 1
            if not quiet:
                print(f"  Generated summary for {session_id[:8]}...")
        else:
            failed += 1
            if not quiet:
                print(f"  Failed to generate summary for {session_id[:8]}...")

    return generated, failed
