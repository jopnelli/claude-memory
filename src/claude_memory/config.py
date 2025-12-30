"""Configuration for claude-memory."""

import os
import socket
from pathlib import Path

# Claude Code projects directory
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def get_project_dirs() -> list[Path]:
    """Get the Claude project directories to index.

    Set CLAUDE_MEMORY_PROJECT to override:
    - Specific path: Index only that directory
    - Not set: Index all project directories (default)
    """
    env_path = os.environ.get("CLAUDE_MEMORY_PROJECT", "")

    if env_path:
        return [Path(env_path).expanduser()]

    # Default: index all project directories
    if CLAUDE_PROJECTS.exists():
        return sorted([d for d in CLAUDE_PROJECTS.iterdir() if d.is_dir()])
    return []


def get_project_dir() -> Path:
    """Get the first Claude project directory (for backwards compatibility)."""
    dirs = get_project_dirs()
    if dirs:
        return dirs[0]
    raise ValueError("No project directories found.")


def get_storage_dir() -> Path:
    """Get the storage directory for chunks and embeddings.

    Set CLAUDE_MEMORY_STORAGE to specify. This should be a git-synced directory
    if you want to share memory across machines.
    """
    if env_path := os.environ.get("CLAUDE_MEMORY_STORAGE"):
        return Path(env_path).expanduser()

    # Default to ~/.claude-memory
    return Path.home() / ".claude-memory"


def get_collection_name() -> str:
    """Get the ChromaDB collection name.

    Set CLAUDE_MEMORY_COLLECTION to override.
    """
    return os.environ.get("CLAUDE_MEMORY_COLLECTION", "conversations")


def get_machine_id() -> str:
    """Get the machine identifier for this host.

    Set CLAUDE_MEMORY_MACHINE_ID to override (useful for distinguishing
    multiple environments on the same host, e.g., different VMs).

    Returns a sanitized hostname by default.
    """
    if env_id := os.environ.get("CLAUDE_MEMORY_MACHINE_ID"):
        return env_id

    # Use hostname, sanitized for filename safety
    hostname = socket.gethostname()
    # Remove domain suffix and sanitize
    short_name = hostname.split(".")[0].lower()
    # Replace any unsafe characters
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in short_name)


def get_chunks_file() -> Path:
    """Get the chunks file path for this machine."""
    machine_id = get_machine_id()
    return get_storage_dir() / f"chunks-{machine_id}.jsonl"


def get_all_chunk_files() -> list[Path]:
    """Get all chunk files from all machines.

    Returns all files matching chunks-*.jsonl in the storage directory,
    plus the legacy chunks.jsonl if it exists.
    """
    storage = get_storage_dir()
    files = []

    # Include legacy chunks.jsonl if it exists
    legacy = storage / "chunks.jsonl"
    if legacy.exists():
        files.append(legacy)

    # Include all machine-specific chunk files
    files.extend(sorted(storage.glob("chunks-*.jsonl")))

    return files


# Resolved paths
PROJECT_DIR = get_project_dir()
STORAGE_DIR = get_storage_dir()
CHUNKS_FILE = get_chunks_file()
CHROMA_DIR = STORAGE_DIR / "chroma"
PROCESSED_FILE = STORAGE_DIR / "processed.json"
COLLECTION_NAME = get_collection_name()

# Embedding model (all-mpnet-base-v2 is recommended for quality)
EMBEDDING_MODEL = os.environ.get("CLAUDE_MEMORY_MODEL", "all-mpnet-base-v2")


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
