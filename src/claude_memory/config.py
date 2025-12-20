"""Configuration for claude-memory."""

import os
from pathlib import Path

# Claude Code projects directory
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def get_project_dir() -> Path:
    """Get the Claude project directory to index.

    Set CLAUDE_MEMORY_PROJECT to override. Otherwise auto-detects based on home path.
    """
    if env_path := os.environ.get("CLAUDE_MEMORY_PROJECT"):
        return Path(env_path).expanduser()

    # Auto-detect based on home directory structure
    home = Path.home()
    if len(home.parts) > 1:
        # Encode path with dashes (e.g., /Users/foo -> -Users-foo, /home/foo -> -home-foo)
        encoded = "-" + "-".join(home.parts[1:])
        return CLAUDE_PROJECTS / encoded

    raise ValueError(
        "Could not auto-detect project directory. "
        "Set CLAUDE_MEMORY_PROJECT environment variable."
    )


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


# Resolved paths
PROJECT_DIR = get_project_dir()
STORAGE_DIR = get_storage_dir()
CHUNKS_FILE = STORAGE_DIR / "chunks.jsonl"
CHROMA_DIR = STORAGE_DIR / "chroma"
PROCESSED_FILE = STORAGE_DIR / "processed.json"
COLLECTION_NAME = get_collection_name()

# Embedding model (all-mpnet-base-v2 is recommended for quality)
EMBEDDING_MODEL = os.environ.get("CLAUDE_MEMORY_MODEL", "all-mpnet-base-v2")


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
