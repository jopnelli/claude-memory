# claude-memory

Semantic search across Claude Code conversations with git-friendly sync for multi-machine setups.

## Why

Each Claude Code conversation is isolated. Past decisions, approaches, and context are lost unless manually documented. `claude-memory` preserves:

- Trade-offs discussed
- Alternatives considered
- The "why" behind decisions
- Context from past sessions

## Features

- **Semantic search** - Find relevant past conversations by meaning, not just keywords
- **Git-friendly sync** - Share memory across machines via any git-synced directory
- **Local embeddings** - Uses sentence-transformers, no API calls needed
- **Fast** - ChromaDB vector store for sub-second searches

## Architecture

```
~/.claude/projects/<project>/*.jsonl    (Claude Code conversations)
                    │
                    ▼
           ┌─────────────────────┐
           │  claude-memory sync │   Parse → Chunk → Append
           └─────────────────────┘
                    │
                    ▼
<storage-dir>/
├── chunks.jsonl     ← Git-synced (text only, no embeddings)
└── chroma/          ← Local only (embeddings, rebuild from chunks)
                    │
                    ▼
           ┌─────────────────────┐
           │ claude-memory search│   Embed query → Vector similarity
           └─────────────────────┘
```

The key insight: **sync text, embed locally**. The `chunks.jsonl` file contains only text and metadata—no embeddings. Each machine generates its own embeddings from this shared text file. This keeps the synced file small and text-diffable.

## Installation

```bash
pip install claude-memory
```

Or install from source:

```bash
git clone https://github.com/jopnelli/claude-memory.git
cd claude-memory
pip install -e .
```

## Quick Start

```bash
# Index your conversations
claude-memory sync

# Search past conversations
claude-memory search "how did we handle authentication"

# Check configuration
claude-memory config
```

## Configuration

Configuration is via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_MEMORY_PROJECT` | Claude project directory to index | Auto-detected from home path |
| `CLAUDE_MEMORY_STORAGE` | Storage directory for chunks and index | `~/.claude-memory` |
| `CLAUDE_MEMORY_COLLECTION` | ChromaDB collection name | `conversations` |
| `CLAUDE_MEMORY_MODEL` | Sentence-transformers model | `all-MiniLM-L6-v2` |

### Multi-Machine Setup

To share memory across machines (e.g., laptop and remote VM):

1. Set `CLAUDE_MEMORY_STORAGE` to a git-synced directory:
   ```bash
   export CLAUDE_MEMORY_STORAGE=~/dotfiles/.claude-memory
   # or
   export CLAUDE_MEMORY_STORAGE=~/obsidian-vault/.memory
   ```

2. Add to your `.gitignore` in that directory:
   ```
   chroma/
   processed.json
   ```

3. Only `chunks.jsonl` syncs between machines. Each machine rebuilds its own embeddings.

### Auto-sync on Session Start

Add to your shell config or a Claude Code hook:

```bash
# ~/.bashrc or ~/.zshrc
claude-memory sync -q  # -q for quiet mode
```

Or create a Claude Code session-start hook:

```bash
# ~/.claude/hooks/session-start
#!/bin/bash
claude-memory sync -q
```

## Commands

| Command | Description |
|---------|-------------|
| `claude-memory sync` | Parse new conversations, update index |
| `claude-memory search "query"` | Semantic search (use `-n` for more results) |
| `claude-memory stats` | Show index statistics |
| `claude-memory rebuild` | Force rebuild index from chunks.jsonl |
| `claude-memory config` | Show current configuration |

## How It Works

1. **Parse**: Reads Claude Code conversation files (`~/.claude/projects/<project>/*.jsonl`)
2. **Chunk**: Extracts user+assistant exchanges, skipping tool calls and thinking blocks
3. **Store**: Appends chunks to `chunks.jsonl` (text + metadata, no embeddings)
4. **Embed**: Generates embeddings locally using sentence-transformers
5. **Index**: Stores vectors in ChromaDB for fast similarity search

## Claude Code Integration

Add this to your `CLAUDE.md` so Claude knows to use memory:

```markdown
## Episodic Memory

Search past conversations with `claude-memory search "query"`.

Use when:
- User asks "what did we discuss about X?"
- Before making architectural decisions (check for prior context)
- When user references past work
```

## License

MIT
