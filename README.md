# claude-memory

Semantic search across Claude Code conversations with git-friendly sync for multi-machine setups.

## Why

Each Claude Code conversation is isolated. Past decisions, approaches, and context are lost unless manually documented. `claude-memory` automatically captures what you *wouldn't* bother to document:

- The quick "should we use X or Y?" discussion where you picked X
- Alternatives you explored but didn't choose
- Context that felt obvious at the time but you'll forget in 3 months
- The reasoning that happens in conversation but never makes it to formal docs

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
| `CLAUDE_MEMORY_MODEL` | Sentence-transformers model | `all-mpnet-base-v2` |

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

4. **First-time setup order** (to avoid merge conflicts):
   - Machine A: `claude-memory sync` → git commit/push
   - Machine B: git pull → `claude-memory sync` → git commit/push
   - Machine A: git pull → `claude-memory rebuild`

   If you do get a merge conflict in `chunks.jsonl`, just accept both versions - the system automatically deduplicates by chunk ID during rebuild.

5. **Index all projects** (optional): Set `CLAUDE_MEMORY_PROJECT="*"` to index conversations from all Claude Code projects, not just the auto-detected one.

### Auto-sync on Session Start

Add a Claude Code hook in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "(claude-memory sync -q &>/dev/null &)",
            "timeout": 1
          }
        ]
      }
    ]
  }
}
```

**Note:** The subshell wrapper `( ... &)` is important. Without it, the sync command blocks Claude startup for several seconds while loading the embedding model. The subshell detaches the process immediately, letting sync run in the background.

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
## Claude Memory

Search past conversations with `claude-memory search "query"`.

Use when:
- User asks "what did we discuss about X?"
- Before making architectural decisions (check for prior context)
- When user references past work
```

## License

MIT
