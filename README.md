# claude-memory

Semantic search across Claude Code conversations.

## Why

Each Claude Code conversation is isolated. Past decisions, approaches, and context are lost unless manually documented. `claude-memory` captures what you *wouldn't* bother to document:

- The quick "should we use X or Y?" discussion where you picked X
- Alternatives you explored but didn't choose
- Context that felt obvious at the time but you'll forget in 3 months

## Quick Start

```bash
pip install claude-memory

claude-memory sync                              # Index conversations
claude-memory search "how did we handle auth"   # Search
```

## Features

- **Hybrid search** - Combines semantic vectors with BM25 keyword matching
- **Tool metadata** - Track which files and commands were used in each conversation
- **Context expansion** - Show surrounding turns inline (1 by default, `-c N` for more)
- **Token-aware chunking** - Splits long exchanges to fit embedding model limits
- **Context windows** - Each chunk includes surrounding turns for better matching
- **Conversation summaries** - Optional LLM-generated summaries via Ollama
- **Multi-machine sync** - Share memory via git

## How It Works

```
~/.claude/projects/**/*.jsonl    →    chunks-{machine}.jsonl    →    ChromaDB + SQLite FTS5
     (conversations)                    (git-synced text)           (vector + keyword index)
```

1. **Parse** - Reads Claude conversation files
2. **Chunk** - Splits into searchable pieces with context
3. **Index** - Vector embeddings (ChromaDB) + keyword index (SQLite FTS5)
4. **Search** - Hybrid: combines semantic similarity with BM25 keyword matching

### Hybrid Search

Searches use both **vector similarity** (semantic meaning) and **BM25 keyword matching** (exact terms):

| Query | Vector finds | BM25 catches |
|-------|--------------|--------------|
| "authentication" | Discussions about auth, login, security | Exact "authentication" mentions |
| "UserService class" | Service layer discussions | Exact "UserService" matches |
| "the jwt bug" | Auth-related bugs | Exact "jwt" keyword |

This ensures you find results whether you remember the exact terms or just the topic.

## Commands

| Command | Description |
|---------|-------------|
| `sync` | Index new conversations |
| `search "query"` | Hybrid search (5 results, 1 turn context by default) |
| `stats` | Show index statistics |
| `rebuild` | Rebuild index from chunk files |
| `summarize` | Generate summaries (requires Ollama) |
| `config` | Show configuration |

## Configuration

Set via environment variables (add to `~/.zshrc`):

```bash
export CLAUDE_MEMORY_STORAGE=~/my-synced-folder/.memory   # Where to store chunks
export CLAUDE_MEMORY_PROJECT=~/.claude/projects/myproject  # Limit to one project
```

Default storage is `~/.claude-memory`. Default indexes all projects.

## Auto-sync

To sync automatically without manual commands, see [docs/auto-sync.md](docs/auto-sync.md).

## Multi-Machine Sync

To share memory across machines via git, see [docs/multi-machine.md](docs/multi-machine.md).

## Summaries (Optional)

Generate 2-3 sentence summaries per conversation for high-level search:

```bash
# One-time setup
brew install ollama
ollama pull qwen2.5:1.5b

# Generate summaries
claude-memory summarize
```

Once Ollama is available, `sync` auto-generates summaries for new conversations.

## Context & Tool Metadata

Search results include surrounding context and tool metadata by default:

```bash
# Just search - shows 5 results with 1 turn before/after
claude-memory search "authentication bug"

# More context if needed
claude-memory search "authentication bug" -c 3
```

Each result shows:
- **Tools used** - Which tools were called (Read, Bash, Edit, etc.)
- **Files touched** - File paths from tool calls

This lets you find conversations by what was *done*, not just what was *said*.

## Claude Code Integration

Add to your `CLAUDE.md`:

```markdown
## Claude Memory

Search past conversations: `claude-memory search "query"`

Use when:
- User asks "what did we discuss about X?"
- Before architectural decisions (check for prior context)
- When user references past work
```

Or copy the skill:
```bash
cp -r skill ~/.claude/skills/claude-memory
```

## Troubleshooting

**No chunks found:**
```bash
claude-memory config   # Check paths are correct
claude-memory sync     # Index conversations
```

**Storage dir wrong:** Add the export to `~/.zshrc` and run `source ~/.zshrc`.

## License

MIT
