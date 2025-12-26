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

Install globally so the command is available in hooks and across shell sessions:

```bash
# Recommended: uses pipx for isolated installation
pipx install claude-memory

# Alternative: install to user site-packages
pip install --user claude-memory
```

After installation, verify it's in your PATH:

```bash
which claude-memory
# Should show: ~/.local/bin/claude-memory
```

If `~/.local/bin` isn't in your PATH, add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Installing from Source

For development or if the package isn't on PyPI yet:

```bash
git clone https://github.com/jopnelli/claude-memory.git
cd claude-memory
pip install --user -e .
```

**Note:** Don't install into a virtualenv if you want the hook to work—hooks run in a non-interactive shell where venvs aren't activated.

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

To share memory across machines (e.g., Mac laptop and remote VM):

**On each machine:**

1. Install claude-memory:
   ```bash
   pip install --user claude-memory
   # Verify it's accessible
   which claude-memory
   ```

2. Add `CLAUDE_MEMORY_STORAGE` to your shell profile (`~/.zshrc` or `~/.bashrc`):
   ```bash
   echo 'export CLAUDE_MEMORY_STORAGE=~/obsidian/.memory' >> ~/.zshrc
   source ~/.zshrc
   ```

   Then verify it's set:
   ```bash
   claude-memory config | grep "Storage dir"
   # Should show your chosen path, not ~/.claude-memory
   ```

3. Add to your `.gitignore` in that directory:
   ```
   chroma/
   processed.json
   ```

4. Run initial sync to download the embedding model (~420MB, one-time):
   ```bash
   claude-memory sync
   ```

**Syncing between machines:**

Only `chunks.jsonl` syncs via git. Each machine maintains its own embeddings in `chroma/`.

First-time setup order (to avoid merge conflicts):
- Machine A: `claude-memory sync` → git commit/push
- Machine B: git pull → `claude-memory sync` → git commit/push
- Machine A: git pull → `claude-memory rebuild`

If you get a merge conflict in `chunks.jsonl`, accept both versions—the system deduplicates by chunk ID during rebuild.

**Optional:** Set `CLAUDE_MEMORY_PROJECT="*"` to index conversations from all Claude Code projects.

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

**Important details:**

- The subshell wrapper `( ... &)` is critical. Without the parentheses, the sync command blocks Claude startup for 10-20 seconds while loading the embedding model.
- The command assumes `claude-memory` is in PATH. If you installed to a venv or non-standard location, use the full path:
  ```json
  "command": "(~/.local/bin/claude-memory sync -q &>/dev/null &)"
  ```

**Troubleshooting slow startup:**

1. Verify the command returns instantly:
   ```bash
   time (claude-memory sync -q &>/dev/null &)
   # Should show: real 0m0.000s
   ```

2. If startup is slow, check:
   - Missing parentheses in the hook command
   - `claude-memory` not in PATH (hooks run in non-interactive shells)
   - First run downloading the embedding model (run `claude-memory sync` manually once first)

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

### Option 1: Install the Skill (Recommended)

Copy the skill to your Claude skills directory:

```bash
cp -r skill ~/.claude/skills/claude-memory
```

This enables Claude to automatically search memory when relevant (e.g., when you ask "what did we discuss about X?").

### Option 2: Add to CLAUDE.md

Alternatively, add this to your `CLAUDE.md`:

```markdown
## Claude Memory

Search past conversations with `claude-memory search "query"`.

Use when:
- User asks "what did we discuss about X?"
- Before making architectural decisions (check for prior context)
- When user references past work
```

## Troubleshooting

If `claude-memory stats` shows 0 chunks or fewer sessions than expected, run:

```bash
claude-memory config
```

**What to check:**
- If "Storage dir" shows `~/.claude-memory` but data exists elsewhere (e.g., `~/obsidian/.memory`), the `CLAUDE_MEMORY_STORAGE` env var isn't set in the current shell
- Fix: add the export to `~/.zshrc` or `~/.bashrc` per step 2 in Multi-Machine Setup, then `source` it

## License

MIT
