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
- **Context-aware chunking** - Each turn includes surrounding turns (1 before + 1 after) for bidirectional context
- **Conversation summaries** - LLM-generated summaries for high-level search (optional, uses Ollama)
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
| `CLAUDE_MEMORY_PROJECT` | Claude project directory to index | All directories in `~/.claude/projects/` |
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

**Note:** All project directories are indexed by default. Set `CLAUDE_MEMORY_PROJECT=/specific/path` to limit to a single directory.

### Auto-sync Setup

#### macOS: launchd + PreCompact Hook (Recommended)

This approach gives you instant startup while ensuring all conversations get indexed:

**1. Create a launchd job that watches for conversation changes:**

```bash
cat > ~/Library/LaunchAgents/com.claude-memory-sync.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.claude-memory-sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/.local/bin/claude-memory</string>
        <string>sync</string>
        <string>-q</string>
    </array>
    <key>WatchPaths</key>
    <array>
        <string>/Users/YOUR_USERNAME/.claude/projects</string>
    </array>
    <key>ThrottleInterval</key>
    <integer>60</integer>
    <key>StandardOutPath</key>
    <string>/tmp/claude-memory-sync.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/claude-memory-sync.log</string>
</dict>
</plist>
EOF
```

Replace `YOUR_USERNAME` with your actual username, and update the path to `claude-memory` if different (check with `which claude-memory`).

**2. Load the job:**

```bash
launchctl load ~/Library/LaunchAgents/com.claude-memory-sync.plist
```

**3. Add a PreCompact hook** in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "claude-memory sync -q",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

**How it works:**

| Component | Trigger | Purpose |
|-----------|---------|---------|
| **WatchPaths** | Conversation files change | Syncs within 60s of any Claude activity |
| **PreCompact** | Before context compaction | Ensures full conversation indexed before summarization |

The launchd job only runs when Claude is actually in use (files changing), not on a fixed timer. This means no wasted CPU when Claude isn't running, and conversations are indexed even if you Ctrl+C out of a session.

#### Linux/Alternative: Hooks Only

If launchd isn't available, use hooks with background subshells:

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
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "claude-memory sync -q",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

**Note:** The subshell wrapper `( ... &)` backgrounds the command for fast startup. Don't use it for PreCompact—that must complete before compaction.

**Caveat:** SessionEnd hooks don't fire reliably on Ctrl+C, so some short sessions may not sync until the next SessionStart

## Commands

| Command | Description |
|---------|-------------|
| `claude-memory sync` | Parse new conversations, update index |
| `claude-memory search "query"` | Semantic search (use `-n` for more results) |
| `claude-memory summarize` | Generate conversation summaries using Ollama |
| `claude-memory stats` | Show index statistics |
| `claude-memory rebuild` | Force rebuild index from chunks.jsonl |
| `claude-memory config` | Show current configuration |

## How It Works

1. **Parse**: Reads Claude Code conversation files (`~/.claude/projects/<project>/*.jsonl`)
2. **Chunk**: Extracts user+assistant exchanges with context from previous turns
3. **Store**: Appends chunks to `chunks.jsonl` (text + metadata, no embeddings)
4. **Embed**: Generates embeddings locally using sentence-transformers
5. **Index**: Stores vectors in ChromaDB for fast similarity search

### Context-Aware Chunking

Each turn is embedded with bidirectional context (1 turn before + 1 turn after). This means when you search for "authentication", you'll find turns where auth was discussed even if the specific turn doesn't mention the word—because the surrounding context captures the full flow.

```
Turn 3 chunk contains:
  [Turn 2 - before]
  ---
  [Turn 3 - current]
  ---
  [Turn 4 - after]
```

### Conversation Summaries (Automatic)

Summaries are generated automatically during sync when Ollama is available. Each conversation gets a 2-3 sentence summary for high-level search like "which conversations discussed database design?".

**Setup (one-time):**
```bash
# Install Ollama (https://ollama.ai)
brew install ollama
ollama pull qwen2.5:1.5b
```

Once set up, `sync` automatically generates summaries for new conversations. Search results show both turn matches (💬) and summary matches (📝).

To generate summaries for existing conversations:
```bash
claude-memory summarize
```

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
