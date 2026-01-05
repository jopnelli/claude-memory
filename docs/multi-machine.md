# Multi-Machine Setup

Share memory across machines (e.g., Mac laptop and remote VM) using git.

## How it works

```
Machine A                          Machine B
─────────                          ─────────
conversations → chunks-a.jsonl     conversations → chunks-b.jsonl
                     │                                  │
                     └──────── git sync ────────────────┘
                                   │
                              shared text
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
         chroma/ (local)                           chroma/ (local)
         embeddings A                              embeddings B
```

Each machine:
- Writes to its own `chunks-{machine_id}.jsonl` file
- Reads from all chunk files (merging memories from all machines)
- Maintains local embeddings (not synced)

## Setup

### 1. Choose a git-synced directory

Use any directory that syncs between machines (Obsidian vault, dotfiles, etc.):

```bash
# Example: Obsidian vault
export CLAUDE_MEMORY_STORAGE=~/obsidian/.memory
```

Add this to your shell profile (`~/.zshrc` or `~/.bashrc`) on each machine.

### 2. Add to .gitignore

In your storage directory:

```
chroma/
processed.json
```

Only `chunks-*.jsonl` files should sync.

### 3. Initial sync

**First machine:**
```bash
claude-memory sync
git add chunks-*.jsonl
git commit -m "Add claude-memory chunks"
git push
```

**Other machines:**
```bash
git pull
claude-memory sync    # Creates local embeddings + adds this machine's chunks
git add chunks-*.jsonl
git commit -m "Add chunks from machine B"
git push
```

### 4. Ongoing sync

After pulling new chunks from another machine:
```bash
git pull
claude-memory rebuild   # Rebuilds index with all chunks
```

## Handling conflicts

If you get a merge conflict in `chunks-*.jsonl`:

1. Accept both versions (the content is append-only)
2. Run `claude-memory rebuild`

The system deduplicates by chunk ID, so duplicate entries are harmless.

## Machine ID

Each machine is identified by hostname. To override:

```bash
export CLAUDE_MEMORY_MACHINE_ID=my-laptop
```

This affects which chunk file is written to (`chunks-my-laptop.jsonl`).

## Troubleshooting

**Chunks not appearing after pull:**
```bash
claude-memory rebuild
```

**Wrong storage directory:**
```bash
claude-memory config | grep "Storage dir"
```

If it shows `~/.claude-memory` instead of your synced directory, the env var isn't set in the current shell.
