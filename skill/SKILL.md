---
name: claude-memory
description: Semantic search across past Claude Code conversations. This skill should be used when the user asks about previous discussions ("what did we discuss about X?", "how did we handle Y?"), before making architectural decisions to check for prior context, or when the user references past work that isn't in the current conversation context.
---

# Claude Memory

## Overview

Search past Claude Code conversations using hybrid search (semantic vectors + BM25 keyword matching). Find relevant prior discussions even if you only remember the topic or exact terms.

## When to Use

- User asks "what did we discuss about X?" or "how did we handle Y before?"
- Before making architectural decisions (check for prior context)
- When user references past work that isn't in current context
- User mentions something that seems to relate to previous conversations

## Search Command

```bash
claude-memory search "query" [-n NUM] [-c CONTEXT]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-n` | 5 | Number of results |
| `-c` | 1 | Turns of context before/after each result |

## Examples

```bash
# Basic search
claude-memory search "authentication setup"

# More results with extra context
claude-memory search "how did we handle the JWT bug" -n 10 -c 3

# Find by exact class/function name (BM25 catches this)
claude-memory search "UserService refactor"
```

## Reading Results

Each result includes:

- **Conversation excerpt** with surrounding context
- **Session ID** and **timestamp**
- **Distance score** (lower = more relevant)
- **Tools used** (Read, Bash, Edit, etc.)
- **Files touched** from tool calls

The tool metadata helps find conversations by what was *done*, not just what was *said*.

## Hybrid Search

Queries use both semantic similarity and keyword matching:

| Query type | Vectors find | BM25 catches |
|------------|--------------|--------------|
| Topic | Related discussions | — |
| Exact term | Related discussions | Exact matches |
| Class name | Code discussions | Exact "ClassName" |

This means you'll find results whether you remember exact terms or just the general topic.

## Notes

- The index auto-syncs via Claude Code hooks (SessionStart, PreCompact, SessionEnd)
- If no results found, run `claude-memory sync` first
- Use `-c 3` or higher when you need more surrounding context
