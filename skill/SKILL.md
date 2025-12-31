---
name: claude-memory
description: Semantic search across past Claude Code conversations. This skill should be used when the user asks about previous discussions ("what did we discuss about X?", "how did we handle Y?"), before making architectural decisions to check for prior context, or when the user references past work that isn't in the current conversation context.
---

# Claude Memory

## Overview

Search and retrieve context from past Claude Code conversations using semantic similarity. The `claude-memory` CLI indexes conversation history and enables natural language queries to find relevant prior discussions.

## When to Use

- User asks "what did we discuss about X?" or "how did we handle Y before?"
- Before making architectural decisions (check for prior context)
- When user references past work that isn't in current context
- User mentions something that seems to relate to previous conversations

## Commands

```bash
# Search for relevant past conversations
claude-memory search "query" [-n NUM_RESULTS]

# Sync new conversations to the index (usually runs automatically)
claude-memory sync

# Show index statistics
claude-memory stats

# Force rebuild the index
claude-memory rebuild

# Show configuration
claude-memory config
```

## Usage Pattern

1. **Search first**: Run `claude-memory search "topic"` to find relevant context
2. **Review results**: Each result shows the conversation excerpt, session ID, timestamp, and similarity distance (lower = more relevant)
3. **Use context**: Incorporate findings into your response or decision-making

## Example

```bash
# User asks: "How did we set up the authentication?"
claude-memory search "authentication setup" -n 3
```

This returns the most relevant past conversations about authentication, including decisions made, alternatives considered, and implementation details.

## Notes

- Results are ranked by semantic similarity, not just keyword matching
- Lower distance scores indicate higher relevance
- The index auto-syncs via Claude Code hooks:
  - `SessionStart`: Catches up on missed syncs
  - `PreCompact`: Indexes chunks before context compaction (prevents data loss)
  - `SessionEnd`: Final sync when session ends
- If no results found, run `claude-memory sync` first
