# Chunking Improvements

## The Problem

`all-mpnet-base-v2` has a **384 token hard limit** (~1,500 chars). Text beyond this is silently truncated.

Current chunk distribution:
- Mean: 2,183 chars (~485 tokens) - often exceeds limit
- Max: 20,058 chars (~4,500 tokens) - losing ~90% of content

**Result:** Long conversations lose most of their searchable content.

---

## Solution

Split chunks that exceed the context window. Keep it simple.

### Design Principles

Per [plan.md](obsidian://open?vault=jopnelli&file=Projects%2FClaudeMemory%2Fplan):
- **Minimal friction** - no new dependencies
- **Works out of the box** - no LLM/API required
- **Easy to navigate** - one code path, not three

### Implementation

```python
# Native recursive splitter - no LangChain needed
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]
MAX_CHUNK_CHARS = 1400  # ~310 tokens, safe margin for 384 limit
OVERLAP_CHARS = 280     # 20% overlap

def recursive_split(text: str, separators: list[str] = SEPARATORS) -> list[str]:
    """Split text recursively at natural boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    # Try each separator in order
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= MAX_CHUNK_CHARS:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If single part exceeds limit, try finer separator
                    if len(part) > MAX_CHUNK_CHARS:
                        chunks.extend(recursive_split(part, separators[separators.index(sep)+1:]))
                    else:
                        current = part

            if current:
                chunks.append(current)

            return chunks if chunks else [text]

    # Last resort: hard split by characters
    return [text[i:i+MAX_CHUNK_CHARS] for i in range(0, len(text), MAX_CHUNK_CHARS - OVERLAP_CHARS)]


def add_overlap(chunks: list[str], overlap: int = OVERLAP_CHARS) -> list[str]:
    """Add overlap between chunks for context continuity."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        # Prepend end of previous chunk
        prev_overlap = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
        result.append(prev_overlap + " " + chunks[i])

    return result
```

### Integration with Existing Chunker

```python
# In chunker.py - modify create_chunk_with_context()

def create_chunks_with_context(exchanges, current_index) -> list[Chunk]:
    """Create chunk(s) from a turn, splitting if needed."""
    # Build the full text with context (existing logic)
    text = build_context_text(exchanges, current_index)

    # Split if exceeds limit
    if len(text) <= MAX_CHUNK_CHARS:
        return [Chunk(id=..., text=text, ...)]

    # Split and create multiple chunks
    parts = add_overlap(recursive_split(text))
    return [
        Chunk(
            id=f"{base_id}-{i}",
            text=part,
            parent_turn_id=base_id,  # Track original turn
            chunk_index=i,
            total_chunks=len(parts),
            ...
        )
        for i, part in enumerate(parts)
    ]
```

---

## Research Summary

### Why These Numbers

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max chunk | 1,400 chars | ~310 tokens, safe margin for 384 limit |
| Overlap | 280 chars (20%) | Industry standard 10-20%, prevents context loss |
| Separators | `\n\n` → `\n` → `. ` → ` ` | Preserve natural boundaries |

### Chunk Size vs Quality

| Chunk Size | Retrieval Quality | Notes |
|------------|-------------------|-------|
| 64-128 tokens | High precision | Too granular for conversations |
| **256-384 tokens** | **Optimal** | Best for our model's context window |
| 512+ tokens | Degraded | Truncation starts |

### What We're NOT Doing

| Approach | Why Not |
|----------|---------|
| LangChain splitter | Adds ~100 deps, violates "minimal friction" |
| Three-tier strategy | Complex branching, harder to maintain |
| Summary chunks | Requires LLM, violates "works out of box" |
| Semantic chunking | Requires embedding calls during sync |

Summaries remain an **optional** feature via `claude-memory summarize` (requires Ollama).

---

## Metadata Schema

When a turn splits into multiple chunks:

```python
@dataclass
class Chunk:
    id: str                    # "{turn_uuid}" or "{turn_uuid}-{index}"
    text: str
    timestamp: str
    session_id: str
    chunk_type: Literal["turn", "summary"] = "turn"
    turn_index: int = 0
    # New fields for split chunks:
    parent_turn_id: str = ""   # Original turn UUID (empty if not split)
    chunk_index: int = 0       # Position within split (0, 1, 2...)
    total_chunks: int = 1      # How many chunks this turn produced
```

This enables:
- Fetching all chunks from a turn
- Deduplicating search results by parent turn
- Maintaining order during retrieval

---

## Migration

```bash
# After implementing, rebuild to re-chunk existing conversations
claude-memory rebuild
```

- Long chunks get split into multiple smaller chunks
- All content becomes searchable
- No data loss

---

## Checklist

- [x] Add `recursive_split()` to chunker.py (native, no deps)
- [x] Add `parent_turn_id`, `chunk_index`, `total_chunks` to Chunk dataclass
- [x] Update `create_chunk_with_context()` to return list
- [x] Handle new fields in `load_all_chunks()` (backwards compat)
- [x] Update search to dedupe by `parent_turn_id`
- [x] Add tests for splitting behavior
- [x] Test `rebuild` with existing chunks.jsonl

---

## Sources

- [Sentence Transformers: all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) - 384 token max
- [Pinecone Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Weaviate Chunking Guide](https://weaviate.io/blog/chunking-strategies-for-rag)
