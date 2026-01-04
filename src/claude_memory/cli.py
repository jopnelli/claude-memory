"""CLI for claude-memory."""

import click

from .chunker import sync_chunks, load_all_chunks
from .store import Store, get_indexed_count
from .summarizer import sync_summaries, is_ollama_available, DEFAULT_MODEL
from .config import (
    CHUNKS_FILE,
    CHROMA_DIR,
    PROJECT_DIR,
    STORAGE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_machine_id,
    get_all_chunk_files,
)


@click.group()
def cli():
    """Semantic search across Claude Code conversations."""
    pass


@cli.command()
@click.option("-q", "--quiet", is_flag=True, help="Suppress output (for use in hooks)")
@click.option("--no-summaries", is_flag=True, help="Skip automatic summary generation")
def sync(quiet: bool, no_summaries: bool):
    """Sync new conversations and update the index."""

    def log(msg):
        if not quiet:
            click.echo(msg)

    log("Syncing conversations...")

    # First, sync chunks from conversations (fast - just parses files)
    new_chunks, new_files = sync_chunks()
    if new_chunks > 0:
        log(f"  Added {new_chunks} chunks from {new_files} conversations")
    else:
        log("  No new chunks found")

    # Auto-generate summaries if Ollama is available
    if not no_summaries and new_files > 0 and is_ollama_available():
        log("Generating summaries...")
        generated, failed = sync_summaries(quiet=True)
        if generated > 0:
            log(f"  Generated {generated} summaries")

    # Early exit: skip embedding model load if no work to do
    if new_chunks == 0:
        # Quick check if index is already in sync (without loading embedding model)
        chunks = load_all_chunks()
        indexed_count = get_indexed_count()
        if len(chunks) == indexed_count:
            log(f"Index up to date ({indexed_count} chunks)")
            return
        log(f"  Index out of sync ({len(chunks)} chunks, {indexed_count} indexed)")

    # Only load embedding model if we have work to do
    log("Updating index...")
    store = Store()
    indexed = store.rebuild_index()
    if indexed > 0:
        log(f"  Indexed {indexed} new chunks")
    else:
        log("  Index up to date")

    log(f"Total: {store.count()} chunks indexed")


@cli.command()
@click.argument("query")
@click.option("-n", "--num-results", default=5, help="Number of results to return")
def search(query: str, num_results: int):
    """Search conversations for relevant context."""
    store = Store()

    if store.count() == 0:
        click.echo("No chunks indexed. Run 'claude-memory sync' first.")
        return

    results = store.search(query, n=num_results)

    if not results:
        click.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        click.echo(f"\n{'='*60}")
        type_label = "📝 Summary" if result.chunk_type == "summary" else f"💬 Turn {result.turn_index + 1}"
        click.echo(f"Result {i} [{type_label}] (distance: {result.distance:.4f})")
        click.echo(f"Session: {result.session_id}")
        click.echo(f"Time: {result.timestamp}")
        click.echo(f"{'='*60}")
        click.echo(result.text)


@cli.command()
def stats():
    """Show index statistics."""
    store = Store()
    chunks = load_all_chunks()
    chunk_files = get_all_chunk_files()

    # Count by type
    turn_chunks = [c for c in chunks if c.chunk_type == "turn"]
    summary_chunks = [c for c in chunks if c.chunk_type == "summary"]

    click.echo(f"Machine ID: {get_machine_id()}")
    click.echo(f"Writing to: {CHUNKS_FILE}")
    click.echo(f"Reading from: {len(chunk_files)} file(s)")
    for f in chunk_files:
        click.echo(f"  - {f.name}")
    click.echo(f"ChromaDB dir: {CHROMA_DIR}")
    click.echo(f"Total chunks in files: {len(chunks)}")
    click.echo(f"  Turn chunks: {len(turn_chunks)}")
    click.echo(f"  Summary chunks: {len(summary_chunks)}")
    click.echo(f"Total chunks indexed: {store.count()}")

    if chunks:
        sessions = set(c.session_id for c in chunks)
        click.echo(f"Unique conversations: {len(sessions)}")
        if summary_chunks:
            coverage = len(summary_chunks) / len(sessions) * 100
            click.echo(f"Summary coverage: {coverage:.1f}%")


@cli.command()
def rebuild():
    """Force rebuild the index from chunks.jsonl."""
    click.echo("Clearing existing index...")
    store = Store()
    store.clear()

    click.echo("Rebuilding index...")
    indexed = store.rebuild_index()
    click.echo(f"Indexed {indexed} chunks")


@cli.command()
@click.option("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
@click.option("-q", "--quiet", is_flag=True, help="Suppress per-conversation output")
def summarize(model: str, quiet: bool):
    """Generate conversation summaries using Ollama.

    This creates summary chunks for each conversation, enabling high-level
    search like "what conversations discussed authentication?".

    Requires Ollama to be installed and running.
    """
    click.echo(f"Generating conversation summaries using {model}...")

    generated, failed = sync_summaries(model=model, quiet=quiet)

    if generated > 0 or failed > 0:
        click.echo(f"\nGenerated {generated} summaries ({failed} failed)")

        # Update index with new summaries
        click.echo("Updating index...")
        store = Store()
        indexed = store.rebuild_index()
        click.echo(f"Indexed {indexed} new chunks")
    else:
        click.echo("All conversations already have summaries")


@cli.command()
def config():
    """Show current configuration."""
    chunk_files = get_all_chunk_files()

    click.echo("Claude Memory Configuration")
    click.echo("=" * 50)
    click.echo(f"Machine ID:      {get_machine_id()}")
    click.echo(f"Project dir:     {PROJECT_DIR}")
    click.echo(f"  exists:        {PROJECT_DIR.exists()}")
    click.echo(f"Storage dir:     {STORAGE_DIR}")
    click.echo(f"  exists:        {STORAGE_DIR.exists()}")
    click.echo(f"Chunks file:     {CHUNKS_FILE}")
    click.echo(f"  exists:        {CHUNKS_FILE.exists()}")
    click.echo(f"All chunk files: {len(chunk_files)} file(s)")
    for f in chunk_files:
        click.echo(f"  - {f.name}")
    click.echo(f"ChromaDB dir:    {CHROMA_DIR}")
    click.echo(f"  exists:        {CHROMA_DIR.exists()}")
    click.echo(f"Collection:      {COLLECTION_NAME}")
    click.echo(f"Embedding model: {EMBEDDING_MODEL}")
    click.echo()
    click.echo("Environment variables:")
    click.echo("  CLAUDE_MEMORY_PROJECT    - Claude project directory to index")
    click.echo("  CLAUDE_MEMORY_STORAGE    - Storage directory (git-sync this)")
    click.echo("  CLAUDE_MEMORY_MACHINE_ID - Machine identifier (default: hostname)")
    click.echo("  CLAUDE_MEMORY_COLLECTION - ChromaDB collection name")
    click.echo("  CLAUDE_MEMORY_MODEL      - Embedding model name")


if __name__ == "__main__":
    cli()
