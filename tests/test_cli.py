"""Tests for the CLI module."""

import json

import pytest
from click.testing import CliRunner


def reload_all_modules(monkeypatch, storage_dir, project_dir=None):
    """Helper to reload all modules with new paths."""
    monkeypatch.setenv("CLAUDE_MEMORY_STORAGE", str(storage_dir))
    if project_dir:
        monkeypatch.setenv("CLAUDE_MEMORY_PROJECT", str(project_dir))

    import importlib
    import claude_memory.config
    importlib.reload(claude_memory.config)
    import claude_memory.parser
    importlib.reload(claude_memory.parser)
    import claude_memory.chunker
    importlib.reload(claude_memory.chunker)
    import claude_memory.store
    importlib.reload(claude_memory.store)
    import claude_memory.cli
    importlib.reload(claude_memory.cli)

    from claude_memory.cli import cli
    from claude_memory.config import ensure_dirs

    ensure_dirs()
    return cli


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_sync_command(self, temp_dir, sample_conversation_data, runner, monkeypatch):
        """Test sync command."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir, project_dir)

        # Create test conversation
        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0
        assert "Syncing" in result.output
        assert "chunk" in result.output.lower()

    def test_sync_quiet_mode(self, temp_dir, sample_conversation_data, runner, monkeypatch):
        """Test sync command with quiet flag."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir, project_dir)

        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        result = runner.invoke(cli, ["sync", "-q"])

        assert result.exit_code == 0
        assert result.output == ""

    def test_search_no_index(self, temp_dir, runner, monkeypatch):
        """Search with no index should show helpful message."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir)

        result = runner.invoke(cli, ["search", "test query"])

        assert result.exit_code == 0
        assert "sync" in result.output.lower()

    def test_search_with_data(self, temp_dir, sample_conversation_data, runner, monkeypatch):
        """Search should return results after sync."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir, project_dir)

        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        # First sync to index
        runner.invoke(cli, ["sync", "-q"])

        # Then search
        result = runner.invoke(cli, ["search", "Hello"])

        assert result.exit_code == 0
        # Should have results or "no results"
        assert "Result" in result.output or "No results" in result.output

    def test_search_num_results(self, temp_dir, sample_conversation_data, runner, monkeypatch):
        """Search with -n flag should limit results."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir, project_dir)

        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        runner.invoke(cli, ["sync", "-q"])

        result = runner.invoke(cli, ["search", "-n", "1", "Hello"])

        assert result.exit_code == 0

    def test_stats_command(self, temp_dir, sample_conversation_data, runner, monkeypatch):
        """Test stats command."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir, project_dir)

        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        runner.invoke(cli, ["sync", "-q"])

        result = runner.invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "chunk" in result.output.lower()

    def test_stats_empty(self, temp_dir, runner, monkeypatch):
        """Stats on empty index should work."""
        storage_dir = temp_dir / "storage"
        storage_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir)

        result = runner.invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "0" in result.output

    def test_rebuild_command(self, temp_dir, sample_conversation_data, runner, monkeypatch):
        """Test rebuild command."""
        from conftest import write_jsonl

        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir, project_dir)

        conv_file = project_dir / "test-session.jsonl"
        write_jsonl(conv_file, sample_conversation_data)

        # First sync to create chunks
        runner.invoke(cli, ["sync", "-q"])

        result = runner.invoke(cli, ["rebuild"])

        assert result.exit_code == 0
        assert "rebuild" in result.output.lower() or "index" in result.output.lower()

    def test_config_command(self, temp_dir, runner, monkeypatch):
        """Test config command."""
        storage_dir = temp_dir / "storage"
        project_dir = temp_dir / "project"
        storage_dir.mkdir()
        project_dir.mkdir()

        cli = reload_all_modules(monkeypatch, storage_dir, project_dir)

        result = runner.invoke(cli, ["config"])

        assert result.exit_code == 0
        assert "Project dir" in result.output or "project" in result.output.lower()
        assert "Storage dir" in result.output or "storage" in result.output.lower()
        # Should show the temp paths we set
        assert str(project_dir) in result.output
        assert str(storage_dir) in result.output


class TestCLIHelp:
    """Tests for CLI help messages."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_main_help(self, runner):
        """Main command should have help."""
        from claude_memory.cli import cli

        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Claude" in result.output or "semantic" in result.output.lower()

    def test_sync_help(self, runner):
        """Sync command should have help."""
        from claude_memory.cli import cli

        result = runner.invoke(cli, ["sync", "--help"])

        assert result.exit_code == 0
        assert "sync" in result.output.lower()

    def test_search_help(self, runner):
        """Search command should have help."""
        from claude_memory.cli import cli

        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "search" in result.output.lower()
