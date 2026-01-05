# Auto-sync Setup

Automatically sync conversations without manual `claude-memory sync` commands.

## macOS: launchd (Recommended)

Uses file watching to sync when Claude conversations change.

### 1. Create the launchd job

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

Replace `YOUR_USERNAME` with your actual username. Check the path to claude-memory with `which claude-memory`.

### 2. Load the job

```bash
launchctl load ~/Library/LaunchAgents/com.claude-memory-sync.plist
```

### 3. Add PreCompact hook (optional but recommended)

Add to `~/.claude/settings.json`:

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

This ensures conversations are indexed before Claude compacts the context.

### How it works

| Component | Trigger | Purpose |
|-----------|---------|---------|
| WatchPaths | Conversation files change | Syncs within 60s of Claude activity |
| PreCompact | Before context compaction | Indexes full conversation before summarization |

The job only runs when Claude is in use (files changing), not on a fixed timer.

---

## Linux: Hooks Only

Without launchd, use Claude hooks with background execution:

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

The `( ... &)` wrapper backgrounds the command for fast startup. Don't use it for PreCompact—that must complete before compaction.

**Caveat:** SessionEnd hooks don't fire reliably on Ctrl+C, so some short sessions may not sync until the next SessionStart.

---

## Verifying it works

Check the log:
```bash
tail -f /tmp/claude-memory-sync.log
```

Or check stats after a conversation:
```bash
claude-memory stats
```
