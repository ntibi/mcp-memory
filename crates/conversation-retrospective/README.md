# retro

Extracts learnings from Claude Code conversation history.

Reads `~/.claude/projects/` conversation logs, filters for interesting ones, and extracts actionable knowledge using Claude. Two-pass workflow: analyze first, review, then store approved learnings.

## Usage

```
# analyze all projects (most conversations first)
cargo run -p conversation-retrospective -- run

# filter to a specific project
cargo run -p conversation-retrospective -- run --project anyshift

# custom state file
cargo run -p conversation-retrospective -- run --state my-state.json
```

After `run` completes, edit `state.json` and set `"approved": true` on learnings you want to keep.

```
# store approved learnings to memory MCP
MEMORY_API_KEY=... cargo run -p conversation-retrospective -- store
```

## Pipeline

```
conversation.jsonl
  -> parse (extract user/assistant text)
  -> heuristic filter (skip single-turn, <500 chars)
  -> condense (strip tool calls, cap at 30k chars)
  -> haiku classify (interesting or not)
  -> sonnet extract (pull out learnings)
  -> state.json
  -> [manual review]
  -> store to memory MCP
```

## State file

The state file tracks progress per conversation. The tool is fully recoverable -- interrupt and restart at any time. Already-processed conversations are skipped.

Each conversation gets a status: `heuristic_skipped`, `not_interesting`, `extracted`, `approved`, `stored`, or `failed`.

## Environment

| Variable | Default | Description |
|---|---|---|
| `MEMORY_URL` | `http://localhost:3001` | Memory MCP base URL |
| `MEMORY_API_KEY` | required for `store` | Memory MCP API key |
| `RUST_LOG` | - | Log level (e.g. `warn` to see failures) |

## Requirements

- `claude` CLI on PATH (uses `--print` mode with `--json-schema` for structured output)
- Must be run outside of a Claude Code session (or the nested session detection is bypassed automatically)
