# retro

Extracts learnings from Claude Code conversation history.

Reads `~/.claude/projects/` conversation logs, filters for interesting ones, and extracts actionable knowledge using Claude. Two-pass workflow: analyze first, review, then store approved learnings.

## Usage

```
# analyze all projects (most conversations first)
cargo run -p conversation-retrospective -- run

# filter to a specific project
cargo run -p conversation-retrospective -- run --project backend

# custom state file
cargo run -p conversation-retrospective -- run --state my-state.json
```

After `run` completes, edit `state.json` and set `"approved": true` on learnings you want to keep.

```
# store approved learnings to memory MCP
MEMORY_API_KEY=... cargo run -p conversation-retrospective -- store

# vote on recalled memories (reinforcement)
MEMORY_API_KEY=... cargo run -p conversation-retrospective -- vote

# filter to a specific project
MEMORY_API_KEY=... cargo run -p conversation-retrospective -- vote --project backend
```

## Voting (reinforcement)

The `vote` subcommand scans conversation logs for memory MCP tool calls (`recall_memory`, `session_start`, `search_by_tags`, etc.), extracts which memories were recalled, and uses an LLM (haiku) to judge whether each memory was helpful or harmful in context.

Votes are submitted to the memory server's REST API (`POST /api/v1/memories/{id}/vote`). These votes feed into the confidence score (Wilson score lower bound) used during recall ranking, creating a reinforcement loop: useful memories surface more, noisy ones sink.

Conversations are chunked at ~80K chars with overlap. When a memory appears in multiple chunks, evaluations are aggregated by majority vote (ties go to helpful). State is tracked in `vote-state.json` for incremental runs.

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
