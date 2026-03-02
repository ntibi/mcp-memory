# memory

Semantic memory store for LLM agents. Stores, retrieves and curates memories using vector similarity search over SQLite.

Exposes an MCP server (Streamable HTTP), a REST API and a web UI.

## Quick start

```bash
cargo run --release
# server starts at http://localhost:8000
# UI at /ui, API at /api/v1, MCP at /mcp
```

## Usage

### CLI

```
memory-server [OPTIONS]

Options:
  -c, --config <CONFIG>            [default: config.toml]
      --listen-addr <LISTEN_ADDR>
      --db-path <DB_PATH>
```

### MCP (Claude Code)

Start the server, then add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "memory": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

```bash
claude --mcp-config mcp.json
```

Three tools are exposed:
- `store_memory` — store content with optional tags
- `recall_memory` — semantic search, returns top N ranked results
- `search_by_tag` — exact tag match

### REST API

```
GET/POST   /api/v1/memories              — list / create
GET/PUT/DELETE /api/v1/memories/{id}      — get / update / delete
POST       /api/v1/memories/{id}/vote     — vote (helpful/harmful)
GET        /api/v1/memories/duplicates    — find similar memories
GET        /api/v1/curation/suggestions   — list curation suggestions
POST       /api/v1/curation/suggestions/{id}/apply|dismiss
GET        /api/v1/health
```

### Web UI

Browse, search, tag, vote and edit memories at `http://localhost:8000/ui`. Supports light/dark theme.

## Configuration

Layered: defaults → `config.toml` → environment variables (`MEMORY__` prefix) → CLI flags.

```toml
listen_addr = "127.0.0.1:8000"
db_path = "memory.db"
api_key = ""  # empty = no auth

[embedding]
provider = "local"       # "local" (ONNX) or "remote" (OpenAI-compatible)
model = "all-MiniLM-L6-v2"
# api_key = ""           # for remote provider
# api_url = ""           # for remote provider

[scoring]
relevance_weight = 0.6
confidence_weight = 0.25
recency_weight = 0.15

[curation]
interval_secs = 3600
similarity_threshold = 0.85
```

Environment variable examples:
```bash
MEMORY__API_KEY=secret
MEMORY__LISTEN_ADDR=0.0.0.0:8000
MEMORY__EMBEDDING__PROVIDER=remote
MEMORY__EMBEDDING__API_KEY=sk-...
```

## How it works

Memories are stored in SQLite with vector embeddings in a `sqlite-vec` virtual table.

**Storage**: each memory gets a ULID, content is embedded (384-dim via `all-MiniLM-L6-v2` locally, or any OpenAI-compatible API), and stored alongside the vector in SQLite.

**Recall**: query is embedded, cosine similarity ANN search runs against `sqlite-vec` with 3x overfetch. Results are re-ranked using a weighted score:

```
score = relevance * 0.6 + confidence * 0.25 + recency * 0.15
```

- **relevance** — cosine similarity between query and memory embeddings
- **confidence** — ratio of helpful votes: `helpful / (helpful + harmful)`, defaults to 0.5
- **recency** — `1 / (1 + age_days)`

**Curation**: background task runs periodically to find near-duplicate memories (above similarity threshold) and suggests merges/prunes.

## Docker

```bash
docker build -t memory .
docker run -p 8000:8000 -e MEMORY__LISTEN_ADDR=0.0.0.0:8000 -e MEMORY__API_KEY=secret -v memory-data:/data memory
```

## Architecture

```
crates/
  memory-core/    — storage, embedding, scoring, curation
  memory-server/  — axum server, MCP + REST + auth + config
  memory-ui/      — web interface (askama templates + HTMX)
```
