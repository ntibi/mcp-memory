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

Start the server, then add it to Claude Code:

```bash
claude mcp add memory --transport http --url http://localhost:8000/mcp --header "Authorization: Bearer <your-api-key>"
```

Tools:
- `store_memory` — store content with tags
- `session_start` — combined tag search + semantic recall for session init
- `recall_memory` — semantic search, returns top N ranked results
- `update_memory` — update content/tags of an existing memory
- `delete_memory` — delete a memory by ID
- `search_by_tags` — exact tag match (AND semantics across multiple tags)

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
bootstrap_user = "admin"  # creates admin user + API key on first run (printed to logs)

[embedding]
provider = "local"       # "local" (ONNX) or "remote" (OpenAI-compatible)
model = "all-MiniLM-L6-v2"
# api_key = ""           # for remote provider
# api_url = ""           # for remote provider

[scoring]
relevance_weight = 0.6
confidence_weight = 0.25
recency_weight = 0.15
recency_half_life_days = 30.0

[curation]
interval_secs = 3600
similarity_threshold = 0.85
```

Environment variable examples:
```bash
MEMORY__BOOTSTRAP_USER=admin
MEMORY__LISTEN_ADDR=0.0.0.0:8000
MEMORY__EMBEDDING__PROVIDER=remote
MEMORY__EMBEDDING__API_KEY=sk-...
```

## Auth

Multi-user API key authentication. On first run, a bootstrap user is created with an API key printed to logs. Use it as a Bearer token or `api_key` cookie.

Admin users can manage users, keys and memories via the admin UI at `/ui/admin` or the REST API at `/api/v1/admin/`.

## How it works

Memories are stored in SQLite with vector embeddings in a `sqlite-vec` virtual table.

**Storage**: each memory gets a ULID, content is embedded (384-dim via `all-MiniLM-L6-v2` locally, or any OpenAI-compatible API), and stored alongside the vector in SQLite.

**Recall**: query is embedded, cosine similarity ANN search runs against `sqlite-vec` with 3x overfetch. Results are re-ranked using a weighted score:

```
score = relevance * 0.6 + confidence * 0.25 + recency * 0.15
```

- **relevance** — cosine similarity between query and memory embeddings
- **confidence** — Wilson score lower bound (95% CI) over helpful/harmful votes, defaults to 0.5
- **recency** — exponential decay with configurable half-life: `e^(-ln2 / half_life * age_days)` (default 30 days)

**Curation**: background task runs periodically to find near-duplicate memories (above similarity threshold) and suggests merges/prunes.

## Docker

```bash
docker build -t memory .
docker run -p 8000:8000 -e MEMORY__LISTEN_ADDR=0.0.0.0:8000 -v memory-data:/data memory
```

## Architecture

```
crates/
  memory-core/                — storage, embedding, scoring, curation
  memory-server/              — axum server, MCP + REST + auth + config
  memory-ui/                  — web interface (askama templates + HTMX)
  conversation-retrospective/ — extracts learnings from Claude Code logs + votes on recalled memories for reinforcement
```
