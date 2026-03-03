use std::sync::Arc;

use memory_core::{
    embed::Embedder,
    memory::{CreateMemory, MemoryStore},
    scoring::Scorer,
};
use rmcp::{
    ServerHandler,
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_handler, tool_router,
};
use serde::Deserialize;

#[derive(Clone)]
pub struct MemoryMcp {
    store: Arc<MemoryStore>,
    embedder: Arc<dyn Embedder>,
    scorer: Arc<Scorer>,
    tool_router: ToolRouter<Self>,
}

impl MemoryMcp {
    pub fn new(
        store: Arc<MemoryStore>,
        embedder: Arc<dyn Embedder>,
        scorer: Arc<Scorer>,
    ) -> Self {
        let tool_router = Self::tool_router();
        Self {
            store,
            embedder,
            scorer,
            tool_router,
        }
    }
}

type ToolRouter<S> = rmcp::handler::server::router::tool::ToolRouter<S>;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct StoreMemoryParams {
    #[schemars(description = "The content to store as a memory. Can be a fact, note, preference, decision, code pattern, or any information worth remembering across conversations.")]
    content: String,
    #[schemars(description = "Tags to categorize the memory across multiple dimensions. Include ALL that apply: project name, language (rust, python), domain (networking, auth), activity (debugging, deployment, testing), tool (docker, git, postgres), subject (user-preference, workflow, team-convention, machine-setup), knowledge type (gotcha, pattern, decision, workaround, preference), source context (code-review, debugging-session, documentation), tool/MCP context (claude-code, slack, github, terminal), scope (universal, project-specific, machine-specific). Prefer lowercase, singular, hyphenated. More tags is always better — they cost nothing and improve retrieval. Aim for at least 3 tags per memory.")]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RecallMemoryParams {
    #[schemars(description = "Natural language query describing what you're looking for. Be specific — 'rust async error handling patterns' works better than 'rust'.")]
    query: String,
    #[schemars(description = "Maximum number of results to return (default: 5). Higher values return more results but may include less relevant matches.")]
    n: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct UpdateMemoryParams {
    #[schemars(description = "The ID of the memory to update.")]
    id: String,
    #[schemars(description = "The new content for the memory. The embedding will be recomputed.")]
    content: String,
    #[schemars(description = "Optional new tags to replace the existing tags. If not provided, tags are left unchanged.")]
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct DeleteMemoryParams {
    #[schemars(description = "The ID of the memory to delete.")]
    id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchByTagParams {
    #[schemars(description = "Tags to filter by (case-sensitive, AND logic). Returns memories that have ALL specified tags.")]
    tags: Vec<String>,
    #[schemars(description = "Maximum number of results to return (default: 10).")]
    n: Option<usize>,
}

#[tool_router]
impl MemoryMcp {
    #[tool(description = "Store a new memory with tags. Tag generously across every relevant dimension — language, domain, tool, activity, project. A memory about 'fixing a postgres connection pool timeout in rust' should get at least [project-name, rust, postgres, connection-pooling, debugging, performance]. Err on the side of too many tags.", annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false))]
    async fn store_memory(
        &self,
        Parameters(params): Parameters<StoreMemoryParams>,
    ) -> Result<String, String> {
        let input = CreateMemory {
            content: params.content,
            tags: params.tags,
        };
        let memory = self
            .store
            .create(input, self.embedder.as_ref())
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&memory).map_err(|e| e.to_string())
    }

    #[tool(description = "Recall memories using semantic search. Finds stored memories most relevant to the query using vector similarity, ranked by relevance, confidence, and recency. Use this for open-ended lookups where you don't know the exact category — describe what you're looking for in natural language.", annotations(read_only_hint = true, destructive_hint = false, open_world_hint = false))]
    async fn recall_memory(
        &self,
        Parameters(params): Parameters<RecallMemoryParams>,
    ) -> Result<String, String> {
        let n = params.n.unwrap_or(20);
        let results = self
            .store
            .recall(
                &params.query,
                n,
                self.embedder.as_ref(),
                self.scorer.as_ref(),
            )
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&results).map_err(|e| e.to_string())
    }

    #[tool(description = "Update an existing memory's content and optionally its tags. The embedding is recomputed from the new content. Use this when a memory is outdated or partially wrong but the core topic is the same.", annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false))]
    async fn update_memory(
        &self,
        Parameters(params): Parameters<UpdateMemoryParams>,
    ) -> Result<String, String> {
        let memory = self
            .store
            .update(&params.id, &params.content, self.embedder.as_ref())
            .await
            .map_err(|e| e.to_string())?;
        if let Some(tags) = params.tags {
            self.store
                .set_tags(&params.id, tags)
                .await
                .map_err(|e| e.to_string())?;
        }
        serde_json::to_string(&memory).map_err(|e| e.to_string())
    }

    #[tool(description = "Delete a memory by ID. Use this when a memory is completely wrong or superseded. This is irreversible.", annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = true, open_world_hint = false))]
    async fn delete_memory(
        &self,
        Parameters(params): Parameters<DeleteMemoryParams>,
    ) -> Result<String, String> {
        self.store
            .delete(&params.id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(format!("deleted memory {}", params.id))
    }

    #[tool(description = "Search memories by exact tag match. Returns all memories tagged with the specified tag, ordered by creation time. Use this when you know the category of information you're looking for rather than searching by content.", annotations(read_only_hint = true, destructive_hint = false, open_world_hint = false))]
    async fn search_by_tags(
        &self,
        Parameters(params): Parameters<SearchByTagParams>,
    ) -> Result<String, String> {
        let n = params.n.unwrap_or(20);
        let results = self
            .store
            .search_by_tags(&params.tags, n)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&results).map_err(|e| e.to_string())
    }
}

#[tool_handler]
impl ServerHandler for MemoryMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "memory-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                ..Default::default()
            },
            instructions: Some(concat!(
                "Persistent semantic memory for LLM agents. ",
                "MANDATORY on every session start: call `search_by_tags` with the current project name prefixed as `project:<name>` (e.g. `project:memory`), then `recall_memory` with a query describing the user's first message. Do this BEFORE any other action. ",
                "Store aggressively: after solving any non-trivial problem, learning a preference, or making an architectural decision, call `store_memory` immediately. Do not wait. Use `update_memory` to fix outdated memories, `delete_memory` to remove wrong or superseded ones. ",
                "Recall often: when encountering a new subtask, bug, or decision point mid-session, call `recall_memory` again with a relevant query. ",
                "Tag every memory with `project:<name>` (e.g. `project:memory`) plus all relevant categories (language, domain, tool, activity, concept, subject, knowledge-type, scope). Aim for at least 3 tags per memory. More tags is always better than fewer. ",
                "When a request is ambiguous, unclear, or seems to lack context: call `recall_memory` with the confusing parts before asking the user for clarification. Prior memories often contain the missing context. ",
                "Prefer `recall_memory` for open-ended lookups, `search_by_tags` for known categories. Multiple tags use AND logic — only memories matching ALL specified tags are returned. ",
            ).to_string()),
        }
    }
}
