use std::sync::Arc;

use memory_core::{
    embed::Embedder,
    memory::{CreateMemory, MemoryStore},
    scoring::Scorer,
};
use memory_core::users::AuthContext;
use rmcp::{
    ServerHandler,
    handler::server::common::Extension,
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Deserializer, Serialize};
use tracing::info;

fn string_or_seq<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrSeq {
        Seq(Vec<String>),
        Str(String),
    }

    match StringOrSeq::deserialize(deserializer)? {
        StringOrSeq::Seq(v) => Ok(v),
        StringOrSeq::Str(s) => serde_json::from_str(&s).map_err(serde::de::Error::custom),
    }
}

fn option_string_or_seq<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OptStringOrSeq {
        Seq(Vec<String>),
        Str(String),
        Null,
    }

    match Option::<OptStringOrSeq>::deserialize(deserializer)? {
        None => Ok(None),
        Some(OptStringOrSeq::Seq(v)) => Ok(Some(v)),
        Some(OptStringOrSeq::Str(s)) => {
            serde_json::from_str(&s).map(Some).map_err(serde::de::Error::custom)
        }
        Some(OptStringOrSeq::Null) => Ok(None),
    }
}

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

fn user_id_from_parts(parts: &axum::http::request::Parts) -> Result<String, String> {
    parts
        .extensions
        .get::<AuthContext>()
        .map(|a| a.user_id.clone())
        .ok_or_else(|| "missing auth context".to_string())
}

type ToolRouter<S> = rmcp::handler::server::router::tool::ToolRouter<S>;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct StoreMemoryParams {
    #[schemars(description = "The content to store as a memory. Can be a fact, note, preference, decision, code pattern, or any information worth remembering across conversations.")]
    content: String,
    #[serde(deserialize_with = "string_or_seq")]
    #[schemars(description = "Tags to categorize the memory across multiple dimensions. Include ALL that apply: project name (project:<name>), language (rust, python), domain (networking, auth), activity (debugging, deployment, testing), tool (docker, git, postgres), subject (user-preference, workflow, team-convention, machine-setup), knowledge type (gotcha, pattern, decision, workaround, preference), source context (code-review, debugging-session, documentation), tool/MCP context (claude-code, slack, github, terminal). Prefer lowercase, singular, hyphenated. More tags is always better — they cost nothing and improve retrieval. Aim for at least 3 tags per memory. IMPORTANT: only use `universal` for memories that apply across ALL projects (user preferences, workflow conventions, machine setup). Do NOT use `universal` for project-specific gotchas, tool-specific debugging, or tech-stack-specific knowledge.")]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RecallMemoryParams {
    #[schemars(description = "Natural language query describing what you're looking for. Be specific — 'rust async error handling patterns' works better than 'rust'.")]
    query: String,
    #[schemars(description = "Maximum number of results to return (default: 20). Higher values return more results but may include less relevant matches.")]
    n: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SessionStartParams {
    #[serde(deserialize_with = "string_or_seq")]
    #[schemars(description = "Tags to search for (e.g. [\"project:memory\"]). Also automatically includes memories tagged \"universal\".")]
    tags: Vec<String>,
    #[schemars(description = "Natural language description of the current task or user's first message, used for semantic recall.")]
    task: String,
    #[schemars(description = "Maximum total memories to return (default: 40).")]
    n: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct UpdateMemoryParams {
    #[schemars(description = "The ID of the memory to update.")]
    id: String,
    #[schemars(description = "The new content for the memory. The embedding will be recomputed.")]
    content: String,
    #[serde(default, deserialize_with = "option_string_or_seq")]
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
    #[serde(deserialize_with = "string_or_seq")]
    #[schemars(description = "Tags to filter by (case-sensitive, AND logic). Returns memories that have ALL specified tags.")]
    tags: Vec<String>,
    #[schemars(description = "Maximum number of results to return (default: 20).")]
    n: Option<usize>,
}

#[tool_router]
impl MemoryMcp {
    #[tool(description = "Store a new memory with tags. Tag generously across every relevant dimension — language, domain, tool, activity, project. A memory about 'fixing a postgres connection pool timeout in rust' should get at least [project-name, rust, postgres, connection-pooling, debugging, performance]. Err on the side of too many tags. Returns the stored memory plus related existing memories for context.", annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false))]
    async fn store_memory(
        &self,
        Extension(parts): Extension<axum::http::request::Parts>,
        Parameters(params): Parameters<StoreMemoryParams>,
    ) -> Result<String, String> {
        let user_id = user_id_from_parts(&parts)?;
        let input = CreateMemory {
            content: params.content.clone(),
            tags: params.tags,
        };
        let memory = self
            .store
            .create(&user_id, input, self.embedder.as_ref())
            .await
            .map_err(|e| e.to_string())?;
        let related = self
            .store
            .recall(&user_id, &params.content, 5, self.embedder.as_ref(), self.scorer.as_ref())
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|s| s.memory.id != memory.id)
            .collect::<Vec<_>>();
        #[derive(Serialize)]
        struct StoreResult {
            stored: memory_core::memory::Memory,
            related: Vec<memory_core::memory::ScoredMemory>,
        }
        let result = StoreResult { stored: memory, related };
        let response = serde_json::to_string(&result).map_err(|e| e.to_string())?;
        let related_ids: Vec<&str> = result.related.iter().map(|s| s.memory.id.as_str()).collect();
        info!(
            user_id = %user_id,
            tool = "store_memory",
            stored_id = %result.stored.id,
            ?related_ids,
            response_bytes = response.len(),
            "mcp tool call"
        );
        Ok(response)
    }

    #[tool(description = "Load context at the start of a session. Combines tag-based search (project + universal) with semantic recall of the current task, returning a deduplicated set of memories. Call this once at session start instead of separate search_by_tags + recall_memory calls.", annotations(read_only_hint = true, destructive_hint = false, open_world_hint = false))]
    async fn session_start(
        &self,
        Extension(parts): Extension<axum::http::request::Parts>,
        Parameters(params): Parameters<SessionStartParams>,
    ) -> Result<String, String> {
        let user_id = user_id_from_parts(&parts)?;
        let n = params.n.unwrap_or(40);
        let half = n / 2;

        let by_tags = if params.tags.is_empty() {
            vec![]
        } else {
            self.store
                .search_by_tags(&user_id, &params.tags, half)
                .await
                .unwrap_or_default()
        };

        let universal = self
            .store
            .search_by_tags(&user_id, &["universal".into()], half / 4)
            .await
            .unwrap_or_default();

        let semantic = self
            .store
            .recall(&user_id, &params.task, half, self.embedder.as_ref(), self.scorer.as_ref())
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.memory)
            .collect::<Vec<_>>();

        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for m in by_tags.into_iter().chain(universal).chain(semantic) {
            if seen.insert(m.id.clone()) {
                result.push(m);
            }
        }
        result.truncate(n);

        let response = serde_json::to_string(&result).map_err(|e| e.to_string())?;
        let memory_ids: Vec<&str> = result.iter().map(|m| m.id.as_str()).collect();
        info!(
            user_id = %user_id,
            tool = "session_start",
            ?memory_ids,
            memories_returned = result.len(),
            response_bytes = response.len(),
            "mcp tool call"
        );
        Ok(response)
    }

    #[tool(description = "Recall memories using semantic search. Finds stored memories most relevant to the query using vector similarity, ranked by relevance, confidence, and recency. Use this for open-ended lookups where you don't know the exact category — describe what you're looking for in natural language.", annotations(read_only_hint = true, destructive_hint = false, open_world_hint = false))]
    async fn recall_memory(
        &self,
        Extension(parts): Extension<axum::http::request::Parts>,
        Parameters(params): Parameters<RecallMemoryParams>,
    ) -> Result<String, String> {
        let user_id = user_id_from_parts(&parts)?;
        let n = params.n.unwrap_or(20);
        let results = self
            .store
            .recall(
                &user_id,
                &params.query,
                n,
                self.embedder.as_ref(),
                self.scorer.as_ref(),
            )
            .await
            .map_err(|e| e.to_string())?;
        let response = serde_json::to_string(&results).map_err(|e| e.to_string())?;
        let memory_ids: Vec<&str> = results.iter().map(|s| s.memory.id.as_str()).collect();
        info!(
            user_id = %user_id,
            tool = "recall_memory",
            ?memory_ids,
            memories_returned = results.len(),
            response_bytes = response.len(),
            "mcp tool call"
        );
        Ok(response)
    }

    #[tool(description = "Update an existing memory's content and optionally its tags. The embedding is recomputed from the new content. Use this when a memory is outdated or partially wrong but the core topic is the same.", annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false))]
    async fn update_memory(
        &self,
        Extension(parts): Extension<axum::http::request::Parts>,
        Parameters(params): Parameters<UpdateMemoryParams>,
    ) -> Result<String, String> {
        let user_id = user_id_from_parts(&parts)?;
        let memory = self
            .store
            .update(&user_id, &params.id, &params.content, self.embedder.as_ref())
            .await
            .map_err(|e| e.to_string())?;
        if let Some(tags) = params.tags {
            self.store
                .set_tags(&user_id, &params.id, tags)
                .await
                .map_err(|e| e.to_string())?;
        }
        let response = serde_json::to_string(&memory).map_err(|e| e.to_string())?;
        info!(
            user_id = %user_id,
            tool = "update_memory",
            updated_id = %params.id,
            response_bytes = response.len(),
            "mcp tool call"
        );
        Ok(response)
    }

    #[tool(description = "Delete a memory by ID. Use this when a memory is completely wrong or superseded. This is irreversible.", annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = true, open_world_hint = false))]
    async fn delete_memory(
        &self,
        Extension(parts): Extension<axum::http::request::Parts>,
        Parameters(params): Parameters<DeleteMemoryParams>,
    ) -> Result<String, String> {
        let user_id = user_id_from_parts(&parts)?;
        self.store
            .delete(&user_id, &params.id)
            .await
            .map_err(|e| e.to_string())?;
        let response = format!("deleted memory {}", params.id);
        info!(
            user_id = %user_id,
            tool = "delete_memory",
            deleted_id = %params.id,
            response_bytes = response.len(),
            "mcp tool call"
        );
        Ok(response)
    }

    #[tool(description = "Search memories by exact tag match. Returns all memories tagged with the specified tag, ordered by creation time. Use this when you know the category of information you're looking for rather than searching by content.", annotations(read_only_hint = true, destructive_hint = false, open_world_hint = false))]
    async fn search_by_tags(
        &self,
        Extension(parts): Extension<axum::http::request::Parts>,
        Parameters(params): Parameters<SearchByTagParams>,
    ) -> Result<String, String> {
        let user_id = user_id_from_parts(&parts)?;
        let n = params.n.unwrap_or(20);
        let results = self
            .store
            .search_by_tags(&user_id, &params.tags, n)
            .await
            .map_err(|e| e.to_string())?;
        let response = serde_json::to_string(&results).map_err(|e| e.to_string())?;
        let memory_ids: Vec<&str> = results.iter().map(|m| m.id.as_str()).collect();
        info!(
            user_id = %user_id,
            tool = "search_by_tags",
            ?memory_ids,
            memories_returned = results.len(),
            response_bytes = response.len(),
            "mcp tool call"
        );
        Ok(response)
    }
}

#[tool_handler]
impl ServerHandler for MemoryMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("memory-server", env!("CARGO_PKG_VERSION")))
            .with_instructions(concat!(
                "Persistent semantic memory for LLM agents. ",
                "\n\n",
                "## CORE RULES\n\n",
                "1. MANDATORY: call `session_start` at the beginning of every session, BEFORE any other action.\n",
                "2. RECALL AGGRESSIVELY: call `recall_memory` before EVERY implementation step, debug attempt, or module switch. A wasted recall costs nothing. A missed recall repeats past mistakes.\n",
                "3. STORE IMMEDIATELY: after solving any non-trivial problem, learning a preference, or making an architectural decision, call `store_memory` right away. `store_memory` returns related existing memories — read them. Use `update_memory` to fix outdated memories, `delete_memory` to remove wrong or superseded ones.\n",
                "\n",
                "## WHEN TO RECALL\n\n",
                "Call `recall_memory` or `search_by_tags` in ALL of these situations:\n",
                "- Before implementing any feature or fix — recall the module/technology you're about to touch\n",
                "- Before debugging — recall the error message, module, or symptom\n",
                "- Before choosing a library, tool, or pattern — past sessions may have evaluated it\n",
                "- When reading unfamiliar code — recall the file path or module name\n",
                "- When the user references something from a past session — recall it, don't guess\n",
                "- When context-switching to a different part of the codebase\n",
                "- Before answering architecture or design questions — prior decisions may exist\n",
                "- When encountering an error or unexpected behavior — recall the error text\n",
                "- Before suggesting a workaround — a known fix may already exist\n",
                "- When unsure about a user preference or convention — recall rather than ask\n",
                "- Before answering any user question — you may have context from prior sessions\n",
                "- Before asking the user a clarifying question — the answer may already exist in memory\n",
                "\n",
                "If you are unsure whether to recall, recall. Default to recalling.\n",
                "\n",
                "## TAGGING\n\n",
                "Tag every memory with `project:<name>` plus all relevant categories (language, domain, tool, activity, concept, subject, knowledge-type, scope). Aim for at least 3 tags per memory. More tags is always better.\n",
                "The `universal` tag is ONLY for memories that genuinely apply across ALL projects: user preferences, machine setup, cross-cutting workflow conventions. ",
                "Do NOT tag as `universal`: project-specific gotchas, tool-specific debugging, library quirks, one-off fixes, or tech-stack-specific knowledge.\n",
                "`session_start` automatically includes `universal` memories, so over-tagging with `universal` pollutes every session.\n",
                "\n",
                "## LOOKUP STRATEGY\n\n",
                "Prefer `recall_memory` for open-ended lookups, `search_by_tags` for known categories. Multiple tags use AND logic. ",
                "When a request is ambiguous or seems to lack context: call `recall_memory` before asking the user for clarification — prior memories often contain the missing context.\n",
            ))
    }
}
