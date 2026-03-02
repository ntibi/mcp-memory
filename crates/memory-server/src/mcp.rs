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
    #[schemars(description = "Tags to categorize the memory across multiple dimensions. Include ALL that apply: language (rust, python), domain (networking, auth), activity (debugging, deployment, testing), tool (docker, git, postgres), project name, and any other relevant category. Prefer lowercase, singular, hyphenated (e.g. error-handling). More tags is always better — they cost nothing and improve retrieval. Aim for at least 3 tags per memory.")]
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
struct SearchByTagParams {
    #[schemars(description = "Exact tag to filter by (case-sensitive). Returns all memories that have this tag.")]
    tag: String,
    #[schemars(description = "Maximum number of results to return (default: 10).")]
    n: Option<usize>,
}

#[tool_router]
impl MemoryMcp {
    #[tool(description = "Store a new memory with tags. Tag generously across every relevant dimension — language, domain, tool, activity, project. A memory about 'fixing a postgres connection pool timeout in rust' should get at least [project-name, rust, postgres, connection-pooling, debugging, performance]. Err on the side of too many tags.")]
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

    #[tool(description = "Recall memories using semantic search. Finds stored memories most relevant to the query using vector similarity, ranked by relevance, confidence, and recency. Use this for open-ended lookups where you don't know the exact category — describe what you're looking for in natural language.")]
    async fn recall_memory(
        &self,
        Parameters(params): Parameters<RecallMemoryParams>,
    ) -> Result<String, String> {
        let n = params.n.unwrap_or(5);
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

    #[tool(description = "Search memories by exact tag match. Returns all memories tagged with the specified tag, ordered by creation time. Use this when you know the category of information you're looking for rather than searching by content.")]
    async fn search_by_tag(
        &self,
        Parameters(params): Parameters<SearchByTagParams>,
    ) -> Result<String, String> {
        let n = params.n.unwrap_or(10);
        let results = self
            .store
            .search_by_tag(&params.tag, n)
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
                "On session start: call `search_by_tag` with the current project name, then `recall_memory` with a query relevant to the user's request. ",
                "After solving a non-trivial problem or learning a user preference: call `store_memory` with the finding. ",
                "Tag every memory with the project name plus all relevant categories (language, domain, tool, activity, concept). Aim for at least 3 tags per memory. More tags is always better than fewer. ",
                "Prefer `recall_memory` for open-ended lookups, `search_by_tag` for known categories. ",
            ).to_string()),
        }
    }
}
