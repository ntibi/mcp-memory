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
    #[schemars(description = "The content to store as a memory")]
    content: String,
    #[schemars(description = "Optional tags to categorize the memory")]
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RecallMemoryParams {
    #[schemars(description = "Natural language query for semantic search")]
    query: String,
    #[schemars(description = "Maximum number of results to return (default: 5)")]
    n: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchByTagParams {
    #[schemars(description = "Tag to search for")]
    tag: String,
    #[schemars(description = "Maximum number of results to return (default: 10)")]
    n: Option<usize>,
}

#[tool_router]
impl MemoryMcp {
    #[tool(description = "Store a new memory with optional tags")]
    async fn store_memory(
        &self,
        Parameters(params): Parameters<StoreMemoryParams>,
    ) -> Result<String, String> {
        let input = CreateMemory {
            content: params.content,
            tags: params.tags.unwrap_or_default(),
        };
        let memory = self
            .store
            .create(input, self.embedder.as_ref())
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&memory).map_err(|e| e.to_string())
    }

    #[tool(description = "Recall memories using semantic search")]
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

    #[tool(description = "Search memories by exact tag match")]
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
            instructions: Some(
                "Memory storage and retrieval service with semantic search.".to_string(),
            ),
        }
    }
}
