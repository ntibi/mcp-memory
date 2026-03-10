use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct State {
    pub conversations: HashMap<String, ConversationState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationState {
    pub project: String,
    pub session_id: String,
    pub status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter_reason: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub learnings: Vec<Learning>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    HeuristicSkipped,
    NotInteresting,
    Interesting,
    Extracted,
    Approved,
    Stored,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Learning {
    pub content: String,
    pub tags: Vec<String>,
    pub category: String,
    #[serde(default)]
    pub approved: bool,
}

impl State {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self {
                conversations: HashMap::new(),
            });
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading state file: {}", path.display()))?;

        serde_json::from_str(&content)
            .with_context(|| format!("parsing state file: {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp_path = path.with_extension("tmp");

        let content = serde_json::to_string_pretty(self)
            .context("serializing state")?;

        std::fs::write(&tmp_path, &content)
            .with_context(|| format!("writing temp state file: {}", tmp_path.display()))?;

        std::fs::rename(&tmp_path, path)
            .with_context(|| format!("renaming {} to {}", tmp_path.display(), path.display()))?;

        Ok(())
    }

    pub fn is_processed(&self, session_id: &str) -> bool {
        self.conversations.contains_key(session_id)
    }
}
