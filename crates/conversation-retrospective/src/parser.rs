use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct LogEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub message: Option<Message>,
    pub timestamp: Option<String>,
    #[serde(rename = "sessionId")]
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: Option<String>,
    pub content: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub role: String,
    pub text: String,
}

#[derive(Debug)]
pub struct Conversation {
    pub session_id: String,
    pub project: String,
    pub messages: Vec<ConversationMessage>,
    pub timestamp: Option<String>,
}

pub fn discover_projects(base: &Path) -> Result<Vec<(String, Vec<PathBuf>)>> {
    let mut projects: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();

    let entries = std::fs::read_dir(base)
        .with_context(|| format!("reading base directory: {}", base.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let project_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        let sub_entries = match std::fs::read_dir(&path) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let mut conversations: Vec<PathBuf> = Vec::new();
        for sub in sub_entries {
            let sub = match sub {
                Ok(s) => s,
                Err(_) => continue,
            };
            let sub_path = sub.path();
            if sub_path.extension().and_then(|e| e.to_str()) == Some("jsonl") && sub_path.is_file() {
                conversations.push(sub_path);
            }
        }

        if conversations.is_empty() {
            continue;
        }

        conversations.sort_by(|a, b| {
            let mtime_a = std::fs::metadata(a).and_then(|m| m.modified()).ok();
            let mtime_b = std::fs::metadata(b).and_then(|m| m.modified()).ok();
            mtime_a.cmp(&mtime_b)
        });

        projects.insert(project_name, conversations);
    }

    let mut result: Vec<(String, Vec<PathBuf>)> = projects.into_iter().collect();
    result.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    Ok(result)
}

pub fn parse_conversation(path: &Path, project: &str) -> Result<Conversation> {
    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .with_context(|| format!("extracting session id from: {}", path.display()))?;

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading conversation file: {}", path.display()))?;

    let mut messages = Vec::new();
    let mut timestamp = None;

    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let entry: LogEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(file = %path.display(), line = i + 1, error = %err, "skipping malformed jsonl line");
                continue;
            }
        };

        if timestamp.is_none() {
            timestamp = entry.timestamp.clone();
        }

        let role = match entry.entry_type.as_str() {
            "user" | "assistant" => entry.entry_type.as_str(),
            _ => continue,
        };

        let text = entry
            .message
            .as_ref()
            .and_then(|m| m.content.as_ref())
            .map(extract_text)
            .unwrap_or_default();

        if text.is_empty() {
            continue;
        }

        messages.push(ConversationMessage {
            role: role.to_string(),
            text,
        });
    }

    Ok(Conversation {
        session_id,
        project: project.to_string(),
        messages,
        timestamp,
    })
}

fn extract_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|item| {
                let obj = item.as_object()?;
                if obj.get("type")?.as_str()? != "text" {
                    return None;
                }
                obj.get("text")?.as_str().map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
