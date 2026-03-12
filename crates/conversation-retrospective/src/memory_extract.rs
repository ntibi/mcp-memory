use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct RecalledMemory {
    pub id: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct TextSegment {
    pub text: String,
    pub memory_ids: Vec<String>,
}

#[derive(Debug)]
pub struct ConversationWithMemories {
    pub segments: Vec<TextSegment>,
    pub memories: HashMap<String, RecalledMemory>,
}

#[derive(Debug)]
pub struct Chunk {
    pub text: String,
    pub memory_ids: Vec<String>,
}

const CHUNK_SIZE: usize = 80_000;
const OVERLAP_SEGMENTS: usize = 10;

impl ConversationWithMemories {
    pub fn into_chunks(self) -> Vec<Chunk> {
        let total_len: usize = self.segments.iter().map(|s| s.text.len()).sum();

        if total_len <= CHUNK_SIZE {
            let text = self
                .segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let memory_ids: Vec<String> = self.memories.into_keys().collect();
            return vec![Chunk { text, memory_ids }];
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut chunk_start = 0;

        while chunk_start < self.segments.len() {
            let mut chunk_len = 0;
            let mut chunk_end = chunk_start;

            while chunk_end < self.segments.len() {
                let seg_len = self.segments[chunk_end].text.len() + 2;
                if chunk_len + seg_len > CHUNK_SIZE && chunk_end > chunk_start {
                    break;
                }
                chunk_len += seg_len;
                chunk_end += 1;
            }

            let overlap_start = if chunk_start > 0 {
                chunk_start.saturating_sub(OVERLAP_SEGMENTS)
            } else {
                chunk_start
            };

            let text = self.segments[overlap_start..chunk_end]
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");

            let owned_memory_ids: HashSet<&str> = self.segments[chunk_start..chunk_end]
                .iter()
                .flat_map(|s| s.memory_ids.iter().map(|id| id.as_str()))
                .collect();

            let memory_ids: Vec<String> = owned_memory_ids
                .into_iter()
                .filter(|id| self.memories.contains_key(*id))
                .map(|id| id.to_string())
                .collect();

            if !memory_ids.is_empty() {
                chunks.push(Chunk { text, memory_ids });
            }

            chunk_start = chunk_end;
        }

        chunks
    }
}

pub fn extract_memories_from_conversation(
    path: &Path,
) -> Result<Option<ConversationWithMemories>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading conversation file: {}", path.display()))?;

    let mut memory_tool_use_ids: HashMap<String, String> = HashMap::new();
    let mut segments: Vec<TextSegment> = Vec::new();
    let mut memories: HashMap<String, RecalledMemory> = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let entry: Value = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let msg = match entry.get("message") {
            Some(m) => m,
            None => continue,
        };

        let role = entry
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let content_val = match msg.get("content") {
            Some(c) => c,
            None => continue,
        };

        match content_val {
            Value::String(s) => {
                if role == "user" || role == "assistant" {
                    let label = if role == "user" { "User" } else { "Assistant" };
                    segments.push(TextSegment {
                        text: format!("{label}: {s}"),
                        memory_ids: vec![],
                    });
                }
            }
            Value::Array(arr) => {
                for block in arr {
                    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");

                    match block_type {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                                if role == "user" || role == "assistant" {
                                    let label =
                                        if role == "user" { "User" } else { "Assistant" };
                                    segments.push(TextSegment {
                                        text: format!("{label}: {text}"),
                                        memory_ids: vec![],
                                    });
                                }
                            }
                        }
                        "thinking" => {
                            if let Some(thinking) = block.get("thinking").and_then(|v| v.as_str())
                            {
                                if !thinking.is_empty() {
                                    segments.push(TextSegment {
                                        text: format!("[thinking]: {thinking}"),
                                        memory_ids: vec![],
                                    });
                                }
                            }
                        }
                        "tool_use" => {
                            let name =
                                block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            if is_memory_tool(name) {
                                let tool_id = block
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let short_name =
                                    name.strip_prefix("mcp__memory__").unwrap_or(name);
                                let input_summary =
                                    summarize_tool_input(short_name, &block["input"]);
                                segments.push(TextSegment {
                                    text: format!(
                                        "[memory tool: {short_name}] {input_summary}"
                                    ),
                                    memory_ids: vec![],
                                });
                                if !tool_id.is_empty() {
                                    memory_tool_use_ids
                                        .insert(tool_id, short_name.to_string());
                                }
                            }
                        }
                        "tool_result" => {
                            let tool_use_id = block
                                .get("tool_use_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");

                            if let Some(tool_name) = memory_tool_use_ids.get(tool_use_id) {
                                let result_text = extract_tool_result_text(block);
                                let mut result_memories: HashMap<String, RecalledMemory> =
                                    HashMap::new();
                                extract_memories_from_result(
                                    &result_text,
                                    &mut result_memories,
                                );

                                let ids: Vec<String> =
                                    result_memories.keys().cloned().collect();

                                if result_memories.is_empty() {
                                    segments.push(TextSegment {
                                        text: format!(
                                            "[{tool_name} result]: (no memories)"
                                        ),
                                        memory_ids: vec![],
                                    });
                                } else {
                                    let summary: String = result_memories
                                        .values()
                                        .map(|m| {
                                            format!("  - [{}] {}", m.id, m.content)
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n");
                                    segments.push(TextSegment {
                                        text: format!(
                                            "[{tool_name} result]: {} memories returned\n{summary}",
                                            result_memories.len()
                                        ),
                                        memory_ids: ids,
                                    });
                                }

                                memories.extend(result_memories);
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    if memories.is_empty() {
        return Ok(None);
    }

    Ok(Some(ConversationWithMemories {
        segments,
        memories,
    }))
}

fn is_memory_tool(name: &str) -> bool {
    name.starts_with("mcp__memory__")
}

fn summarize_tool_input(tool_name: &str, input: &Value) -> String {
    match tool_name {
        "recall_memory" => {
            let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("?");
            let n = input.get("n").and_then(|v| v.as_u64()).unwrap_or(0);
            if n > 0 {
                format!("query=\"{query}\" n={n}")
            } else {
                format!("query=\"{query}\"")
            }
        }
        "search_by_tags" | "search_by_tag" => {
            let tags = input
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .or_else(|| input.get("tag").and_then(|v| v.as_str()).map(|s| s.to_string()))
                .unwrap_or_default();
            format!("tags=[{tags}]")
        }
        "session_start" => {
            let task = input.get("task").and_then(|v| v.as_str()).unwrap_or("?");
            format!("task=\"{task}\"")
        }
        "store_memory" => {
            let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("?");
            let truncated: String = content.chars().take(80).collect();
            if truncated.len() < content.len() {
                format!("content=\"{truncated}...\"")
            } else {
                format!("content=\"{truncated}\"")
            }
        }
        "update_memory" => {
            let id = input.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            format!("id={id}")
        }
        "delete_memory" => {
            let id = input.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            format!("id={id}")
        }
        _ => format!("{}", input),
    }
}

fn extract_tool_result_text(block: &Value) -> String {
    match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|item| {
                if item.get("type")?.as_str()? == "text" {
                    item.get("text")?.as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn extract_memories_from_result(text: &str, memories: &mut HashMap<String, RecalledMemory>) {
    let parsed: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return,
    };

    fn collect(val: &Value, memories: &mut HashMap<String, RecalledMemory>) {
        if let Some(mem) = extract_single_memory(val) {
            memories.insert(mem.id.clone(), mem);
            return;
        }
        match val {
            Value::Array(arr) => {
                for item in arr {
                    collect(item, memories);
                }
            }
            Value::Object(map) => {
                for v in map.values() {
                    collect(v, memories);
                }
            }
            _ => {}
        }
    }

    collect(&parsed, memories);
}

fn extract_single_memory(val: &Value) -> Option<RecalledMemory> {
    let memory_obj = val.get("memory").unwrap_or(val);

    let id = memory_obj.get("id")?.as_str()?.to_string();
    let content = memory_obj.get("content")?.as_str()?.to_string();

    if id.is_empty() || content.is_empty() {
        return None;
    }

    Some(RecalledMemory { id, content })
}
