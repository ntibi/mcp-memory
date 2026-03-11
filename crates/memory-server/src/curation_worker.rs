use std::collections::HashSet;
use std::sync::Arc;

use memory_core::curation::{self, CandidateGroup, CurationRun, CurationSettings};
pub use memory_core::curation::{new_progress_map, ProgressMap, ProgressStatus, RunProgress};
use memory_core::memory::MemoryStore;
use reqwest::Client;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

pub async fn execute_run(
    conn: tokio_rusqlite::Connection,
    store: Arc<MemoryStore>,
    settings: CurationSettings,
    progress_map: ProgressMap,
    cancel: CancellationToken,
) {
    let user_id = settings.user_id.clone();

    let mut run = match curation::create_run(&conn, &user_id).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(user_id = %user_id, error = %e, "failed to create curation run");
            return;
        }
    };

    let mut already_processed = load_previously_processed(&conn, &user_id).await;

    let groups = match curation::select_candidate_groups(
        &conn,
        &user_id,
        settings.similarity_threshold,
        &already_processed.iter().cloned().collect::<Vec<_>>(),
    )
    .await
    {
        Ok(g) => g,
        Err(e) => {
            tracing::error!(user_id = %user_id, run_id = %run.id, error = %e, "failed to select candidate groups");
            finalize_run(&conn, &mut run, "failed", Some(e.to_string())).await;
            progress_map.remove(&run.id);
            return;
        }
    };

    tracing::info!(
        user_id = %user_id,
        run_id = %run.id,
        groups = groups.len(),
        excluded = already_processed.len(),
        threshold = settings.similarity_threshold,
        "candidate group selection complete"
    );

    run.total_groups = groups.len() as i64;
    let _ = curation::update_run(&conn, &run).await;

    progress_map.insert(
        run.id.clone(),
        RunProgress {
            run_id: run.id.clone(),
            status: ProgressStatus::Running,
            total_groups: groups.len(),
            processed_groups: 0,
            current_group_label: String::new(),
            suggestions_created: 0,
            cost_usd: 0.0,
        },
    );

    let api_key = match &settings.api_key {
        Some(k) => k.clone(),
        None => {
            tracing::error!(user_id = %user_id, run_id = %run.id, "no api key configured");
            finalize_run(&conn, &mut run, "failed", Some("no api key configured".to_string())).await;
            progress_map.remove(&run.id);
            return;
        }
    };

    let http = Client::new();
    let mut llm_failures = 0u64;

    for group in &groups {
        if cancel.is_cancelled() {
            tracing::info!(user_id = %user_id, run_id = %run.id, "curation run cancelled");
            finalize_run(&conn, &mut run, "cancelled", None).await;
            progress_map.remove(&run.id);
            return;
        }

        if let Some(budget) = settings.budget_limit_usd {
            if run.cost_usd >= budget {
                tracing::info!(user_id = %user_id, run_id = %run.id, cost = run.cost_usd, budget, "budget limit reached");
                break;
            }
        }

        run.current_group_label = Some(group.label.clone());
        if let Some(mut entry) = progress_map.get_mut(&run.id) {
            entry.current_group_label = group.label.clone();
            entry.status = if cancel.is_cancelled() {
                ProgressStatus::Cancelling
            } else {
                ProgressStatus::Running
            };
        }

        let memories = match load_group_memories(&store, &user_id, group).await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(user_id = %user_id, run_id = %run.id, group_label = %group.label, error = %e, "failed to load memories for group, skipping");
                continue;
            }
        };

        if memories.is_empty() {
            continue;
        }

        let prompt = build_prompt(&memories);

        match call_llm(&http, &settings.provider, &api_key, &settings.model, &prompt).await {
            Ok((response, usage)) => {
                run.tokens_used += usage.total_tokens();
                run.cost_usd += usage.estimate_cost(&settings.provider, &settings.model);

                for suggestion in &response.suggestions {
                    let suggestion_json = serde_json::json!({
                        "action": suggestion.action,
                        "content": suggestion.content,
                        "tags": suggestion.tags,
                        "reasoning": suggestion.reasoning,
                    });

                    match curation::store_suggestion(
                        &conn,
                        &user_id,
                        &suggestion.action,
                        &suggestion.memory_ids,
                        &suggestion_json.to_string(),
                        "llm",
                    )
                    .await
                    {
                        Ok(_) => {
                            run.suggestions_created += 1;
                        }
                        Err(e) => {
                            tracing::warn!(run_id = %run.id, error = %e, "failed to store suggestion");
                        }
                    }
                }
            }
            Err(e) => {
                llm_failures += 1;
                tracing::warn!(user_id = %user_id, run_id = %run.id, provider = %settings.provider, model = %settings.model, group_label = %group.label, error = %e, "llm call failed for group, skipping");
            }
        }

        for id in &group.memory_ids {
            already_processed.insert(id.clone());
        }
        run.processed_memory_ids = already_processed.iter().cloned().collect();
        run.processed_groups += 1;

        let _ = curation::update_run(&conn, &run).await;

        if let Some(mut entry) = progress_map.get_mut(&run.id) {
            entry.processed_groups = run.processed_groups as usize;
            entry.suggestions_created = run.suggestions_created as usize;
            entry.cost_usd = run.cost_usd;
        }
    }

    let (status, error) = if llm_failures == groups.len() as u64 && !groups.is_empty() {
        ("failed", Some(format!("all {llm_failures} llm calls failed")))
    } else {
        ("completed", None)
    };
    finalize_run(&conn, &mut run, status, error).await;
    progress_map.remove(&run.id);
}

async fn load_previously_processed(
    conn: &tokio_rusqlite::Connection,
    user_id: &str,
) -> HashSet<String> {
    let mut processed = HashSet::new();

    if let Ok(runs) = curation::list_runs(conn, user_id, 20).await {
        for run in runs {
            if run.status == "completed" {
                processed.extend(run.processed_memory_ids);
            }
        }
    }

    if let Ok(pending) = curation::list_suggestions(conn, user_id, Some("pending")).await {
        for s in pending {
            processed.extend(s.memory_ids);
        }
    }

    processed
}

async fn load_group_memories(
    store: &MemoryStore,
    user_id: &str,
    group: &CandidateGroup,
) -> anyhow::Result<Vec<memory_core::memory::Memory>> {
    let mut memories = Vec::new();
    for id in &group.memory_ids {
        match store.get(user_id, id).await {
            Ok(m) => memories.push(m),
            Err(e) => {
                tracing::warn!(user_id = %user_id, memory_id = %id, error = %e, "failed to load memory, skipping");
            }
        }
    }
    Ok(memories)
}

const SYSTEM_PROMPT: &str = "\
You are a memory curation assistant. Analyze groups of similar memories and suggest how to consolidate them.

For each group, either:
- **merge**: combine multiple memories into one better version (preserving all useful information)
- **rewrite**: rewrite and consolidate memories, keeping the first memory as the primary and absorbing the rest
- return an empty suggestions array if memories are distinct enough to keep separate

Respond with a JSON object containing a \"suggestions\" array. Each suggestion has:
- action: \"merge\" or \"rewrite\"
- memory_ids: array of all memory IDs being merged
- content: the proposed consolidated content
- tags: proposed tags for the result
- reasoning: brief explanation";

fn build_prompt(memories: &[memory_core::memory::Memory]) -> String {
    let mut prompt = String::new();

    for m in memories {
        prompt.push_str(&format!("## Memory {}\n", m.id));
        prompt.push_str(&format!("Content: {}\n", m.content));
        prompt.push_str(&format!("Tags: {}\n", m.tags.join(", ")));
        prompt.push_str(&format!("Created: {}\n\n", m.created_at));
    }

    prompt.push_str("Analyze these memories and provide your suggestions.");

    prompt
}

fn build_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": { "type": "string", "enum": ["merge", "rewrite"] },
                        "memory_ids": { "type": "array", "items": { "type": "string" } },
                        "content": { "type": "string" },
                        "tags": { "type": "array", "items": { "type": "string" } },
                        "reasoning": { "type": "string" }
                    },
                    "required": ["action", "memory_ids", "content", "tags", "reasoning"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["suggestions"],
        "additionalProperties": false
    })
}

#[derive(Deserialize)]
struct SuggestionResponse {
    suggestions: Vec<LlmSuggestion>,
}

#[derive(Deserialize)]
struct LlmSuggestion {
    action: String,
    memory_ids: Vec<String>,
    content: String,
    tags: Vec<String>,
    reasoning: String,
}

#[derive(Debug, Default)]
struct LlmUsage {
    input_tokens: i64,
    output_tokens: i64,
}

impl LlmUsage {
    fn total_tokens(&self) -> i64 {
        self.input_tokens + self.output_tokens
    }

    fn estimate_cost(&self, provider: &str, model: &str) -> f64 {
        let (input_per_m, output_per_m) = match provider {
            "anthropic" => match model {
                m if m.contains("haiku") => (0.25, 1.25),
                m if m.contains("sonnet") => (3.0, 15.0),
                m if m.contains("opus") => (15.0, 75.0),
                _ => (3.0, 15.0),
            },
            "openai" => match model {
                m if m.contains("gpt-4o-mini") => (0.15, 0.60),
                m if m.contains("gpt-4o") => (2.50, 10.0),
                m if m.contains("gpt-4") => (30.0, 60.0),
                m if m.contains("o1") => (15.0, 60.0),
                m if m.contains("o3") => (10.0, 40.0),
                _ => (2.50, 10.0),
            },
            "gemini" => match model {
                m if m.contains("flash") => (0.075, 0.30),
                m if m.contains("pro") => (1.25, 5.0),
                _ => (0.075, 0.30),
            },
            _ => (1.0, 1.0),
        };
        (self.input_tokens as f64 * input_per_m + self.output_tokens as f64 * output_per_m) / 1_000_000.0
    }
}

async fn call_llm(
    http: &Client,
    provider: &str,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> anyhow::Result<(SuggestionResponse, LlmUsage)> {
    let (text, usage) = match provider {
        "anthropic" => call_anthropic(http, api_key, model, prompt).await?,
        "openai" => call_openai(http, api_key, model, prompt).await?,
        "gemini" => call_gemini(http, api_key, model, prompt).await?,
        other => anyhow::bail!("unsupported llm provider: {other}"),
    };

    let text = text.trim();
    let text = text.strip_prefix("```json").or_else(|| text.strip_prefix("```")).unwrap_or(text);
    let text = text.strip_suffix("```").unwrap_or(text);
    let text = text.trim();

    let response: SuggestionResponse = serde_json::from_str(text)
        .map_err(|e| anyhow::anyhow!("failed to parse llm response: {e}\nraw: {}", &text[..text.len().min(500)]))?;
    Ok((response, usage))
}

async fn call_anthropic(
    http: &Client,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> anyhow::Result<(String, LlmUsage)> {
    let schema = build_schema();
    let system = format!(
        "{SYSTEM_PROMPT}\n\nYou MUST respond with valid JSON matching this schema:\n{}",
        serde_json::to_string_pretty(&schema).unwrap()
    );

    let body = serde_json::json!({
        "model": model,
        "max_tokens": 16384,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    });

    let resp = http
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("anthropic api error ({}): {}", status, &text[..text.len().min(500)]);
    }

    let resp: serde_json::Value = serde_json::from_str(&text)?;
    let content = resp["content"]
        .as_array()
        .and_then(|blocks| blocks.iter().find(|b| b["type"] == "text"))
        .and_then(|b| b["text"].as_str())
        .ok_or_else(|| anyhow::anyhow!("no text in anthropic response: {}", &text[..text.len().min(500)]))?;

    let usage = LlmUsage {
        input_tokens: resp["usage"]["input_tokens"].as_i64().unwrap_or(0),
        output_tokens: resp["usage"]["output_tokens"].as_i64().unwrap_or(0),
    };

    Ok((content.to_string(), usage))
}

async fn call_openai(
    http: &Client,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> anyhow::Result<(String, LlmUsage)> {
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "curation",
                "strict": true,
                "schema": build_schema()
            }
        }
    });

    let resp = http
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("openai api error ({}): {}", status, &text[..text.len().min(500)]);
    }

    let resp: serde_json::Value = serde_json::from_str(&text)?;
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("no content in openai response: {}", &text[..text.len().min(500)]))?;

    let usage = LlmUsage {
        input_tokens: resp["usage"]["prompt_tokens"].as_i64().unwrap_or(0),
        output_tokens: resp["usage"]["completion_tokens"].as_i64().unwrap_or(0),
    };

    Ok((content.to_string(), usage))
}

async fn call_gemini(
    http: &Client,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> anyhow::Result<(String, LlmUsage)> {
    let full_prompt = format!("{SYSTEM_PROMPT}\n\n{prompt}");

    let body = serde_json::json!({
        "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": build_schema()
        }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );

    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("gemini api error ({}): {}", status, &text[..text.len().min(500)]);
    }

    let resp: serde_json::Value = serde_json::from_str(&text)?;
    let content = resp["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("no text in gemini response: {}", &text[..text.len().min(500)]))?;

    let usage = LlmUsage {
        input_tokens: resp["usageMetadata"]["promptTokenCount"].as_i64().unwrap_or(0),
        output_tokens: resp["usageMetadata"]["candidatesTokenCount"].as_i64().unwrap_or(0),
    };

    Ok((content.to_string(), usage))
}

async fn finalize_run(
    conn: &tokio_rusqlite::Connection,
    run: &mut CurationRun,
    status: &str,
    error: Option<String>,
) {
    run.status = status.to_string();
    run.completed_at = Some(chrono::Utc::now().to_rfc3339());
    run.error = error;
    if let Err(e) = curation::update_run(conn, run).await {
        tracing::error!(run_id = %run.id, error = %e, "failed to finalize curation run");
    }
}
