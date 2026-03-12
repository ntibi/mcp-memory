use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::claude::run_claude;
use crate::memory_extract::{self, ConversationWithMemories};
use crate::parser;

#[derive(Debug, Serialize, Deserialize)]
pub struct VoteState {
    pub conversations: HashMap<String, ConversationVoteState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationVoteState {
    pub project: String,
    pub session_id: String,
    pub status: VoteStatus,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub votes: Vec<MemoryVote>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VoteStatus {
    NoMemories,
    Evaluated,
    Voted,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryVote {
    pub memory_id: String,
    pub vote: String,
    pub reason: String,
    #[serde(default)]
    pub submitted: bool,
}

impl VoteState {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self {
                conversations: HashMap::new(),
            });
        }
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading vote state: {}", path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("parsing vote state: {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("tmp");
        let content = serde_json::to_string_pretty(self).context("serializing vote state")?;
        std::fs::write(&tmp, &content)
            .with_context(|| format!("writing temp vote state: {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
    }
}

#[derive(Debug, Deserialize)]
struct EvalResponse {
    evaluations: Vec<MemoryEval>,
}

#[derive(Debug, Deserialize)]
struct MemoryEval {
    memory_id: String,
    vote: String,
    reason: String,
}

fn build_eval_prompt(chunk_text: &str, memories_list: &str) -> String {
    format!(
        "You are evaluating whether memories recalled during a Claude Code conversation were useful or hurtful.\n\
        \n\
        A memory is \"helpful\" if it:\n\
        - Provided relevant context that improved the assistant's response\n\
        - Contained information that was directly applicable to the task\n\
        - Helped avoid mistakes or saved time\n\
        - Correctly informed a decision or approach\n\
        \n\
        A memory is \"harmful\" if it:\n\
        - Was completely irrelevant to the conversation\n\
        - Contained outdated or wrong information that could mislead\n\
        - Added noise without value\n\
        - Led to incorrect assumptions or approaches\n\
        \n\
        If a memory is borderline or neutral, vote \"helpful\" — only vote \"harmful\" when you're confident it added no value or was actively detrimental.\n\
        \n\
        <conversation>\n{chunk_text}\n</conversation>\n\
        \n\
        <memories_to_evaluate>\n{memories_list}\n</memories_to_evaluate>\n\
        \n\
        For each memory, provide your evaluation with its exact ID, vote (\"helpful\" or \"harmful\"), and a brief reason."
    )
}

fn eval_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "memory_id": { "type": "string" },
                        "vote": { "type": "string", "enum": ["helpful", "harmful"] },
                        "reason": { "type": "string" }
                    },
                    "required": ["memory_id", "vote", "reason"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["evaluations"],
        "additionalProperties": false
    })
}

pub async fn evaluate_conversation(conv: ConversationWithMemories) -> Result<Vec<MemoryVote>> {
    let chunks = conv.into_chunks();
    let memories = &chunks
        .iter()
        .flat_map(|c| c.memory_ids.iter())
        .collect::<std::collections::HashSet<_>>();

    let _ = memories;

    let schema = eval_schema();
    let mut all_evals: HashMap<String, Vec<MemoryEval>> = HashMap::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.memory_ids.is_empty() {
            continue;
        }

        let memories_list: String = chunk
            .memory_ids
            .iter()
            .enumerate()
            .map(|(j, id)| format!("{}. [ID: {}]", j + 1, id))
            .collect::<Vec<_>>()
            .join("\n");

        let chunk_label = if chunks.len() > 1 {
            format!(" (chunk {}/{})", i + 1, chunks.len())
        } else {
            String::new()
        };

        let prompt = build_eval_prompt(&chunk.text, &memories_list);
        let output = run_claude(&prompt, "haiku", &schema)
            .await
            .with_context(|| format!("evaluating chunk{chunk_label}"))?;
        let response: EvalResponse =
            serde_json::from_str(&output).context("parsing evaluation response")?;

        for eval in response.evaluations {
            all_evals
                .entry(eval.memory_id.clone())
                .or_default()
                .push(eval);
        }
    }

    Ok(all_evals
        .into_iter()
        .map(|(memory_id, evals)| {
            let helpful = evals.iter().filter(|e| e.vote == "helpful").count();
            let harmful = evals.len() - helpful;
            let winner = if helpful >= harmful { "helpful" } else { "harmful" };
            let reason = evals
                .into_iter()
                .find(|e| e.vote == winner)
                .map(|e| e.reason)
                .unwrap_or_default();
            MemoryVote {
                memory_id,
                vote: winner.to_string(),
                reason,
                submitted: false,
            }
        })
        .collect())
}

pub async fn submit_votes(
    client: &Client,
    base_url: &str,
    api_key: &str,
    votes: &mut [MemoryVote],
) -> Result<(usize, usize)> {
    let mut submitted = 0;
    let mut failed = 0;

    for vote in votes.iter_mut() {
        if vote.submitted {
            continue;
        }

        let url = format!("{base_url}/api/v1/memories/{}/vote", vote.memory_id);
        let body = serde_json::json!({ "vote": vote.vote });

        match client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                println!("  {} {} ({})", vote.memory_id, vote.vote, vote.reason);
                vote.submitted = true;
                submitted += 1;
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                tracing::warn!(
                    memory_id = %vote.memory_id,
                    status = %status,
                    body = %text,
                    "vote submission failed"
                );
                failed += 1;
            }
            Err(e) => {
                tracing::warn!(memory_id = %vote.memory_id, error = %e, "vote request failed");
                failed += 1;
            }
        }
    }

    Ok((submitted, failed))
}

pub async fn run_vote(
    state_path: &Path,
    project_filter: Option<&str>,
    base_url: &str,
    api_key: &str,
) -> Result<()> {
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("no home directory"))?
        .join(".claude/projects");

    let mut state = VoteState::load(state_path)?;
    let projects = parser::discover_projects(&base)?;
    let client = Client::new();

    let mut total_evaluated = 0;
    let mut total_voted = 0;
    let mut total_skipped = 0;
    let mut total_failed = 0;

    for (project_name, conv_paths) in &projects {
        if let Some(filter) = project_filter {
            if !project_name.contains(filter) {
                continue;
            }
        }

        let clean_name = clean_project_name(project_name);
        let pending: Vec<&PathBuf> = conv_paths
            .iter()
            .filter(|p| {
                let sid = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                !state.conversations.contains_key(sid)
            })
            .collect();

        if pending.is_empty() {
            let evaluated = state
                .conversations
                .values()
                .filter(|c| c.project == *project_name && c.status == VoteStatus::Evaluated)
                .count();
            let voted = state
                .conversations
                .values()
                .filter(|c| c.project == *project_name && c.status == VoteStatus::Voted)
                .count();
            let no_mem = state
                .conversations
                .values()
                .filter(|c| c.project == *project_name && c.status == VoteStatus::NoMemories)
                .count();
            println!(
                "\u{2713} {:<30} {:>3} conv  {:>3} no-mem  {:>3} evaluated  {:>3} voted",
                clean_name,
                conv_paths.len(),
                no_mem,
                evaluated,
                voted
            );
            continue;
        }

        let total = conv_paths.len();
        let mut handled = total - pending.len();

        for path in &pending {
            let session_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            handled += 1;
            eprint!(
                "\x1b[2K\r  {:<30} {}/{}",
                clean_name, handled, total
            );

            let conv = match memory_extract::extract_memories_from_conversation(path)
            {
                Ok(Some(c)) => c,
                Ok(None) => {
                    state.conversations.insert(
                        session_id.clone(),
                        ConversationVoteState {
                            project: project_name.clone(),
                            session_id,
                            status: VoteStatus::NoMemories,
                            votes: vec![],
                        },
                    );
                    total_skipped += 1;
                    state.save(state_path)?;
                    continue;
                }
                Err(e) => {
                    tracing::warn!(session = %session_id, error = %e, "failed to extract memories");
                    state.conversations.insert(
                        session_id.clone(),
                        ConversationVoteState {
                            project: project_name.clone(),
                            session_id,
                            status: VoteStatus::Failed,
                            votes: vec![],
                        },
                    );
                    total_failed += 1;
                    state.save(state_path)?;
                    continue;
                }
            };

            let memory_count = conv.memories.len();
            eprint!(
                "\x1b[2K\r  {:<30} {}/{} | evaluating {} memories...",
                clean_name, handled, total, memory_count
            );

            let mut votes = match evaluate_conversation(conv).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(session = %session_id, error = %e, "evaluation failed");
                    state.conversations.insert(
                        session_id.clone(),
                        ConversationVoteState {
                            project: project_name.clone(),
                            session_id,
                            status: VoteStatus::Failed,
                            votes: vec![],
                        },
                    );
                    total_failed += 1;
                    state.save(state_path)?;
                    continue;
                }
            };

            total_evaluated += votes.len();

            eprint!(
                "\x1b[2K\r  {:<30} {}/{} | submitting {} votes...",
                clean_name,
                handled,
                total,
                votes.len()
            );

            let (submitted, failed) =
                submit_votes(&client, base_url, api_key, &mut votes).await?;
            total_voted += submitted;
            total_failed += failed;

            let status = if votes.iter().all(|v| v.submitted) {
                VoteStatus::Voted
            } else {
                VoteStatus::Evaluated
            };

            state.conversations.insert(
                session_id.clone(),
                ConversationVoteState {
                    project: project_name.clone(),
                    session_id,
                    status,
                    votes,
                },
            );

            state.save(state_path)?;
        }

        eprintln!("\x1b[2K\r\u{2713} {:<30} done", clean_name);
    }

    let retry_ids: Vec<String> = state
        .conversations
        .iter()
        .filter(|(_, c)| {
            c.status == VoteStatus::Evaluated
                && project_filter
                    .map(|f| c.project.contains(f))
                    .unwrap_or(true)
        })
        .map(|(id, _)| id.clone())
        .collect();

    if !retry_ids.is_empty() {
        eprintln!(
            "retrying {} conversations with unsubmitted votes...",
            retry_ids.len()
        );
        for session_id in &retry_ids {
            let conv_state = state.conversations.get_mut(session_id).unwrap();
            let unsubmitted = conv_state.votes.iter().filter(|v| !v.submitted).count();
            eprint!(
                "\x1b[2K\r  {} | submitting {} votes...",
                session_id, unsubmitted
            );
            let (submitted, failed) =
                submit_votes(&client, base_url, api_key, &mut conv_state.votes).await?;
            total_voted += submitted;
            total_failed += failed;

            if conv_state.votes.iter().all(|v| v.submitted) {
                conv_state.status = VoteStatus::Voted;
            }
            state.save(state_path)?;
        }
        eprintln!("\x1b[2K\rdone retrying");
    }

    println!();
    println!(
        "evaluated: {total_evaluated}, voted: {total_voted}, skipped (no memories): {total_skipped}, failed: {total_failed}"
    );

    Ok(())
}

fn clean_project_name(name: &str) -> String {
    if name == "-home-ntibi" {
        return "~".to_string();
    }
    if let Some(rest) = name.strip_prefix("-home-ntibi-ws-") {
        return rest.to_string();
    }
    if let Some(rest) = name.strip_prefix("-home-ntibi-") {
        return rest.to_string();
    }
    name.to_string()
}
