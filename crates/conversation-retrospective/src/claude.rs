use std::process::Stdio;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
pub struct FilterResponse {
    pub interesting: bool,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
pub struct ExtractionResponse {
    pub learnings: Vec<ExtractedLearning>,
}

#[derive(Debug, Deserialize)]
pub struct ExtractedLearning {
    pub content: String,
    pub tags: Vec<String>,
    pub category: String,
}

pub async fn filter_conversation(condensed: &str) -> Result<FilterResponse> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "interesting": { "type": "boolean" },
            "reason": { "type": "string" }
        },
        "required": ["interesting", "reason"],
        "additionalProperties": false
    });

    let prompt = format!(
        "Analyze this Claude Code conversation log. Is it interesting enough to extract learnings from?\n\
        \n\
        Interesting = contains debugging insights, architectural decisions, workflow preferences, gotchas, quirks, \
        patterns, workarounds, user preferences, things that took multiple attempts, things the LLM got wrong initially, \
        or anything that would make a similar future conversation shorter/smoother.\n\
        \n\
        NOT interesting = simple one-off questions, routine code generation with no friction, conversations where \
        everything went smoothly on the first try with no notable patterns.\n\
        \n\
        <conversation>\n{condensed}\n</conversation>"
    );

    let output = run_claude(&prompt, "haiku", &schema).await?;
    serde_json::from_str(&output).context("parsing filter response")
}

pub async fn extract_learnings(condensed: &str) -> Result<ExtractionResponse> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "learnings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "tags": { "type": "array", "items": { "type": "string" } },
                        "category": { "type": "string" }
                    },
                    "required": ["content", "tags", "category"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["learnings"],
        "additionalProperties": false
    });

    let prompt = format!(
        "Analyze this Claude Code conversation log. Extract ALL actionable learnings.\n\
        \n\
        Each learning should be something that, if known ahead of time, would have made this conversation \
        shorter, smoother, or avoided mistakes. Include:\n\
        - Debugging insights and root causes\n\
        - Architectural decisions and trade-offs\n\
        - User preferences and workflow patterns\n\
        - Gotchas, quirks, and workarounds\n\
        - Tool usage patterns\n\
        - Codebase-specific knowledge\n\
        - Anything else of value\n\
        \n\
        Categories: debugging, architecture, preference, gotcha, workflow, pattern, tool-usage, codebase-knowledge\n\
        \n\
        For tags, include: the project domain, technologies involved, and the category.\n\
        \n\
        <conversation>\n{condensed}\n</conversation>"
    );

    let output = run_claude(&prompt, "sonnet", &schema).await?;
    serde_json::from_str(&output).context("parsing extraction response")
}

async fn run_claude(prompt: &str, model: &str, json_schema: &serde_json::Value) -> Result<String> {
    let schema_str = serde_json::to_string(json_schema)?;

    let mut cmd = Command::new("claude");
    cmd.args([
        "--print",
        "--model",
        model,
        "--output-format",
        "json",
        "--json-schema",
        &schema_str,
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        prompt,
    ]);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.env_remove("CLAUDECODE");

    let child = cmd.spawn().context("spawning claude process")?;
    let output = child
        .wait_with_output()
        .await
        .context("waiting for claude process")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("claude exited with {}: {}", output.status, stderr);
    }

    let stdout = String::from_utf8(output.stdout).context("claude output is not utf-8")?;

    let wrapper: serde_json::Value =
        serde_json::from_str(&stdout).context("parsing claude json output")?;

    if wrapper.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false) {
        let msg = wrapper.get("result").and_then(|v| v.as_str()).unwrap_or("unknown error");
        bail!("claude returned an error: {msg}");
    }

    let structured = wrapper
        .get("structured_output")
        .ok_or_else(|| anyhow::anyhow!("no structured_output in claude response"))?;

    Ok(serde_json::to_string(structured)?)
}
