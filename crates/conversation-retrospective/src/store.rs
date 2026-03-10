use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct JsonRpcResponse {
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    message: String,
}

pub async fn store_learnings(
    client: &Client,
    base_url: &str,
    api_key: &str,
    content: &str,
    tags: &[String],
) -> Result<()> {
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "store_memory",
            "arguments": {
                "content": content,
                "tags": tags
            }
        }
    });

    let response = client
        .post(format!("{base_url}/mcp"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await
        .context("sending store request")?;

    let status = response.status();
    let text = response.text().await.context("reading response")?;

    if !status.is_success() {
        bail!("store failed with {status}: {text}");
    }

    let rpc_response: JsonRpcResponse =
        serde_json::from_str(&text).context("parsing json-rpc response")?;

    if let Some(err) = rpc_response.error {
        bail!("json-rpc error: {}", err.message);
    }

    Ok(())
}
