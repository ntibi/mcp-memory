use std::collections::HashSet;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Extension, Router,
};
use memory_core::curation::{self, ProgressMap};
use memory_core::embed::Embedder;
use memory_core::memory::MemoryStore;
use memory_core::users::AuthContext;
use memory_ui::templates::{
    CurationDashboardTemplate, CurationIndicatorTemplate, CurationRunsTemplate,
    CurationSettingsTemplate, CurationStatusTemplate, CurationSuggestionsTemplate, SuggestionCard,
};
use serde::Deserialize;

use reqwest::Client;

use crate::scheduler::Scheduler;

#[derive(Clone)]
pub struct CurationUiState {
    pub conn: tokio_rusqlite::Connection,
    pub store: Arc<MemoryStore>,
    pub embedder: Arc<dyn Embedder>,
    pub scheduler: Arc<tokio::sync::Mutex<Scheduler>>,
    pub progress_map: ProgressMap,
}

pub fn router() -> Router<CurationUiState> {
    Router::new()
        .route("/", get(dashboard_page))
        .route("/settings", get(settings_page).put(update_settings))
        .route("/check-model", post(check_model))
        .route("/list-models", get(list_models))
        .route("/status", get(status_fragment))
        .route("/runs", get(runs_fragment))
        .route("/run", post(trigger_run))
        .route("/run/{id}/cancel", post(cancel_run))
        .route("/suggestions", get(suggestions_fragment))
        .route("/suggestions/{id}/apply", post(apply_suggestion))
        .route("/suggestions/{id}/dismiss", post(dismiss_suggestion))
        .route("/indicator", get(indicator_fragment))
}

async fn settings_page(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
) -> Response {
    let settings = match curation::get_settings(&state.conn, &auth.user_id).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("failed to load curation settings: {e}");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let schedule_days: Vec<u8> = settings
        .schedule_windows
        .iter()
        .flat_map(|w| w.days.iter().copied())
        .collect();

    CurationSettingsTemplate {
        settings,
        is_admin: auth.is_admin,
        schedule_days,
    }
    .into_response()
}

fn build_settings_response(settings: curation::CurationSettings, is_admin: bool) -> Response {
    let schedule_days: Vec<u8> = settings
        .schedule_windows
        .iter()
        .flat_map(|w| w.days.iter().copied())
        .collect();
    CurationSettingsTemplate {
        settings,
        is_admin,
        schedule_days,
    }
    .into_response()
}

#[derive(Deserialize)]
struct SettingsForm {
    provider: Option<String>,
    api_key: Option<String>,
    similarity_threshold: Option<f64>,
    budget_limit_usd: Option<String>,
    model: Option<String>,
    enabled: Option<String>,
    schedule_days: Option<String>,
    schedule_start: Option<String>,
    schedule_end: Option<String>,
}

async fn update_settings(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
    axum::Form(form): axum::Form<SettingsForm>,
) -> Response {
    let mut settings = match curation::get_settings(&state.conn, &auth.user_id).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("failed to load curation settings: {e}");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    if let Some(ref provider) = form.provider {
        if !provider.is_empty() {
            settings.provider = provider.clone();
        }
    }

    if let Some(ref key) = form.api_key {
        if key.is_empty() {
            settings.api_key = None;
        } else if !key.starts_with("***") {
            settings.api_key = Some(key.clone());
        }
    }

    if let Some(threshold) = form.similarity_threshold {
        settings.similarity_threshold = threshold.clamp(0.5, 1.0);
    }

    if let Some(ref budget) = form.budget_limit_usd {
        settings.budget_limit_usd = if budget.is_empty() {
            None
        } else {
            budget.parse::<f64>().ok()
        };
    }

    if let Some(ref model) = form.model {
        if !model.is_empty() {
            settings.model = model.clone();
        }
    }

    settings.enabled = form.enabled.as_deref() == Some("on");

    let mut windows = Vec::new();
    if let (Some(days_str), Some(start), Some(end)) =
        (&form.schedule_days, &form.schedule_start, &form.schedule_end)
    {
        if !days_str.is_empty() && !start.is_empty() && !end.is_empty() {
            let days: Vec<u8> = days_str
                .split(',')
                .filter_map(|d| d.trim().parse().ok())
                .collect();
            if !days.is_empty() {
                windows.push(curation::ScheduleWindow {
                    days,
                    start: start.clone(),
                    end: end.clone(),
                });
            }
        }
    }
    settings.schedule_windows = windows;

    if let Some(ref api_key) = settings.api_key {
        if !api_key.starts_with("***") {
            match fetch_models(api_key).await {
                Ok(models) => {
                    if !models.iter().any(|(id, _)| id == &settings.model) {
                        return Html(format!(
                            "<h1>Curation Settings</h1><div class=\"curation-error\">model '{}' not found in your Anthropic account</div>",
                            settings.model
                        )).into_response();
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "model validation failed, saving anyway");
                }
            }
        }
    }

    if let Err(e) = curation::upsert_settings(&state.conn, &settings).await {
        tracing::error!("failed to save curation settings: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    build_settings_response(settings, auth.is_admin)
}

async fn fetch_models(api_key: &str) -> anyhow::Result<Vec<(String, String)>> {
    let http = Client::new();
    let mut models = Vec::new();
    let mut after_id: Option<String> = None;
    loop {
        let mut url = "https://api.anthropic.com/v1/models?limit=100".to_string();
        if let Some(ref cursor) = after_id {
            url.push_str(&format!("&after_id={cursor}"));
        }
        let resp = http
            .get(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;
        if !resp.status().is_success() {
            anyhow::bail!("anthropic api error: {}", resp.status());
        }
        let body: serde_json::Value = resp.json().await?;
        let empty = vec![];
        let data = body["data"].as_array().unwrap_or(&empty);
        for m in data {
            if let (Some(id), Some(name)) = (m["id"].as_str(), m["display_name"].as_str()) {
                models.push((id.to_string(), name.to_string()));
            }
        }
        if body["has_more"].as_bool() != Some(true) {
            break;
        }
        after_id = body["last_id"].as_str().map(|s| s.to_string());
    }
    models.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(models)
}

async fn resolve_api_key(
    conn: &tokio_rusqlite::Connection,
    user_id: &str,
    form_key: Option<&str>,
) -> Result<String, Response> {
    if let Some(k) = form_key {
        if !k.is_empty() && !k.starts_with("***") {
            return Ok(k.to_string());
        }
    }
    match curation::get_settings(conn, user_id).await {
        Ok(s) => match s.api_key {
            Some(k) => Ok(k),
            None => Err(Html("<span class=\"check-error\">no API key configured</span>").into_response()),
        },
        Err(_) => Err(Html("<span class=\"check-error\">failed to load settings</span>").into_response()),
    }
}

#[derive(Deserialize)]
struct CheckModelForm {
    model: String,
    api_key: Option<String>,
}

async fn check_model(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
    axum::Form(form): axum::Form<CheckModelForm>,
) -> Response {
    let api_key = match resolve_api_key(&state.conn, &auth.user_id, form.api_key.as_deref()).await {
        Ok(k) => k,
        Err(r) => return r,
    };

    match fetch_models(&api_key).await {
        Ok(models) => {
            if models.iter().any(|(id, _)| id == &form.model) {
                Html("<span class=\"check-ok\">valid</span>").into_response()
            } else {
                Html(format!("<span class=\"check-error\">model '{}' not found</span>", form.model)).into_response()
            }
        }
        Err(e) => Html(format!("<span class=\"check-error\">{e}</span>")).into_response(),
    }
}

async fn list_models(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
) -> Response {
    let api_key = match curation::get_settings(&state.conn, &auth.user_id).await {
        Ok(s) => match s.api_key {
            Some(k) => k,
            None => return axum::Json(serde_json::json!({"error": "no API key"})).into_response(),
        },
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    match fetch_models(&api_key).await {
        Ok(models) => {
            let items: Vec<serde_json::Value> = models
                .into_iter()
                .map(|(id, name)| serde_json::json!({"id": id, "name": name}))
                .collect();
            axum::Json(serde_json::json!({"models": items})).into_response()
        }
        Err(e) => axum::Json(serde_json::json!({"error": e.to_string()})).into_response(),
    }
}

async fn dashboard_page(Extension(auth): Extension<AuthContext>) -> Response {
    CurationDashboardTemplate {
        is_admin: auth.is_admin,
    }
    .into_response()
}

async fn status_fragment(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
) -> Response {
    let progress = match curation::get_active_run(&state.conn, &auth.user_id).await {
        Ok(Some(run)) => state.progress_map.get(&run.id).map(|e| e.value().clone()),
        _ => None,
    };

    let last_run = match curation::list_runs(&state.conn, &auth.user_id, 1).await {
        Ok(runs) => runs.into_iter().next(),
        Err(_) => None,
    };

    CurationStatusTemplate {
        progress,
        last_run,
    }
    .into_response()
}

async fn runs_fragment(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
) -> Response {
    let runs = curation::list_runs(&state.conn, &auth.user_id, 20)
        .await
        .unwrap_or_default();

    CurationRunsTemplate { runs }.into_response()
}

async fn trigger_run(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
) -> Response {
    let mut scheduler = state.scheduler.lock().await;
    if let Err(e) = scheduler.start_run(&auth.user_id).await {
        tracing::warn!("failed to trigger curation run: {e}");
        return Html(format!(
            "<div class=\"curation-error\">{e}</div>"
        ))
        .into_response();
    }
    drop(scheduler);

    let progress = match curation::get_active_run(&state.conn, &auth.user_id).await {
        Ok(Some(run)) => state.progress_map.get(&run.id).map(|e| e.value().clone()),
        _ => None,
    };
    let last_run = match curation::list_runs(&state.conn, &auth.user_id, 1).await {
        Ok(runs) => runs.into_iter().next(),
        Err(_) => None,
    };

    CurationStatusTemplate {
        progress,
        last_run,
    }
    .into_response()
}

async fn cancel_run(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    let mut scheduler = state.scheduler.lock().await;
    let _ = scheduler.cancel_run(&auth.user_id, &id).await;
    drop(scheduler);

    let progress = match curation::get_active_run(&state.conn, &auth.user_id).await {
        Ok(Some(run)) => state.progress_map.get(&run.id).map(|e| e.value().clone()),
        _ => None,
    };
    let last_run = match curation::list_runs(&state.conn, &auth.user_id, 1).await {
        Ok(runs) => runs.into_iter().next(),
        Err(_) => None,
    };

    CurationStatusTemplate {
        progress,
        last_run,
    }
    .into_response()
}

async fn suggestions_fragment(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
) -> Response {
    let suggestions =
        match curation::list_suggestions(&state.conn, &auth.user_id, Some("pending")).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("failed to list suggestions: {e}");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        };

    let mut cards = Vec::new();
    for suggestion in suggestions {
        let parsed: SuggestionData = match serde_json::from_str(&suggestion.suggestion) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mut source_memories = Vec::new();
        for mid in &suggestion.memory_ids {
            if let Ok(m) = state.store.get(&auth.user_id, mid).await {
                source_memories.push(m);
            }
        }

        let existing_tags: HashSet<String> = source_memories
            .iter()
            .flat_map(|m| m.tags.iter().cloned())
            .collect();
        let proposed_tags: HashSet<String> = parsed.tags.iter().cloned().collect();

        let added_tags: Vec<String> = proposed_tags.difference(&existing_tags).cloned().collect();
        let removed_tags: Vec<String> = existing_tags.difference(&proposed_tags).cloned().collect();
        let unchanged_tags: Vec<String> =
            proposed_tags.intersection(&existing_tags).cloned().collect();

        cards.push(SuggestionCard {
            id: suggestion.id,
            action: parsed.action,
            reasoning: parsed.reasoning,
            proposed_content: parsed.content,
            proposed_tags: parsed.tags,
            source_memories,
            added_tags,
            removed_tags,
            unchanged_tags,
        });
    }

    CurationSuggestionsTemplate { suggestions: cards }.into_response()
}

#[derive(Deserialize)]
struct SuggestionData {
    action: String,
    content: String,
    tags: Vec<String>,
    reasoning: String,
}

async fn apply_suggestion(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    match curation::apply_suggestion(
        &state.conn,
        &state.store,
        state.embedder.as_ref(),
        &auth.user_id,
        &id,
    )
    .await
    {
        Ok(()) => Html("").into_response(),
        Err(e) => {
            tracing::error!("failed to apply suggestion: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn dismiss_suggestion(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    let suggestion = match curation::get_suggestion(&state.conn, &auth.user_id, &id).await {
        Ok(s) => s,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };

    if let Err(e) =
        curation::update_suggestion_status(&state.conn, &auth.user_id, &id, "dismissed").await
    {
        tracing::error!("failed to dismiss suggestion: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    let memory_ids = &suggestion.memory_ids;
    for i in 0..memory_ids.len() {
        for j in (i + 1)..memory_ids.len() {
            let _ = curation::store_dismissed_pair(
                &state.conn,
                &auth.user_id,
                &memory_ids[i],
                &memory_ids[j],
            )
            .await;
        }
    }

    Html("").into_response()
}

async fn indicator_fragment(
    State(state): State<CurationUiState>,
    Extension(auth): Extension<AuthContext>,
) -> Response {
    let is_running = matches!(
        curation::get_active_run(&state.conn, &auth.user_id).await,
        Ok(Some(_))
    );

    let pending_count =
        match curation::list_suggestions(&state.conn, &auth.user_id, Some("pending")).await {
            Ok(s) => s.len(),
            Err(_) => 0,
        };

    CurationIndicatorTemplate {
        is_running,
        pending_count,
    }
    .into_response()
}
