use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use memory_core::embed::Embedder;
use memory_core::error::Error;
use memory_core::memory::{CreateMemory, ListFilter, MemoryStore};
use memory_core::scoring::Scorer;
use memory_core::users::{AuthContext, UserStore};
use serde::Deserialize;
use serde_json::json;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<MemoryStore>,
    pub embedder: Arc<dyn Embedder>,
    pub scorer: Arc<Scorer>,
    pub conn: tokio_rusqlite::Connection,
    pub user_store: Arc<UserStore>,
    pub scheduler: Arc<tokio::sync::Mutex<crate::scheduler::Scheduler>>,
    pub progress_map: crate::curation_worker::ProgressMap,
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/memories", get(list_memories).post(create_memory))
        .route(
            "/memories/{id}",
            get(get_memory).put(update_memory).delete(delete_memory),
        )
        .route("/memories/{id}/vote", post(vote_memory))
        .route("/memories/duplicates", get(find_duplicates_handler))
        .route("/curation/suggestions", get(list_suggestions_handler))
        .route(
            "/curation/suggestions/{id}/apply",
            post(apply_suggestion_handler),
        )
        .route(
            "/curation/suggestions/{id}/dismiss",
            post(dismiss_suggestion_handler),
        )
        .route(
            "/curation/settings",
            get(get_curation_settings).put(update_curation_settings),
        )
        .route("/curation/run", post(trigger_run))
        .route("/curation/run/{id}/cancel", post(cancel_run))
        .route("/curation/runs", get(list_curation_runs))
        .route("/health", get(health))
}

#[derive(Deserialize)]
struct CreateBody {
    content: String,
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct UpdateBody {
    content: String,
}

#[derive(Deserialize)]
struct VoteBody {
    vote: String,
}

#[derive(Deserialize)]
struct ListQuery {
    tag: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_memories(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    let tags = query
        .tag
        .as_deref()
        .map(memory_core::tags::parse_comma_separated)
        .unwrap_or_default();
    let filter = ListFilter {
        tags,
        limit: query.limit,
        offset: query.offset,
        ..Default::default()
    };

    match state.store.list(&auth.user_id, filter).await {
        Ok(memories) => Json(json!(memories)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn get_memory(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.get(&auth.user_id, &id).await {
        Ok(memory) => Json(json!(memory)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn create_memory(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<CreateBody>,
) -> impl IntoResponse {
    let input = CreateMemory {
        content: body.content,
        tags: body.tags.unwrap_or_default(),
    };

    match state
        .store
        .create(&auth.user_id, input, state.embedder.as_ref())
        .await
    {
        Ok(memory) => (StatusCode::CREATED, Json(json!(memory))).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn update_memory(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(body): Json<UpdateBody>,
) -> impl IntoResponse {
    match state
        .store
        .update(&auth.user_id, &id, &body.content, state.embedder.as_ref())
        .await
    {
        Ok(memory) => Json(json!(memory)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn delete_memory(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.delete(&auth.user_id, &id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(&e),
    }
}

async fn vote_memory(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(body): Json<VoteBody>,
) -> impl IntoResponse {
    match state.store.vote(&auth.user_id, &id, &body.vote).await {
        Ok(vote) => (StatusCode::CREATED, Json(json!(vote))).into_response(),
        Err(e) => error_response(&e),
    }
}

#[derive(Deserialize)]
struct DuplicatesQuery {
    threshold: Option<f64>,
}

#[derive(Deserialize)]
struct SuggestionsQuery {
    status: Option<String>,
}

async fn find_duplicates_handler(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<DuplicatesQuery>,
) -> impl IntoResponse {
    let threshold = query.threshold.unwrap_or(0.85);
    match memory_core::curation::find_duplicates(&state.conn, &auth.user_id, threshold).await {
        Ok(candidates) => Json(json!(candidates)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn list_suggestions_handler(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<SuggestionsQuery>,
) -> impl IntoResponse {
    match memory_core::curation::list_suggestions(&state.conn, &auth.user_id, query.status.as_deref())
        .await
    {
        Ok(suggestions) => Json(json!(suggestions)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn apply_suggestion_handler(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match memory_core::curation::apply_suggestion(
        &state.conn,
        &state.store,
        state.embedder.as_ref(),
        &auth.user_id,
        &id,
    )
    .await
    {
        Ok(()) => Json(json!({"status": "applied"})).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn dismiss_suggestion_handler(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let suggestion = match memory_core::curation::get_suggestion(&state.conn, &auth.user_id, &id).await {
        Ok(s) => s,
        Err(e) => return error_response(&e),
    };

    if let Err(e) = memory_core::curation::update_suggestion_status(
        &state.conn,
        &auth.user_id,
        &id,
        "dismissed",
    )
    .await
    {
        return error_response(&e);
    }

    let memory_ids = &suggestion.memory_ids;
    for i in 0..memory_ids.len() {
        for j in (i + 1)..memory_ids.len() {
            let _ = memory_core::curation::store_dismissed_pair(
                &state.conn,
                &auth.user_id,
                &memory_ids[i],
                &memory_ids[j],
            )
            .await;
        }
    }

    Json(json!({"status": "dismissed"})).into_response()
}

fn mask_api_key(key: &Option<String>) -> serde_json::Value {
    match key {
        Some(k) if k.len() >= 3 => {
            let suffix = &k[k.len() - 3..];
            json!(format!("sk-...{suffix}"))
        }
        Some(_) => json!("sk-...***"),
        None => json!(null),
    }
}

async fn get_curation_settings(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
) -> impl IntoResponse {
    match memory_core::curation::get_settings(&state.conn, &auth.user_id).await {
        Ok(settings) => Json(json!({
            "user_id": settings.user_id,
            "api_key": mask_api_key(&settings.api_key),
            "schedule_windows": settings.schedule_windows,
            "similarity_threshold": settings.similarity_threshold,
            "budget_limit_usd": settings.budget_limit_usd,
            "model": settings.model,
            "enabled": settings.enabled,
        }))
        .into_response(),
        Err(e) => error_response(&e),
    }
}

#[derive(Deserialize)]
struct UpdateCurationSettingsBody {
    api_key: Option<Option<String>>,
    schedule_windows: Option<Vec<memory_core::curation::ScheduleWindow>>,
    similarity_threshold: Option<f64>,
    budget_limit_usd: Option<Option<f64>>,
    model: Option<String>,
    enabled: Option<bool>,
}

async fn update_curation_settings(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<UpdateCurationSettingsBody>,
) -> impl IntoResponse {
    let mut settings = match memory_core::curation::get_settings(&state.conn, &auth.user_id).await {
        Ok(s) => s,
        Err(e) => return error_response(&e),
    };

    if let Some(key) = body.api_key {
        settings.api_key = key;
    }
    if let Some(windows) = body.schedule_windows {
        settings.schedule_windows = windows;
    }
    if let Some(threshold) = body.similarity_threshold {
        settings.similarity_threshold = threshold;
    }
    if let Some(budget) = body.budget_limit_usd {
        settings.budget_limit_usd = budget;
    }
    if let Some(model) = body.model {
        settings.model = model;
    }
    if let Some(enabled) = body.enabled {
        settings.enabled = enabled;
    }

    match memory_core::curation::upsert_settings(&state.conn, &settings).await {
        Ok(()) => Json(json!({
            "user_id": settings.user_id,
            "api_key": mask_api_key(&settings.api_key),
            "schedule_windows": settings.schedule_windows,
            "similarity_threshold": settings.similarity_threshold,
            "budget_limit_usd": settings.budget_limit_usd,
            "model": settings.model,
            "enabled": settings.enabled,
        }))
        .into_response(),
        Err(e) => error_response(&e),
    }
}

async fn trigger_run(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
) -> impl IntoResponse {
    let mut scheduler = state.scheduler.lock().await;
    match scheduler.start_run(&auth.user_id).await {
        Ok(run_id) => Json(json!({"run_id": run_id})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

async fn cancel_run(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let mut scheduler = state.scheduler.lock().await;
    match scheduler.cancel_run(&auth.user_id, &id).await {
        Ok(()) => Json(json!({"status": "cancelling"})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

#[derive(Deserialize)]
struct RunsQuery {
    limit: Option<usize>,
}

async fn list_curation_runs(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<RunsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(20);
    match memory_core::curation::list_runs(&state.conn, &auth.user_id, limit).await {
        Ok(runs) => Json(json!(runs)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

pub fn error_response(err: &Error) -> axum::response::Response {
    let (status, message) = match err {
        Error::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
        Error::InvalidInput(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal server error".to_string(),
        ),
    };
    (status, Json(json!({"error": message}))).into_response()
}
