use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use memory_core::embed::Embedder;
use memory_core::error::Error;
use memory_core::memory::{CreateMemory, ListFilter, MemoryStore};
use memory_core::scoring::Scorer;
use serde::Deserialize;
use serde_json::json;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<MemoryStore>,
    pub embedder: Arc<dyn Embedder>,
    pub scorer: Arc<Scorer>,
    pub conn: tokio_rusqlite::Connection,
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
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    let filter = ListFilter {
        tag: query.tag,
        limit: query.limit,
        offset: query.offset,
    };

    match state.store.list(filter).await {
        Ok(memories) => Json(json!(memories)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.get(&id).await {
        Ok(memory) => Json(json!(memory)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn create_memory(
    State(state): State<AppState>,
    Json(body): Json<CreateBody>,
) -> impl IntoResponse {
    let input = CreateMemory {
        content: body.content,
        tags: body.tags.unwrap_or_default(),
    };

    match state.store.create(input, state.embedder.as_ref()).await {
        Ok(memory) => (StatusCode::CREATED, Json(json!(memory))).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<UpdateBody>,
) -> impl IntoResponse {
    match state
        .store
        .update(&id, &body.content, state.embedder.as_ref())
        .await
    {
        Ok(memory) => Json(json!(memory)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.delete(&id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(&e),
    }
}

async fn vote_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<VoteBody>,
) -> impl IntoResponse {
    match state.store.vote(&id, &body.vote).await {
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
    Query(query): Query<DuplicatesQuery>,
) -> impl IntoResponse {
    let threshold = query.threshold.unwrap_or(0.85);
    match memory_core::curation::find_duplicates(&state.conn, threshold).await {
        Ok(candidates) => Json(json!(candidates)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn list_suggestions_handler(
    State(state): State<AppState>,
    Query(query): Query<SuggestionsQuery>,
) -> impl IntoResponse {
    match memory_core::curation::list_suggestions(&state.conn, query.status.as_deref()).await {
        Ok(suggestions) => Json(json!(suggestions)).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn apply_suggestion_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match memory_core::curation::update_suggestion_status(&state.conn, &id, "applied").await {
        Ok(()) => Json(json!({"status": "applied"})).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn dismiss_suggestion_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match memory_core::curation::update_suggestion_status(&state.conn, &id, "dismissed").await {
        Ok(()) => Json(json!({"status": "dismissed"})).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

fn error_response(err: &Error) -> axum::response::Response {
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
