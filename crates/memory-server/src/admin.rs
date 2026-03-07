use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use memory_core::memory::ListFilter;
use serde::Deserialize;
use serde_json::json;

use crate::api::{self, AppState};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/users", get(list_users).post(create_user))
        .route("/users/{id}", axum::routing::delete(delete_user))
        .route("/users/{id}/keys", get(list_keys).post(create_key))
        .route("/keys/{id}", axum::routing::delete(revoke_key))
        .route("/admins", get(list_admins).post(add_admin))
        .route("/admins/{user_id}", axum::routing::delete(remove_admin))
        .route("/memories", get(list_all_memories))
        .route("/memories/{id}", axum::routing::delete(delete_any_memory))
        .route("/stats", get(stats))
}

#[derive(Deserialize)]
struct CreateUserBody {
    name: String,
}

#[derive(Deserialize)]
struct CreateKeyBody {
    name: String,
}

#[derive(Deserialize)]
struct AddAdminBody {
    user_id: String,
}

#[derive(Deserialize)]
struct AdminMemoryQuery {
    user_id: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_users(State(state): State<AppState>) -> impl IntoResponse {
    match state.user_store.list_users().await {
        Ok(users) => Json(json!(users)).into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn create_user(
    State(state): State<AppState>,
    Json(body): Json<CreateUserBody>,
) -> impl IntoResponse {
    match state.user_store.create_user(&body.name).await {
        Ok(user) => (StatusCode::CREATED, Json(json!(user))).into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn delete_user(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.user_store.delete_user(&id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn list_keys(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match state.user_store.list_api_keys(&user_id).await {
        Ok(keys) => Json(json!(keys)).into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn create_key(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Json(body): Json<CreateKeyBody>,
) -> impl IntoResponse {
    match state.user_store.create_api_key(&user_id, &body.name).await {
        Ok(key) => (StatusCode::CREATED, Json(json!(key))).into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn revoke_key(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.user_store.revoke_api_key(&id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn list_admins(State(state): State<AppState>) -> impl IntoResponse {
    match state.user_store.list_admins().await {
        Ok(admins) => Json(json!(admins)).into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn add_admin(
    State(state): State<AppState>,
    Json(body): Json<AddAdminBody>,
) -> impl IntoResponse {
    match state.user_store.add_admin(&body.user_id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn remove_admin(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match state.user_store.remove_admin(&user_id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn list_all_memories(
    State(state): State<AppState>,
    Query(query): Query<AdminMemoryQuery>,
) -> impl IntoResponse {
    let filter = ListFilter {
        tags: vec![],
        limit: query.limit,
        offset: query.offset,
    };
    match query.user_id {
        Some(uid) => match state.store.list(&uid, filter).await {
            Ok(memories) => Json(json!(memories)).into_response(),
            Err(e) => api::error_response(&e),
        },
        None => match state.store.admin_list(filter).await {
            Ok(memories) => Json(json!(memories)).into_response(),
            Err(e) => api::error_response(&e),
        },
    }
}

async fn delete_any_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.admin_delete(&id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => api::error_response(&e),
    }
}

async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    match state.store.admin_stats().await {
        Ok(stats) => Json(json!(stats)).into_response(),
        Err(e) => api::error_response(&e),
    }
}
