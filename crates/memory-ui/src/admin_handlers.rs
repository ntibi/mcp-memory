use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Extension,
};
use memory_core::users::{ApiKey, AuthContext, UserStore};

use crate::templates::{AdminStatsPageTemplate, AdminStatsDataTemplate, AdminUsersListTemplate, AdminUsersPageTemplate};

#[derive(Clone)]
pub struct AdminUiState {
    pub user_store: Arc<UserStore>,
    pub store: Arc<memory_core::memory::MemoryStore>,
}

pub async fn users_page() -> Response {
    AdminUsersPageTemplate.into_response()
}

pub async fn users_list(
    State(state): State<AdminUiState>,
    Extension(_auth): Extension<AuthContext>,
) -> Response {
    let users = match state.user_store.list_users().await {
        Ok(u) => u,
        Err(e) => {
            tracing::error!("failed to list users: {e}");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let admin_list = state.user_store.list_admins().await.unwrap_or_default();
    let admin_ids: Vec<String> = admin_list.into_iter().map(|a| a.id).collect();

    let mut keys: HashMap<String, Vec<ApiKey>> = HashMap::new();
    for user in &users {
        let user_keys = state.user_store.list_api_keys(&user.id).await.unwrap_or_default();
        keys.insert(user.id.clone(), user_keys);
    }

    AdminUsersListTemplate { users, admin_ids, keys }.into_response()
}

pub async fn stats_page() -> Response {
    AdminStatsPageTemplate.into_response()
}

pub async fn stats_data(
    State(state): State<AdminUiState>,
    Extension(_auth): Extension<AuthContext>,
) -> Response {
    match state.store.admin_stats().await {
        Ok(stats) => AdminStatsDataTemplate { stats }.into_response(),
        Err(e) => {
            tracing::error!("failed to get stats: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}
