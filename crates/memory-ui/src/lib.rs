pub mod handlers;
pub mod templates;
pub mod static_files;

use std::sync::Arc;
use axum::{routing::{get, put, post}, Router};
use memory_core::{embed::Embedder, memory::MemoryStore, scoring::Scorer};

#[derive(Clone)]
pub struct UiState {
    pub store: Arc<MemoryStore>,
    pub embedder: Arc<dyn Embedder>,
    pub scorer: Arc<Scorer>,
}

pub fn router() -> Router<UiState> {
    Router::new()
        .route("/", get(handlers::index))
        .route("/memories", get(handlers::list_memories))
        .route("/memories/{id}/vote", post(handlers::vote_memory))
        .route("/memories/{id}", put(handlers::update_memory).delete(handlers::delete_memory))
        .route("/memories/{id}/edit", get(handlers::edit_memory_form))
        .route("/memories/{id}/card", get(handlers::get_card))
        .route("/tags", get(handlers::list_tags))
}

pub fn static_service() -> axum_embed::ServeEmbed<static_files::Assets> {
    static_files::serve()
}
