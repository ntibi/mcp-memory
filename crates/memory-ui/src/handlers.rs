use axum::{
    extract::{Path, Query, State},
    response::{Html, IntoResponse, Response},
    http::StatusCode,
};
use axum_htmx::HxRequest;
use serde::Deserialize;

use crate::UiState;
use crate::templates::*;

#[derive(Deserialize, Default)]
pub struct MemoryQuery {
    pub q: Option<String>,
    pub tag: Option<String>,
    pub cursor: Option<String>,
}

pub async fn index(HxRequest(is_htmx): HxRequest) -> Response {
    if is_htmx {
        StatusCode::OK.into_response()
    } else {
        LayoutTemplate.into_response()
    }
}

pub async fn list_memories(
    State(state): State<UiState>,
    HxRequest(_is_htmx): HxRequest,
    Query(q): Query<MemoryQuery>,
) -> Response {
    let limit = 20;
    let memories = if let Some(ref query) = q.q {
        if query.is_empty() {
            state.store.list(memory_core::memory::ListFilter {
                tag: q.tag.clone(),
                limit: Some(limit + 1),
                offset: None,
            }).await
        } else {
            match state.store.recall(query, limit + 1, state.embedder.as_ref(), state.scorer.as_ref()).await {
                Ok(scored) => Ok(scored.into_iter().map(|s| s.memory).collect()),
                Err(e) => Err(e),
            }
        }
    } else {
        state.store.list(memory_core::memory::ListFilter {
            tag: q.tag.clone(),
            limit: Some(limit + 1),
            offset: None,
        }).await
    };

    match memories {
        Ok(mut mems) => {
            let has_more = mems.len() > limit;
            mems.truncate(limit);

            let mut cards = Vec::new();
            for m in mems {
                let (helpful, harmful) = state.store.get_vote_counts(&m.id).await.unwrap_or((0, 0));
                cards.push(MemoryCard::from_memory(m, helpful, harmful));
            }

            let next_cursor = cards.last().map(|c| c.id.clone()).unwrap_or_default();

            CardGridTemplate {
                memories: cards,
                has_more,
                next_cursor,
                query: q.q.unwrap_or_default(),
                tag: q.tag.unwrap_or_default(),
            }.into_response()
        }
        Err(e) => {
            tracing::error!("failed to list memories: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

pub async fn list_tags(
    State(state): State<UiState>,
    Query(q): Query<MemoryQuery>,
) -> Response {
    match state.store.list_tags().await {
        Ok(tags) => {
            let total_count: usize = tags.iter().map(|t| t.1).sum();
            TagSidebarTemplate {
                tags,
                total_count,
                active_tag: q.tag.unwrap_or_default(),
            }.into_response()
        }
        Err(e) => {
            tracing::error!("failed to list tags: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[derive(Deserialize)]
pub struct VoteQuery {
    pub vote: String,
}

pub async fn vote_memory(
    State(state): State<UiState>,
    Path(id): Path<String>,
    Query(params): Query<VoteQuery>,
) -> Response {
    if let Err(e) = state.store.vote(&id, &params.vote).await {
        tracing::error!("vote failed: {e}");
        return StatusCode::BAD_REQUEST.into_response();
    }

    let (helpful, harmful) = state.store.get_vote_counts(&id).await.unwrap_or((0, 0));
    let memory = match state.store.get(&id).await {
        Ok(m) => m,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };

    VoteButtonsTemplate {
        card: MemoryCard::from_memory(memory, helpful, harmful),
    }.into_response()
}

pub async fn delete_memory(
    State(state): State<UiState>,
    Path(id): Path<String>,
) -> Response {
    match state.store.delete(&id).await {
        Ok(()) => Html("").into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn edit_memory_form(
    State(state): State<UiState>,
    Path(id): Path<String>,
) -> Response {
    let memory = match state.store.get(&id).await {
        Ok(m) => m,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let (helpful, harmful) = state.store.get_vote_counts(&id).await.unwrap_or((0, 0));
    let card = MemoryCard::from_memory(memory, helpful, harmful);

    CardEditTemplate { card }.into_response()
}

#[derive(Deserialize)]
pub struct UpdateForm {
    pub content: String,
    pub tags: Option<String>,
}

pub async fn update_memory(
    State(state): State<UiState>,
    Path(id): Path<String>,
    axum::Form(form): axum::Form<UpdateForm>,
) -> Response {
    if let Err(e) = state.store.update(&id, &form.content, state.embedder.as_ref()).await {
        tracing::error!("update failed: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    if let Some(tags_str) = form.tags {
        let tags: Vec<String> = tags_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let _ = state.store.set_tags(&id, tags).await;
    }

    let memory = match state.store.get(&id).await {
        Ok(m) => m,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let (helpful, harmful) = state.store.get_vote_counts(&id).await.unwrap_or((0, 0));

    CardTemplate {
        card: MemoryCard::from_memory(memory, helpful, harmful),
    }.into_response()
}

pub async fn get_card(
    State(state): State<UiState>,
    Path(id): Path<String>,
) -> Response {
    let memory = match state.store.get(&id).await {
        Ok(m) => m,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let (helpful, harmful) = state.store.get_vote_counts(&id).await.unwrap_or((0, 0));
    CardTemplate {
        card: MemoryCard::from_memory(memory, helpful, harmful),
    }.into_response()
}
