use axum::{
    extract::{Path, Query, State},
    response::{Html, IntoResponse, Response},
    http::StatusCode,
    Extension,
};
use axum_htmx::HxRequest;
use memory_core::users::AuthContext;
use serde::Deserialize;

use crate::UiState;
use crate::templates::{
    CardEditTemplate, CardGridTemplate, CardTemplate, LayoutTemplate, LoginTemplate, MemoryCard,
    TagSidebarTemplate, VoteButtonsTemplate,
};

#[derive(Deserialize, Default)]
pub struct MemoryQuery {
    pub q: Option<String>,
    pub tag: Option<String>,
    pub cursor: Option<String>,
}

pub async fn index(
    Extension(auth): Extension<AuthContext>,
    HxRequest(is_htmx): HxRequest,
) -> Response {
    if is_htmx {
        StatusCode::OK.into_response()
    } else {
        LayoutTemplate { is_admin: auth.is_admin }.into_response()
    }
}

pub async fn list_memories(
    State(state): State<UiState>,
    Extension(auth): Extension<AuthContext>,
    HxRequest(is_htmx): HxRequest,
    Query(q): Query<MemoryQuery>,
) -> Response {
    if !is_htmx {
        return LayoutTemplate { is_admin: auth.is_admin }.into_response();
    }

    let limit = 20;
    let offset = q.cursor.as_deref().and_then(|c| c.parse::<usize>().ok()).unwrap_or(0);
    let is_semantic = q.q.as_ref().is_some_and(|s| !s.is_empty());
    let tags = q.tag.as_deref().map(memory_core::tags::parse_comma_separated).unwrap_or_default();

    let scored_results = if let Some(ref query) = q.q {
        if query.is_empty() {
            state.store.list(&auth.user_id, memory_core::memory::ListFilter {
                tags: tags.clone(),
                limit: Some(limit + 1),
                offset: Some(offset),
            }).await.map(|mems| mems.into_iter().map(|m| (m, None)).collect::<Vec<_>>())
        } else {
            state.store.recall(&auth.user_id, query, limit + 1, state.embedder.as_ref(), state.scorer.as_ref()).await
                .map(|scored| scored.into_iter().map(|s| (s.memory, Some((s.relevance, s.confidence, s.recency, s.score)))).collect::<Vec<_>>())
        }
    } else {
        state.store.list(&auth.user_id, memory_core::memory::ListFilter {
            tags: tags.clone(),
            limit: Some(limit + 1),
            offset: Some(offset),
        }).await.map(|mems| mems.into_iter().map(|m| (m, None)).collect::<Vec<_>>())
    };

    match scored_results {
        Ok(mut items) => {
            let has_more = !is_semantic && items.len() > limit;
            items.truncate(limit);

            let ids: Vec<String> = items.iter().map(|(m, _)| m.id.clone()).collect();
            let votes = state.store.get_vote_counts_batch(&ids).await.unwrap_or_default();
            let cards: Vec<MemoryCard> = items.into_iter().map(|(m, score)| {
                let (helpful, harmful) = votes.get(&m.id).copied().unwrap_or((0, 0));
                let card = MemoryCard::from_memory(m, helpful, harmful);
                match score {
                    Some((rel, conf, rec, total)) => card.with_score(rel, conf, rec, total),
                    None => card,
                }
            }).collect();

            let next_cursor = (offset + cards.len()).to_string();

            CardGridTemplate {
                memories: cards,
                has_more,
                next_cursor,
                query: q.q.unwrap_or_default(),
                tag: tags.join(","),
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
    Extension(auth): Extension<AuthContext>,
    Query(q): Query<MemoryQuery>,
) -> Response {
    match state.store.list_tags(&auth.user_id).await {
        Ok(tags) => {
            let total_count = state.store.count(&auth.user_id).await.unwrap_or(0);
            TagSidebarTemplate {
                tags,
                total_count,
                active_tags: q.tag.as_deref().map(memory_core::tags::parse_comma_separated).unwrap_or_default(),
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
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Query(params): Query<VoteQuery>,
) -> Response {
    if let Err(e) = state.store.vote(&auth.user_id, &id, &params.vote).await {
        tracing::error!("vote failed: {e}");
        return StatusCode::BAD_REQUEST.into_response();
    }

    let (helpful, harmful) = state.store.get_vote_counts(&id).await.unwrap_or((0, 0));
    let memory = match state.store.get(&auth.user_id, &id).await {
        Ok(m) => m,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };

    VoteButtonsTemplate {
        card: MemoryCard::from_memory(memory, helpful, harmful),
    }.into_response()
}

pub async fn delete_memory(
    State(state): State<UiState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    match state.store.delete(&auth.user_id, &id).await {
        Ok(()) => Html("").into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn edit_memory_form(
    State(state): State<UiState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    let memory = match state.store.get(&auth.user_id, &id).await {
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
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    axum::Form(form): axum::Form<UpdateForm>,
) -> Response {
    if let Err(e) = state.store.update(&auth.user_id, &id, &form.content, state.embedder.as_ref()).await {
        tracing::error!("update failed: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    if let Some(ref tags_str) = form.tags {
        let tags = memory_core::tags::parse_comma_separated(tags_str);
        let _ = state.store.set_tags(&auth.user_id, &id, tags).await;
    }

    let memory = match state.store.get(&auth.user_id, &id).await {
        Ok(m) => m,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let (helpful, harmful) = state.store.get_vote_counts(&id).await.unwrap_or((0, 0));

    CardTemplate {
        card: MemoryCard::from_memory(memory, helpful, harmful),
    }.into_response()
}

pub async fn login_page() -> Response {
    LoginTemplate.into_response()
}

pub async fn get_card(
    State(state): State<UiState>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    let memory = match state.store.get(&auth.user_id, &id).await {
        Ok(m) => m,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let (helpful, harmful) = state.store.get_vote_counts(&id).await.unwrap_or((0, 0));
    CardTemplate {
        card: MemoryCard::from_memory(memory, helpful, harmful),
    }.into_response()
}
