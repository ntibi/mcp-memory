use std::sync::Arc;

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Redirect, Response},
};
use memory_core::users::{AuthContext, UserStore};

pub async fn auth_middleware(request: Request, next: Next) -> Response {
    let user_store = match request.extensions().get::<Arc<UserStore>>().cloned() {
        Some(s) => s,
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let is_browser = request.uri().path().starts_with("/ui");

    let raw_key = match extract_key(&request) {
        Some(k) => k,
        None if is_browser => return Redirect::to("/ui/login").into_response(),
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };

    let auth_context = match user_store.authenticate(&raw_key).await {
        Ok(ctx) => ctx,
        Err(_) if is_browser => return Redirect::to("/ui/login").into_response(),
        Err(_) => return StatusCode::UNAUTHORIZED.into_response(),
    };

    let mut request = request;
    request.extensions_mut().insert(auth_context);
    next.run(request).await
}

pub async fn admin_middleware(request: Request, next: Next) -> Result<Response, StatusCode> {
    let auth = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if !auth.is_admin {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}

fn extract_key(request: &Request) -> Option<String> {
    if let Some(auth_header) = request.headers().get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(token) = auth_header.strip_prefix("Bearer ") {
            return Some(token.to_string());
        }
    }

    if let Some(cookie_header) = request.headers().get("cookie").and_then(|v| v.to_str().ok()) {
        for cookie in cookie_header.split(';') {
            let cookie = cookie.trim();
            if let Some(value) = cookie.strip_prefix("api_key=") {
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
    }

    None
}
