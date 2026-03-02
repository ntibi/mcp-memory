use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use subtle::ConstantTimeEq;

#[derive(Clone)]
pub struct ApiKey(pub String);

pub async fn bearer_auth(request: Request, next: Next) -> Result<Response, StatusCode> {
    let expected_key = request
        .extensions()
        .get::<ApiKey>()
        .map(|k| k.0.clone())
        .unwrap_or_default();

    if expected_key.is_empty() {
        return Ok(next.run(request).await);
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        if token.as_bytes().ct_eq(expected_key.as_bytes()).into() {
            return Ok(next.run(request).await);
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}
