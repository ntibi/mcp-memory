use std::sync::{Arc, LazyLock};

use axum_test::TestServer;
use memory_core::{
    db,
    embed::{Embedder, LocalEmbedder},
    memory::MemoryStore,
    scoring::{Scorer, ScoringConfig},
    users::UserStore,
};
use memory_server::{api::{self, AppState}, auth};
use serde_json::{Value, json};

static EMBEDDER: LazyLock<Arc<dyn Embedder>> = LazyLock::new(|| {
    Arc::new(LocalEmbedder::new("all-MiniLM-L6-v2").unwrap())
});

async fn test_server() -> (TestServer, String) {
    let conn = db::open(":memory:").await.unwrap();
    let scorer = Arc::new(Scorer::new(ScoringConfig::default()));
    let store = Arc::new(MemoryStore::new(conn.clone()));
    let user_store = Arc::new(UserStore::new(conn.clone()));

    let key = user_store.bootstrap("test-admin").await.unwrap().unwrap();

    let state = AppState {
        store,
        embedder: EMBEDDER.clone(),
        scorer,
        conn,
        user_store: user_store.clone(),
    };

    let app = api::router()
        .with_state(state)
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        .layer(axum::Extension(user_store));

    (TestServer::new(app).unwrap(), key.raw_key)
}

#[tokio::test]
async fn should_create_and_get_memory_via_api() {
    let (server, api_key) = test_server().await;

    let create_resp = server
        .post("/memories")
        .authorization_bearer(&api_key)
        .json(&json!({
            "content": "rust is great for systems programming",
            "tags": ["rust", "systems"]
        }))
        .await;

    create_resp.assert_status(axum::http::StatusCode::CREATED);
    let body: Value = create_resp.json();
    let id = body["id"].as_str().unwrap();
    assert_eq!(body["content"], "rust is great for systems programming");
    assert!(!id.is_empty());

    let get_resp = server
        .get(&format!("/memories/{id}"))
        .authorization_bearer(&api_key)
        .await;
    get_resp.assert_status(axum::http::StatusCode::OK);
    let fetched: Value = get_resp.json();
    assert_eq!(fetched["content"], "rust is great for systems programming");
    assert_eq!(fetched["id"], id);
}

#[tokio::test]
async fn should_return_404_when_memory_not_found() {
    let (server, api_key) = test_server().await;

    let resp = server
        .get("/memories/nonexistent")
        .authorization_bearer(&api_key)
        .await;
    resp.assert_status(axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_delete_memory_via_api() {
    let (server, api_key) = test_server().await;

    let create_resp = server
        .post("/memories")
        .authorization_bearer(&api_key)
        .json(&json!({ "content": "to be deleted" }))
        .await;

    create_resp.assert_status(axum::http::StatusCode::CREATED);
    let body: Value = create_resp.json();
    let id = body["id"].as_str().unwrap();

    let delete_resp = server
        .delete(&format!("/memories/{id}"))
        .authorization_bearer(&api_key)
        .await;
    delete_resp.assert_status(axum::http::StatusCode::NO_CONTENT);

    let get_resp = server
        .get(&format!("/memories/{id}"))
        .authorization_bearer(&api_key)
        .await;
    get_resp.assert_status(axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_vote_on_memory_via_api() {
    let (server, api_key) = test_server().await;

    let create_resp = server
        .post("/memories")
        .authorization_bearer(&api_key)
        .json(&json!({ "content": "votable memory" }))
        .await;

    create_resp.assert_status(axum::http::StatusCode::CREATED);
    let body: Value = create_resp.json();
    let id = body["id"].as_str().unwrap();

    let vote_resp = server
        .post(&format!("/memories/{id}/vote"))
        .authorization_bearer(&api_key)
        .json(&json!({ "vote": "helpful" }))
        .await;

    vote_resp.assert_status(axum::http::StatusCode::CREATED);
    let vote_body: Value = vote_resp.json();
    assert_eq!(vote_body["vote"], "helpful");
    assert_eq!(vote_body["memory_id"], id);
}

#[tokio::test]
async fn should_return_ok_on_health() {
    let (server, api_key) = test_server().await;

    let resp = server
        .get("/health")
        .authorization_bearer(&api_key)
        .await;
    resp.assert_status(axum::http::StatusCode::OK);
    let body: Value = resp.json();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn should_reject_unauthenticated_requests() {
    let (server, _) = test_server().await;

    let resp = server.get("/memories").await;
    resp.assert_status(axum::http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn should_reject_invalid_api_key() {
    let (server, _) = test_server().await;

    let resp = server
        .get("/memories")
        .authorization_bearer("mem_invalid")
        .await;
    resp.assert_status(axum::http::StatusCode::UNAUTHORIZED);
}
