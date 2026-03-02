use std::sync::{Arc, LazyLock};

use axum_test::TestServer;
use memory_core::{
    db,
    embed::{Embedder, LocalEmbedder},
    memory::MemoryStore,
    scoring::{Scorer, ScoringConfig},
};
use memory_server::api::{AppState, router};
use serde_json::{Value, json};

static EMBEDDER: LazyLock<Arc<dyn Embedder>> = LazyLock::new(|| {
    Arc::new(LocalEmbedder::new("all-MiniLM-L6-v2").unwrap())
});

async fn test_server() -> TestServer {
    let conn = db::open(":memory:").await.unwrap();
    let scorer = Arc::new(Scorer::new(ScoringConfig::default()));
    let store = Arc::new(MemoryStore::new(conn.clone()));

    let state = AppState {
        store,
        embedder: EMBEDDER.clone(),
        scorer,
        conn,
    };

    let app = router().with_state(state);
    TestServer::new(app).unwrap()
}

#[tokio::test]
async fn should_create_and_get_memory_via_api() {
    let server = test_server().await;

    let create_resp = server
        .post("/memories")
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

    let get_resp = server.get(&format!("/memories/{id}")).await;
    get_resp.assert_status(axum::http::StatusCode::OK);
    let fetched: Value = get_resp.json();
    assert_eq!(fetched["content"], "rust is great for systems programming");
    assert_eq!(fetched["id"], id);
}

#[tokio::test]
async fn should_return_404_when_memory_not_found() {
    let server = test_server().await;

    let resp = server.get("/memories/nonexistent").await;
    resp.assert_status(axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_delete_memory_via_api() {
    let server = test_server().await;

    let create_resp = server
        .post("/memories")
        .json(&json!({
            "content": "to be deleted"
        }))
        .await;

    create_resp.assert_status(axum::http::StatusCode::CREATED);
    let body: Value = create_resp.json();
    let id = body["id"].as_str().unwrap();

    let delete_resp = server.delete(&format!("/memories/{id}")).await;
    delete_resp.assert_status(axum::http::StatusCode::NO_CONTENT);

    let get_resp = server.get(&format!("/memories/{id}")).await;
    get_resp.assert_status(axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_vote_on_memory_via_api() {
    let server = test_server().await;

    let create_resp = server
        .post("/memories")
        .json(&json!({
            "content": "votable memory"
        }))
        .await;

    create_resp.assert_status(axum::http::StatusCode::CREATED);
    let body: Value = create_resp.json();
    let id = body["id"].as_str().unwrap();

    let vote_resp = server
        .post(&format!("/memories/{id}/vote"))
        .json(&json!({ "vote": "helpful" }))
        .await;

    vote_resp.assert_status(axum::http::StatusCode::CREATED);
    let vote_body: Value = vote_resp.json();
    assert_eq!(vote_body["vote"], "helpful");
    assert_eq!(vote_body["memory_id"], id);
}

#[tokio::test]
async fn should_return_ok_on_health() {
    let server = test_server().await;

    let resp = server.get("/health").await;
    resp.assert_status(axum::http::StatusCode::OK);
    let body: Value = resp.json();
    assert_eq!(body["status"], "ok");
}
