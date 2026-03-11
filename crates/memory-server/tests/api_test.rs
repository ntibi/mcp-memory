use std::sync::{Arc, LazyLock};

use axum_test::TestServer;
use memory_core::{
    curation,
    db,
    embed::{Embedder, LocalEmbedder},
    memory::MemoryStore,
    scoring::{Scorer, ScoringConfig},
    users::UserStore,
};
use memory_server::{api::{self, AppState}, auth, curation_worker, scheduler};
use serde_json::{Value, json};

static EMBEDDER: LazyLock<Arc<dyn Embedder>> = LazyLock::new(|| {
    Arc::new(LocalEmbedder::new("all-MiniLM-L6-v2").unwrap())
});

struct TestContext {
    server: TestServer,
    api_key: String,
    user_id: String,
    conn: tokio_rusqlite::Connection,
    store: Arc<MemoryStore>,
}

async fn test_server() -> TestContext {
    let conn = db::open(":memory:").await.unwrap();
    let scorer = Arc::new(Scorer::new(ScoringConfig::default()));
    let store = Arc::new(MemoryStore::new(conn.clone()));
    let user_store = Arc::new(UserStore::new(conn.clone()));

    let key = user_store.bootstrap("test-admin").await.unwrap().unwrap();

    let progress_map = curation_worker::new_progress_map();
    let scheduler = scheduler::Scheduler::spawn(
        conn.clone(),
        store.clone(),
        progress_map.clone(),
    );

    let state = AppState {
        store: store.clone(),
        embedder: EMBEDDER.clone(),
        scorer,
        conn: conn.clone(),
        user_store: user_store.clone(),
        scheduler,
        progress_map,
    };

    let app = api::router()
        .with_state(state)
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        .layer(axum::Extension(user_store));

    TestContext {
        server: TestServer::new(app).unwrap(),
        user_id: key.api_key.user_id,
        api_key: key.raw_key,
        conn,
        store,
    }
}

#[tokio::test]
async fn should_create_and_get_memory_via_api() {
    let ctx = test_server().await;

    let create_resp = ctx.server
        .post("/memories")
        .authorization_bearer(&ctx.api_key)
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

    let get_resp = ctx.server
        .get(&format!("/memories/{id}"))
        .authorization_bearer(&ctx.api_key)
        .await;
    get_resp.assert_status(axum::http::StatusCode::OK);
    let fetched: Value = get_resp.json();
    assert_eq!(fetched["content"], "rust is great for systems programming");
    assert_eq!(fetched["id"], id);
}

#[tokio::test]
async fn should_return_404_when_memory_not_found() {
    let ctx = test_server().await;

    let resp = ctx.server
        .get("/memories/nonexistent")
        .authorization_bearer(&ctx.api_key)
        .await;
    resp.assert_status(axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_delete_memory_via_api() {
    let ctx = test_server().await;

    let create_resp = ctx.server
        .post("/memories")
        .authorization_bearer(&ctx.api_key)
        .json(&json!({ "content": "to be deleted" }))
        .await;

    create_resp.assert_status(axum::http::StatusCode::CREATED);
    let body: Value = create_resp.json();
    let id = body["id"].as_str().unwrap();

    let delete_resp = ctx.server
        .delete(&format!("/memories/{id}"))
        .authorization_bearer(&ctx.api_key)
        .await;
    delete_resp.assert_status(axum::http::StatusCode::NO_CONTENT);

    let get_resp = ctx.server
        .get(&format!("/memories/{id}"))
        .authorization_bearer(&ctx.api_key)
        .await;
    get_resp.assert_status(axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_vote_on_memory_via_api() {
    let ctx = test_server().await;

    let create_resp = ctx.server
        .post("/memories")
        .authorization_bearer(&ctx.api_key)
        .json(&json!({ "content": "votable memory" }))
        .await;

    create_resp.assert_status(axum::http::StatusCode::CREATED);
    let body: Value = create_resp.json();
    let id = body["id"].as_str().unwrap();

    let vote_resp = ctx.server
        .post(&format!("/memories/{id}/vote"))
        .authorization_bearer(&ctx.api_key)
        .json(&json!({ "vote": "helpful" }))
        .await;

    vote_resp.assert_status(axum::http::StatusCode::CREATED);
    let vote_body: Value = vote_resp.json();
    assert_eq!(vote_body["vote"], "helpful");
    assert_eq!(vote_body["memory_id"], id);
}

#[tokio::test]
async fn should_return_ok_on_health() {
    let ctx = test_server().await;

    let resp = ctx.server
        .get("/health")
        .authorization_bearer(&ctx.api_key)
        .await;
    resp.assert_status(axum::http::StatusCode::OK);
    let body: Value = resp.json();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn should_reject_unauthenticated_requests() {
    let ctx = test_server().await;

    let resp = ctx.server.get("/memories").await;
    resp.assert_status(axum::http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn should_reject_invalid_api_key() {
    let ctx = test_server().await;

    let resp = ctx.server
        .get("/memories")
        .authorization_bearer("mem_invalid")
        .await;
    resp.assert_status(axum::http::StatusCode::UNAUTHORIZED);
}

// Curation integration tests

#[tokio::test]
async fn should_get_default_curation_settings() {
    let ctx = test_server().await;

    let resp = ctx.server
        .get("/curation/settings")
        .authorization_bearer(&ctx.api_key)
        .await;
    resp.assert_status(axum::http::StatusCode::OK);

    let body: Value = resp.json();
    assert_eq!(body["enabled"], false);
    assert_eq!(body["similarity_threshold"], 0.85);
    assert!(body["api_key"].is_null());
}

#[tokio::test]
async fn should_update_curation_settings() {
    let ctx = test_server().await;

    let resp = ctx.server
        .put("/curation/settings")
        .authorization_bearer(&ctx.api_key)
        .json(&json!({
            "api_key": "sk-ant-test-key",
            "similarity_threshold": 0.9,
            "model": "claude-haiku-4-5",
            "enabled": true,
        }))
        .await;
    resp.assert_status(axum::http::StatusCode::OK);

    let body: Value = resp.json();
    assert_eq!(body["enabled"], true);
    assert_eq!(body["similarity_threshold"], 0.9);
    assert_eq!(body["model"], "claude-haiku-4-5");

    let get_resp = ctx.server
        .get("/curation/settings")
        .authorization_bearer(&ctx.api_key)
        .await;
    let settings: Value = get_resp.json();
    assert_eq!(settings["enabled"], true);
    assert_eq!(settings["similarity_threshold"], 0.9);
    assert_eq!(settings["model"], "claude-haiku-4-5");
}

#[tokio::test]
async fn should_list_empty_curation_runs() {
    let ctx = test_server().await;

    let resp = ctx.server
        .get("/curation/runs")
        .authorization_bearer(&ctx.api_key)
        .await;
    resp.assert_status(axum::http::StatusCode::OK);

    let body: Value = resp.json();
    assert!(body.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn should_apply_merge_suggestion_via_api() {
    let ctx = test_server().await;

    let m1 = ctx.server
        .post("/memories")
        .authorization_bearer(&ctx.api_key)
        .json(&json!({
            "content": "rust has great error handling",
            "tags": ["rust", "errors"]
        }))
        .await;
    m1.assert_status(axum::http::StatusCode::CREATED);
    let id1 = m1.json::<Value>()["id"].as_str().unwrap().to_string();

    let m2 = ctx.server
        .post("/memories")
        .authorization_bearer(&ctx.api_key)
        .json(&json!({
            "content": "rust error handling uses Result type",
            "tags": ["rust", "result"]
        }))
        .await;
    m2.assert_status(axum::http::StatusCode::CREATED);
    let id2 = m2.json::<Value>()["id"].as_str().unwrap().to_string();

    let memory_ids = vec![id1.clone(), id2.clone()];
    let suggestion_json = json!({
        "action": "merge",
        "memory_ids": &memory_ids,
        "content": "rust has great error handling using the Result type",
        "tags": ["rust", "errors", "result"],
        "reasoning": "these two memories cover the same topic"
    })
    .to_string();
    let suggestion_id = curation::store_suggestion(
        &ctx.conn,
        &ctx.user_id,
        "merge",
        &memory_ids,
        &suggestion_json,
        "llm",
    )
    .await
    .unwrap();

    let apply_resp = ctx.server
        .post(&format!("/curation/suggestions/{suggestion_id}/apply"))
        .authorization_bearer(&ctx.api_key)
        .await;
    apply_resp.assert_status(axum::http::StatusCode::OK);
    let apply_body: Value = apply_resp.json();
    assert_eq!(apply_body["status"], "applied");

    let get1 = ctx.server
        .get(&format!("/memories/{id1}"))
        .authorization_bearer(&ctx.api_key)
        .await;
    get1.assert_status(axum::http::StatusCode::NOT_FOUND);

    let get2 = ctx.server
        .get(&format!("/memories/{id2}"))
        .authorization_bearer(&ctx.api_key)
        .await;
    get2.assert_status(axum::http::StatusCode::NOT_FOUND);

    let all = ctx.server
        .get("/memories")
        .authorization_bearer(&ctx.api_key)
        .await;
    let items: Vec<Value> = all.json();
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0]["content"],
        "rust has great error handling using the Result type"
    );
}

#[tokio::test]
async fn should_dismiss_suggestion_via_api() {
    let ctx = test_server().await;

    let m1 = ctx.server
        .post("/memories")
        .authorization_bearer(&ctx.api_key)
        .json(&json!({ "content": "memory one", "tags": ["a"] }))
        .await;
    let id1 = m1.json::<Value>()["id"].as_str().unwrap().to_string();

    let suggestion_json = json!({
        "action": "update",
        "content": "updated memory one",
        "tags": ["a", "b"],
        "reasoning": "add missing tag"
    })
    .to_string();

    let suggestion_id = curation::store_suggestion(
        &ctx.conn,
        &ctx.user_id,
        "merge",
        &[id1.clone()],
        &suggestion_json,
        "llm",
    )
    .await
    .unwrap();

    let dismiss_resp = ctx.server
        .post(&format!("/curation/suggestions/{suggestion_id}/dismiss"))
        .authorization_bearer(&ctx.api_key)
        .await;
    dismiss_resp.assert_status(axum::http::StatusCode::OK);

    let suggestions = ctx.server
        .get("/curation/suggestions?status=pending")
        .authorization_bearer(&ctx.api_key)
        .await;
    let body: Value = suggestions.json();
    assert!(body.as_array().unwrap().is_empty());

    let dismissed = ctx.server
        .get("/curation/suggestions?status=dismissed")
        .authorization_bearer(&ctx.api_key)
        .await;
    let body: Value = dismissed.json();
    assert_eq!(body.as_array().unwrap().len(), 1);
}
