use std::sync::Arc;

use axum::{
    extract::{Request, State},
    response::IntoResponse,
    Extension,
};
use clap::Parser;
use memory_core::users::AuthContext;
use memory_server::{admin, api, auth, config, mcp};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use tower::ServiceExt;
use tracing_subscriber::EnvFilter;

use api::AppState;
use config::{Cli, Settings};

#[derive(Clone)]
struct McpState {
    store: Arc<memory_core::memory::MemoryStore>,
    embedder: Arc<dyn memory_core::embed::Embedder>,
    scorer: Arc<memory_core::scoring::Scorer>,
}

async fn mcp_handler(
    Extension(auth): Extension<AuthContext>,
    State(state): State<McpState>,
    request: Request,
) -> impl IntoResponse {
    let service = StreamableHttpService::new(
        {
            let store = state.store.clone();
            let embedder = state.embedder.clone();
            let scorer = state.scorer.clone();
            let user_id = auth.user_id.clone();
            move || {
                Ok(mcp::MemoryMcp::new(
                    store.clone(),
                    embedder.clone(),
                    scorer.clone(),
                    user_id.clone(),
                ))
            }
        },
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );
    match service.oneshot(request).await {
        Ok(resp) => resp.into_response(),
        Err(e) => {
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let settings = Settings::load(&cli).expect("failed to load configuration");

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .init();

    tracing::info!("opening database at {}", settings.db_path);
    let conn = memory_core::db::open(&settings.db_path).await?;

    let embedder: Arc<dyn memory_core::embed::Embedder> = match settings.embedding.provider.as_str()
    {
        "local" => Arc::new(memory_core::embed::LocalEmbedder::new(
            &settings.embedding.model,
        )?),
        "openai" => Arc::new(memory_core::embed::RemoteEmbedder::new(
            settings
                .embedding
                .api_url
                .as_deref()
                .unwrap_or("https://api.openai.com"),
            settings.embedding.api_key.as_deref().unwrap_or(""),
            &settings.embedding.model,
            settings.embedding.dimension.unwrap_or(384),
        )),
        other => anyhow::bail!("unsupported embedding provider: {other}"),
    };

    let scorer = Arc::new(memory_core::scoring::Scorer::new(
        memory_core::scoring::ScoringConfig {
            relevance_weight: settings.scoring.relevance_weight,
            confidence_weight: settings.scoring.confidence_weight,
            recency_weight: settings.scoring.recency_weight,
            recency_half_life_days: settings.scoring.recency_half_life_days,
        },
    ));

    let store = Arc::new(memory_core::memory::MemoryStore::new(conn.clone()));

    let user_store = Arc::new(memory_core::users::UserStore::new(conn.clone()));
    if let Some(key) = user_store.bootstrap(&settings.bootstrap_user).await? {
        tracing::info!(
            "bootstrapped admin user '{}' with api key: {}",
            settings.bootstrap_user,
            key.raw_key
        );
    }

    let app_state = AppState {
        store: store.clone(),
        embedder: embedder.clone(),
        scorer: scorer.clone(),
        conn: conn.clone(),
        user_store: user_store.clone(),
    };

    let mcp_state = McpState {
        store: store.clone(),
        embedder: embedder.clone(),
        scorer: scorer.clone(),
    };

    let ui_state = memory_ui::UiState {
        store: store.clone(),
        embedder: embedder.clone(),
        scorer: scorer.clone(),
    };

    let admin_ui_state = memory_ui::AdminUiState {
        user_store: user_store.clone(),
        store: store.clone(),
    };

    let admin = axum::Router::new()
        .nest("/api/v1/admin", admin::router().with_state(app_state.clone()))
        .nest("/ui/admin", memory_ui::admin_router().with_state(admin_ui_state))
        .layer(axum::middleware::from_fn(auth::admin_middleware))
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        .layer(axum::Extension(user_store.clone()));

    let authed = axum::Router::new()
        .route(
            "/mcp",
            axum::routing::post(mcp_handler).with_state(mcp_state),
        )
        .nest("/api/v1", api::router().with_state(app_state))
        .nest("/ui", memory_ui::router().with_state(ui_state))
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        .layer(axum::Extension(user_store.clone()));

    let router = axum::Router::new()
        .merge(admin)
        .merge(authed)
        .nest_service("/static", memory_ui::static_service())
        .route(
            "/ui/login",
            axum::routing::get(memory_ui::handlers::login_page),
        )
        .route(
            "/",
            axum::routing::get(|| async { axum::response::Redirect::to("/ui") }),
        )
        .route(
            "/version",
            axum::routing::get(|| async { option_env!("MEMORY_GIT_SHA").unwrap_or("unknown") }),
        )
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&settings.listen_addr).await?;
    tracing::info!("listening on {}", settings.listen_addr);

    axum::serve(listener, router)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.unwrap() })
        .await?;

    Ok(())
}
