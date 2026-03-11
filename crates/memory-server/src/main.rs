use std::sync::Arc;

use axum::{
    extract::{Request, State},
    response::IntoResponse,
    Extension,
};
use clap::Parser;
use memory_core::users::AuthContext;
use memory_server::{admin, api, auth, config, curation_ui, curation_worker, mcp, scheduler};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, StreamableHttpServerConfig,
    session::local::LocalSessionManager,
};
use tracing_subscriber::EnvFilter;

use api::AppState;
use config::{Cli, Settings};

type McpService = StreamableHttpService<mcp::MemoryMcp, LocalSessionManager>;

async fn mcp_handler(
    Extension(auth): Extension<AuthContext>,
    State(service): State<Arc<McpService>>,
    request: Request,
) -> impl IntoResponse {
    let mut request = request;
    request.extensions_mut().insert(auth);
    service.handle(request).await.into_response()
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

    let scorer = Arc::new(memory_core::scoring::Scorer::new(settings.scoring.clone()));

    let store = Arc::new(memory_core::memory::MemoryStore::new(conn.clone()));

    let user_store = Arc::new(memory_core::users::UserStore::new(conn.clone()));
    if let Some(key) = user_store.bootstrap(&settings.bootstrap_user).await? {
        tracing::info!(
            "bootstrapped admin user '{}' with api key: {}",
            settings.bootstrap_user,
            key.raw_key
        );
    }

    let progress_map = curation_worker::new_progress_map();

    let scheduler = scheduler::Scheduler::spawn(
        conn.clone(),
        store.clone(),
        progress_map.clone(),
    );

    let app_state = AppState {
        store: store.clone(),
        embedder: embedder.clone(),
        scorer: scorer.clone(),
        conn: conn.clone(),
        user_store: user_store.clone(),
        scheduler,
        progress_map,
    };

    let mcp_service = {
        let store = store.clone();
        let embedder = embedder.clone();
        let scorer = scorer.clone();
        Arc::new(StreamableHttpService::new(
            move || Ok(mcp::MemoryMcp::new(store.clone(), embedder.clone(), scorer.clone())),
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig {
                stateful_mode: false,
                json_response: true,
                ..Default::default()
            },
        ))
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

    let curation_ui_state = curation_ui::CurationUiState {
        conn: conn.clone(),
        store: store.clone(),
        embedder: embedder.clone(),
        scheduler: app_state.scheduler.clone(),
        progress_map: app_state.progress_map.clone(),
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
            axum::routing::post(mcp_handler)
                .with_state(mcp_service),
        )
        .nest("/api/v1", api::router().with_state(app_state))
        .nest("/ui/curation", curation_ui::router().with_state(curation_ui_state))
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
