use std::sync::Arc;

use clap::Parser;
use memory_server::{api, auth, config, mcp};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use tracing_subscriber::EnvFilter;

use api::AppState;
use auth::ApiKey;
use config::{Cli, Settings};

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
        },
    ));

    let store = Arc::new(memory_core::memory::MemoryStore::new(conn.clone()));

    let app_state = AppState {
        store: store.clone(),
        embedder: embedder.clone(),
        scorer: scorer.clone(),
        conn: conn.clone(),
    };

    let mcp_service = StreamableHttpService::new(
        {
            let store = store.clone();
            let embedder = embedder.clone();
            let scorer = scorer.clone();
            move || Ok(mcp::MemoryMcp::new(store.clone(), embedder.clone(), scorer.clone()))
        },
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );

    let ui_state = memory_ui::UiState {
        store: store.clone(),
        embedder: embedder.clone(),
        scorer: scorer.clone(),
    };

    let api_key = ApiKey(settings.api_key.clone());

    let authed = axum::Router::new()
        .nest_service("/mcp", mcp_service)
        .layer(axum::middleware::from_fn(auth::bearer_auth))
        .layer(axum::Extension(api_key));

    let router = axum::Router::new()
        .merge(authed)
        .nest("/api/v1", api::router().with_state(app_state))
        .nest("/ui", memory_ui::router().with_state(ui_state))
        .nest_service("/static", memory_ui::static_service())
        .route("/", axum::routing::get(|| async { axum::response::Redirect::to("/ui") }))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&settings.listen_addr).await?;
    tracing::info!("listening on {}", settings.listen_addr);

    axum::serve(listener, router)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.unwrap() })
        .await?;

    Ok(())
}
