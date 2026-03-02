use clap::Parser;
use config::{Config, Environment, File};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub listen_addr: String,
    pub db_path: String,
    pub api_key: String,
    pub embedding: EmbeddingConfig,
    pub scoring: ScoringConfig,
    pub curation: CurationConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub api_url: Option<String>,
    pub dimension: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ScoringConfig {
    pub relevance_weight: f64,
    pub confidence_weight: f64,
    pub recency_weight: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CurationConfig {
    pub interval_secs: u64,
    pub similarity_threshold: f64,
    pub llm: Option<LlmConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LlmConfig {
    pub api_key: String,
    pub model: String,
    pub api_url: String,
}

#[derive(Parser, Debug)]
#[command(name = "memory-server")]
pub struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    pub config: String,
    #[arg(long)]
    pub listen_addr: Option<String>,
    #[arg(long)]
    pub db_path: Option<String>,
}

impl Settings {
    pub fn load(cli: &Cli) -> Result<Self, config::ConfigError> {
        let mut builder = Config::builder()
            .set_default("listen_addr", "127.0.0.1:8000")?
            .set_default("db_path", "memory.db")?
            .set_default("api_key", "")?
            .set_default("embedding.provider", "local")?
            .set_default("embedding.model", "all-MiniLM-L6-v2")?
            .set_default("embedding.dimension", 384_i64)?
            .set_default("scoring.relevance_weight", 0.6)?
            .set_default("scoring.confidence_weight", 0.25)?
            .set_default("scoring.recency_weight", 0.15)?
            .set_default("curation.interval_secs", 3600)?
            .set_default("curation.similarity_threshold", 0.85)?
            .add_source(File::with_name(&cli.config).required(false))
            .add_source(
                Environment::with_prefix("MEMORY")
                    .separator("__")
                    .try_parsing(true),
            );

        if let Some(addr) = &cli.listen_addr {
            builder = builder.set_override("listen_addr", addr.clone())?;
        }
        if let Some(path) = &cli.db_path {
            builder = builder.set_override("db_path", path.clone())?;
        }

        builder.build()?.try_deserialize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_load_defaults_when_no_config_file() {
        let cli = Cli {
            config: "nonexistent.toml".to_string(),
            listen_addr: None,
            db_path: None,
        };

        let settings = Settings::load(&cli).expect("should load defaults");

        assert_eq!(settings.listen_addr, "127.0.0.1:8000");
        assert_eq!(settings.db_path, "memory.db");
        assert_eq!(settings.api_key, "");
        assert_eq!(settings.embedding.provider, "local");
        assert_eq!(settings.embedding.model, "all-MiniLM-L6-v2");
        assert!(settings.embedding.api_key.is_none());
        assert!(settings.embedding.api_url.is_none());
        assert!((settings.scoring.relevance_weight - 0.6).abs() < f64::EPSILON);
        assert!((settings.scoring.confidence_weight - 0.25).abs() < f64::EPSILON);
        assert!((settings.scoring.recency_weight - 0.15).abs() < f64::EPSILON);
        assert_eq!(settings.curation.interval_secs, 3600);
        assert!((settings.curation.similarity_threshold - 0.85).abs() < f64::EPSILON);
        assert!(settings.curation.llm.is_none());
    }

    #[test]
    fn should_override_with_cli_args() {
        let cli = Cli {
            config: "nonexistent.toml".to_string(),
            listen_addr: Some("0.0.0.0:9000".to_string()),
            db_path: Some("/tmp/test.db".to_string()),
        };

        let settings = Settings::load(&cli).expect("should load with overrides");

        assert_eq!(settings.listen_addr, "0.0.0.0:9000");
        assert_eq!(settings.db_path, "/tmp/test.db");
    }
}
