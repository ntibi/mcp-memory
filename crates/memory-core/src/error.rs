use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("database error: {0}")]
    Database(#[from] tokio_rusqlite::Error),

    #[error("rusqlite error: {0}")]
    Rusqlite(#[from] rusqlite::Error),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("curation error: {0}")]
    Curation(String),
}

pub type Result<T> = std::result::Result<T, Error>;
