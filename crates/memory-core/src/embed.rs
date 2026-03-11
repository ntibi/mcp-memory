use std::sync::{Mutex, OnceLock};

use crate::error::{Error, Result};

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

pub struct LocalEmbedder {
    model: OnceLock<std::result::Result<Mutex<fastembed::TextEmbedding>, String>>,
    model_type: fastembed::EmbeddingModel,
    dimension: usize,
}

impl LocalEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let model_type = match model_name {
            "all-MiniLM-L6-v2" => fastembed::EmbeddingModel::AllMiniLML6V2,
            other => return Err(Error::Embedding(format!("unsupported model: {other}"))),
        };
        Ok(Self {
            model: OnceLock::new(),
            model_type,
            dimension: 384,
        })
    }

    fn get_model(&self) -> Result<&Mutex<fastembed::TextEmbedding>> {
        self.model
            .get_or_init(|| {
                tracing::info!("loading embedding model");
                fastembed::TextEmbedding::try_new(
                    fastembed::InitOptions::new(self.model_type.clone())
                        .with_show_download_progress(true),
                )
                .map(Mutex::new)
                .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| Error::Embedding(e.clone()))
    }
}

impl Embedder for LocalEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut model = self
            .get_model()?
            .lock()
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let results = model
            .embed(vec![text], None)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("no embedding returned".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut model = self
            .get_model()?
            .lock()
            .map_err(|e| Error::Embedding(e.to_string()))?;
        model
            .embed(texts.to_vec(), None)
            .map_err(|e| Error::Embedding(e.to_string()))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

pub struct RemoteEmbedder {
    client: reqwest::Client,
    api_url: String,
    api_key: String,
    model: String,
    dimension: usize,
}

#[derive(serde::Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(serde::Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(serde::Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl RemoteEmbedder {
    pub fn new(api_url: &str, api_key: &str, model: &str, dimension: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            dimension,
        }
    }
}

impl Embedder for RemoteEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let client = self.client.clone();
        let url = format!("{}/v1/embeddings", self.api_url);
        let req = EmbeddingRequest {
            input: vec![text.to_string()],
            model: self.model.clone(),
        };
        let api_key = self.api_key.clone();

        std::thread::scope(|s| {
            s.spawn(|| {
                rt.block_on(async {
                    let resp = client
                        .post(&url)
                        .bearer_auth(&api_key)
                        .json(&req)
                        .send()
                        .await
                        .map_err(|e| Error::Embedding(e.to_string()))?
                        .error_for_status()
                        .map_err(|e| Error::Embedding(e.to_string()))?;
                    let body: EmbeddingResponse = resp
                        .json()
                        .await
                        .map_err(|e| Error::Embedding(e.to_string()))?;
                    body.data
                        .into_iter()
                        .next()
                        .map(|d| d.embedding)
                        .ok_or_else(|| Error::Embedding("no embedding returned".into()))
                })
            })
            .join()
            .unwrap()
        })
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let client = self.client.clone();
        let url = format!("{}/v1/embeddings", self.api_url);
        let req = EmbeddingRequest {
            input: texts.iter().map(|s| s.to_string()).collect(),
            model: self.model.clone(),
        };
        let api_key = self.api_key.clone();

        std::thread::scope(|s| {
            s.spawn(|| {
                rt.block_on(async {
                    let resp = client
                        .post(&url)
                        .bearer_auth(&api_key)
                        .json(&req)
                        .send()
                        .await
                        .map_err(|e| Error::Embedding(e.to_string()))?
                        .error_for_status()
                        .map_err(|e| Error::Embedding(e.to_string()))?;
                    let body: EmbeddingResponse = resp
                        .json()
                        .await
                        .map_err(|e| Error::Embedding(e.to_string()))?;
                    Ok(body.data.into_iter().map(|d| d.embedding).collect())
                })
            })
            .join()
            .unwrap()
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedder() -> LocalEmbedder {
        LocalEmbedder::new("all-MiniLM-L6-v2").expect("failed to create embedder")
    }

    #[test]
    fn should_produce_384_dim_embedding_when_local_model() {
        let e = embedder();
        let vec = e.embed("hello world").unwrap();
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn should_produce_normalized_vectors_when_embedded() {
        let e = embedder();
        let vec = e.embed("hello world").unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected unit norm, got {norm}"
        );
    }

    #[test]
    fn should_embed_batch_when_multiple_texts() {
        let e = embedder();
        let vecs = e.embed_batch(&["hello", "world"]).unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 384);
        assert_eq!(vecs[1].len(), 384);
    }
}
