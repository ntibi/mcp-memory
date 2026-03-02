use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Serialize)]
pub struct DuplicateCandidate {
    pub memory_id_a: String,
    pub memory_id_b: String,
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurationSuggestion {
    pub id: String,
    pub suggestion_type: String,
    pub memory_ids: Vec<String>,
    pub suggestion: String,
    pub source: String,
    pub status: String,
    pub created_at: String,
}

pub async fn find_duplicates(conn: &Connection, threshold: f64) -> Result<Vec<DuplicateCandidate>> {
    conn.call(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT memory_id, embedding FROM memory_embeddings",
        )?;

        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, blob))
        })?;

        let entries: Vec<(String, Vec<f32>)> = rows
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .filter_map(|(id, blob)| {
                if blob.len() % 4 != 0 {
                    return None;
                }
                let floats: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Some((id, floats))
            })
            .collect();

        let mut candidates = Vec::new();
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let sim = cosine_similarity(&entries[i].1, &entries[j].1);
                if sim > threshold {
                    candidates.push(DuplicateCandidate {
                        memory_id_a: entries[i].0.clone(),
                        memory_id_b: entries[j].0.clone(),
                        similarity: sim,
                    });
                }
            }
        }

        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates)
    })
    .await
    .map_err(Error::Database)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub async fn store_suggestion(
    conn: &Connection,
    suggestion_type: &str,
    memory_ids: &[String],
    suggestion: &str,
    source: &str,
) -> Result<String> {
    let id = ulid::Ulid::new().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    let memory_ids_json = serde_json::to_string(memory_ids)
        .map_err(|e| Error::Curation(format!("failed to serialize memory_ids: {e}")))?;

    let id_clone = id.clone();
    let suggestion_type = suggestion_type.to_string();
    let suggestion = suggestion.to_string();
    let source = source.to_string();

    conn.call(move |conn| {
        conn.execute(
            "INSERT INTO curation_suggestions (id, type, memory_ids, suggestion, source, status, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, 'pending', ?6)",
            rusqlite::params![id_clone, suggestion_type, memory_ids_json, suggestion, source, now],
        )?;
        Ok(())
    })
    .await
    .map_err(Error::Database)?;

    Ok(id)
}

pub async fn list_suggestions(
    conn: &Connection,
    status: Option<&str>,
) -> Result<Vec<CurationSuggestion>> {
    let status = status.map(|s| s.to_string());

    conn.call(move |conn| {
        let suggestions = if let Some(ref status) = status {
            let mut stmt = conn.prepare(
                "SELECT id, type, memory_ids, suggestion, source, status, created_at \
                 FROM curation_suggestions WHERE status = ?1 ORDER BY created_at DESC",
            )?;
            let rows = stmt.query_map(rusqlite::params![status], parse_suggestion_row)?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, type, memory_ids, suggestion, source, status, created_at \
                 FROM curation_suggestions ORDER BY created_at DESC",
            )?;
            let rows = stmt.query_map([], parse_suggestion_row)?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(suggestions)
    })
    .await
    .map_err(Error::Database)
}

fn parse_suggestion_row(row: &rusqlite::Row) -> rusqlite::Result<CurationSuggestion> {
    let memory_ids_raw: String = row.get(2)?;
    let memory_ids: Vec<String> = serde_json::from_str(&memory_ids_raw).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(
            2,
            rusqlite::types::Type::Text,
            Box::new(e),
        )
    })?;
    Ok(CurationSuggestion {
        id: row.get(0)?,
        suggestion_type: row.get(1)?,
        memory_ids,
        suggestion: row.get(3)?,
        source: row.get(4)?,
        status: row.get(5)?,
        created_at: row.get(6)?,
    })
}

pub async fn update_suggestion_status(
    conn: &Connection,
    id: &str,
    status: &str,
) -> Result<()> {
    if status != "applied" && status != "dismissed" {
        return Err(Error::InvalidInput(format!(
            "status must be 'applied' or 'dismissed', got '{status}'"
        )));
    }

    let id = id.to_string();
    let status = status.to_string();

    let changed = conn
        .call(move |conn| {
            let changed = conn.execute(
                "UPDATE curation_suggestions SET status = ?1 WHERE id = ?2",
                rusqlite::params![status, id],
            )?;
            Ok(changed)
        })
        .await
        .map_err(Error::Database)?;

    if changed == 0 {
        return Err(Error::NotFound(format!("suggestion not found")));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::LocalEmbedder;
    use crate::memory::{CreateMemory, MemoryStore};

    #[tokio::test]
    async fn should_detect_duplicate_memories() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        store.create(CreateMemory { content: "rust is a systems programming language".into(), tags: vec![] }, &embedder).await.unwrap();
        store.create(CreateMemory { content: "rust is a systems-level programming language".into(), tags: vec![] }, &embedder).await.unwrap();
        store.create(CreateMemory { content: "chocolate cake recipe with frosting".into(), tags: vec![] }, &embedder).await.unwrap();

        let dupes = find_duplicates(&conn, 0.8).await.unwrap();
        assert!(!dupes.is_empty());
        assert!(dupes[0].similarity > 0.8);
    }

    #[tokio::test]
    async fn should_not_find_duplicates_when_dissimilar() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        store.create(CreateMemory { content: "rust is a systems programming language".into(), tags: vec![] }, &embedder).await.unwrap();
        store.create(CreateMemory { content: "chocolate cake recipe with frosting".into(), tags: vec![] }, &embedder).await.unwrap();

        let dupes = find_duplicates(&conn, 0.9).await.unwrap();
        assert!(dupes.is_empty());
    }

    #[tokio::test]
    async fn should_store_and_list_suggestions() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let ids = vec!["id_a".to_string(), "id_b".to_string()];
        let suggestion_id = store_suggestion(&conn, "merge", &ids, "merge these two", "auto").await.unwrap();
        assert!(!suggestion_id.is_empty());

        let all = list_suggestions(&conn, None).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].suggestion_type, "merge");
        assert_eq!(all[0].memory_ids, ids);
        assert_eq!(all[0].status, "pending");

        let pending = list_suggestions(&conn, Some("pending")).await.unwrap();
        assert_eq!(pending.len(), 1);

        let applied = list_suggestions(&conn, Some("applied")).await.unwrap();
        assert!(applied.is_empty());
    }

    #[tokio::test]
    async fn should_update_suggestion_status() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let ids = vec!["id_a".to_string()];
        let suggestion_id = store_suggestion(&conn, "prune", &ids, "remove stale memory", "llm").await.unwrap();

        update_suggestion_status(&conn, &suggestion_id, "applied").await.unwrap();

        let all = list_suggestions(&conn, Some("applied")).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, suggestion_id);
    }

    #[tokio::test]
    async fn should_reject_invalid_suggestion_status() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let ids = vec!["id_a".to_string()];
        let suggestion_id = store_suggestion(&conn, "prune", &ids, "remove it", "auto").await.unwrap();

        let result = update_suggestion_status(&conn, &suggestion_id, "invalid").await;
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn should_return_not_found_for_missing_suggestion() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let result = update_suggestion_status(&conn, "nonexistent", "applied").await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }
}
