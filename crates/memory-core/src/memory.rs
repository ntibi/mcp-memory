use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;
use zerocopy::AsBytes;

use crate::embed::Embedder;
use crate::error::{Error, Result};
use crate::scoring::Scorer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub user_id: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMemory {
    pub content: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListFilter {
    pub tags: Vec<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub id: String,
    pub memory_id: String,
    pub vote: String,
    pub voted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub relevance: f64,
    pub confidence: f64,
    pub score: f64,
}

pub struct MemoryStore {
    conn: Connection,
}

impl MemoryStore {
    pub fn new(conn: Connection) -> Self {
        Self { conn }
    }

    pub async fn create(&self, user_id: &str, input: CreateMemory, embedder: &dyn Embedder) -> Result<Memory> {
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now();
        let embedding = embedder.embed(&input.content)?;
        let embedding_bytes: Vec<u8> = embedding.as_bytes().to_vec();

        let memory = Memory {
            id: id.clone(),
            user_id: user_id.to_string(),
            content: input.content.clone(),
            created_at: now,
            updated_at: now,
            tags: input.tags.clone(),
        };

        let tags = input.tags.clone();
        let content = input.content;
        let id_clone = id.clone();
        let user_id = user_id.to_string();
        let created_at = now.to_rfc3339();
        let updated_at = now.to_rfc3339();

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                tx.execute(
                    "INSERT INTO memories (id, user_id, content, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![id_clone, user_id, content, created_at, updated_at],
                )?;

                tx.execute(
                    "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?1, ?2)",
                    rusqlite::params![id_clone, embedding_bytes],
                )?;

                for tag in &tags {
                    tx.execute(
                        "INSERT INTO memory_tags (memory_id, tag) VALUES (?1, ?2)",
                        rusqlite::params![id_clone, tag],
                    )?;
                }

                tx.commit()?;
                Ok(())
            })
            .await
            .map_err(Error::Database)?;

        Ok(memory)
    }

    pub async fn get(&self, user_id: &str, id: &str) -> Result<Memory> {
        let id = id.to_string();
        let user_id = user_id.to_string();
        let id_for_err = id.clone();
        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, user_id, content, created_at, updated_at FROM memories WHERE id = ?1 AND user_id = ?2",
                )?;
                let memory = stmt
                    .query_row(rusqlite::params![id, user_id], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                        ))
                    })
                    .optional()?;

                let (id, user_id, content, created_at, updated_at) = memory
                    .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)?;

                let tags = load_tags(conn, &id)?;

                let created_at = parse_datetime(&created_at)?;
                let updated_at = parse_datetime(&updated_at)?;

                Ok(Memory {
                    id,
                    user_id,
                    content,
                    created_at,
                    updated_at,
                    tags,
                })
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound(format!("memory not found: {}", id_for_err))
                }
                other => Error::Database(other),
            })
    }

    pub async fn count(&self, user_id: &str) -> Result<usize> {
        let user_id = user_id.to_string();
        self.conn
            .call(move |conn| {
                let count = conn.query_row(
                    "SELECT COUNT(*) FROM memories WHERE user_id = ?1",
                    rusqlite::params![user_id],
                    |row| row.get(0),
                )?;
                Ok(count)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn list(&self, user_id: &str, filter: ListFilter) -> Result<Vec<Memory>> {
        let limit = filter.limit.unwrap_or(50);
        let offset = filter.offset.unwrap_or(0);
        let tags = filter.tags;
        let user_id = user_id.to_string();

        self.conn
            .call(move |conn| {
                let ids: Vec<String> = if tags.is_empty() {
                    let mut stmt = conn.prepare(
                        "SELECT id FROM memories \
                         WHERE user_id = ?1 \
                         ORDER BY created_at DESC \
                         LIMIT ?2 OFFSET ?3",
                    )?;
                    let rows = stmt.query_map(
                        rusqlite::params![user_id, limit as i64, offset as i64],
                        |row| row.get(0),
                    )?;
                    rows.collect::<std::result::Result<Vec<_>, _>>()?
                } else {
                    let mut param_idx = 1;
                    let user_id_param = param_idx;
                    param_idx += 1;
                    let tag_start = param_idx;
                    let placeholders: Vec<String> = (tag_start..tag_start + tags.len()).map(|i| format!("?{i}")).collect();
                    param_idx += tags.len();
                    let tag_count_param = param_idx;
                    param_idx += 1;
                    let limit_param = param_idx;
                    param_idx += 1;
                    let offset_param = param_idx;
                    let sql = format!(
                        "SELECT m.id FROM memories m \
                         JOIN memory_tags mt ON m.id = mt.memory_id \
                         WHERE m.user_id = ?{user_id_param} AND mt.tag IN ({}) \
                         GROUP BY m.id \
                         HAVING COUNT(DISTINCT mt.tag) = ?{tag_count_param} \
                         ORDER BY m.created_at DESC \
                         LIMIT ?{limit_param} OFFSET ?{offset_param}",
                        placeholders.join(", ")
                    );
                    let mut stmt = conn.prepare(&sql)?;
                    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                    params.push(Box::new(user_id.clone()));
                    for t in &tags {
                        params.push(Box::new(t.clone()));
                    }
                    params.push(Box::new(tags.len() as i64));
                    params.push(Box::new(limit as i64));
                    params.push(Box::new(offset as i64));
                    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
                    let rows = stmt.query_map(&*param_refs, |row| row.get(0))?;
                    rows.collect::<std::result::Result<Vec<_>, _>>()?
                };

                let mut memories = Vec::with_capacity(ids.len());
                for id in ids {
                    let mut stmt = conn.prepare(
                        "SELECT user_id, content, created_at, updated_at FROM memories WHERE id = ?1",
                    )?;
                    let (mem_user_id, content, created_at_str, updated_at_str) = stmt.query_row(
                        rusqlite::params![id],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, String>(3)?,
                            ))
                        },
                    )?;

                    let tags = load_tags(conn, &id)?;
                    let created_at = parse_datetime(&created_at_str)?;
                    let updated_at = parse_datetime(&updated_at_str)?;

                    memories.push(Memory {
                        id,
                        user_id: mem_user_id,
                        content,
                        created_at,
                        updated_at,
                        tags,
                    });
                }

                Ok(memories)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn update(
        &self,
        user_id: &str,
        id: &str,
        content: &str,
        embedder: &dyn Embedder,
    ) -> Result<Memory> {
        let embedding = embedder.embed(content)?;
        let embedding_bytes: Vec<u8> = embedding.as_bytes().to_vec();
        let now = Utc::now();
        let updated_at_str = now.to_rfc3339();
        let id = id.to_string();
        let user_id = user_id.to_string();
        let content = content.to_string();

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                let changed = tx.execute(
                    "UPDATE memories SET content = ?1, updated_at = ?2 WHERE id = ?3 AND user_id = ?4",
                    rusqlite::params![content, updated_at_str, id, user_id],
                )?;

                if changed == 0 {
                    return Err(rusqlite::Error::QueryReturnedNoRows.into());
                }

                tx.execute(
                    "UPDATE memory_embeddings SET embedding = ?1 WHERE memory_id = ?2",
                    rusqlite::params![embedding_bytes, id],
                )?;

                let mut stmt = tx.prepare(
                    "SELECT user_id, content, created_at, updated_at FROM memories WHERE id = ?1",
                )?;
                let (mem_user_id, content, created_at_str, updated_at_str) = stmt.query_row(
                    rusqlite::params![id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    },
                )?;
                drop(stmt);

                let tags = load_tags(&tx, &id)?;
                let created_at = parse_datetime(&created_at_str)?;
                let updated_at = parse_datetime(&updated_at_str)?;

                let memory = Memory {
                    id,
                    user_id: mem_user_id,
                    content,
                    created_at,
                    updated_at,
                    tags,
                };

                tx.commit()?;
                Ok(memory)
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound(format!("memory not found"))
                }
                other => Error::Database(other),
            })
    }

    pub async fn delete(&self, user_id: &str, id: &str) -> Result<()> {
        let id = id.to_string();
        let user_id = user_id.to_string();
        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                // vec0 tables don't CASCADE, delete embeddings first
                tx.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                    rusqlite::params![id],
                )?;

                let changed =
                    tx.execute("DELETE FROM memories WHERE id = ?1 AND user_id = ?2", rusqlite::params![id, user_id])?;
                if changed == 0 {
                    return Err(rusqlite::Error::QueryReturnedNoRows.into());
                }

                tx.commit()?;
                Ok(())
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound(format!("memory not found"))
                }
                other => Error::Database(other),
            })
    }

    pub async fn vote(&self, user_id: &str, memory_id: &str, vote: &str) -> Result<Vote> {
        if vote != "helpful" && vote != "harmful" {
            return Err(Error::InvalidInput(format!(
                "vote must be 'helpful' or 'harmful', got '{vote}'"
            )));
        }

        let id = ulid::Ulid::new().to_string();
        let now = Utc::now();
        let voted_at_str = now.to_rfc3339();
        let memory_id = memory_id.to_string();
        let user_id = user_id.to_string();
        let vote = vote.to_string();
        let id_clone = id.clone();
        let memory_id_clone = memory_id.clone();
        let vote_clone = vote.clone();
        let memory_id_for_err = memory_id.clone();

        self.conn
            .call(move |conn| {
                let exists: bool = conn.query_row(
                    "SELECT 1 FROM memories WHERE id = ?1 AND user_id = ?2",
                    rusqlite::params![memory_id_clone, user_id],
                    |_| Ok(true),
                ).optional()?.unwrap_or(false);

                if !exists {
                    return Err(rusqlite::Error::QueryReturnedNoRows.into());
                }

                conn.execute(
                    "INSERT INTO memory_votes (id, memory_id, vote, voted_at) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![id_clone, memory_id_clone, vote_clone, voted_at_str],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound(format!("memory not found: {}", memory_id_for_err))
                }
                other => Error::Database(other),
            })?;

        Ok(Vote {
            id,
            memory_id,
            vote,
            voted_at: now,
        })
    }

    pub async fn list_tags(&self, user_id: &str) -> Result<Vec<(String, usize)>> {
        let user_id = user_id.to_string();
        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT mt.tag, COUNT(*) as cnt FROM memory_tags mt \
                     JOIN memories m ON mt.memory_id = m.id \
                     WHERE m.user_id = ?1 \
                     GROUP BY mt.tag ORDER BY cnt DESC",
                )?;
                let tags = stmt
                    .query_map(rusqlite::params![user_id], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(tags)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn set_tags(&self, user_id: &str, memory_id: &str, tags: Vec<String>) -> Result<()> {
        let memory_id = memory_id.to_string();
        let user_id = user_id.to_string();
        let memory_id_for_err = memory_id.clone();
        self.conn
            .call(move |conn| {
                let exists: bool = conn.query_row(
                    "SELECT 1 FROM memories WHERE id = ?1 AND user_id = ?2",
                    rusqlite::params![memory_id, user_id],
                    |_| Ok(true),
                ).optional()?.unwrap_or(false);

                if !exists {
                    return Err(rusqlite::Error::QueryReturnedNoRows.into());
                }

                let tx = conn.transaction()?;
                tx.execute(
                    "DELETE FROM memory_tags WHERE memory_id = ?1",
                    rusqlite::params![memory_id],
                )?;
                for tag in &tags {
                    tx.execute(
                        "INSERT INTO memory_tags (memory_id, tag) VALUES (?1, ?2)",
                        rusqlite::params![memory_id, tag],
                    )?;
                }
                tx.commit()?;
                Ok(())
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound(format!("memory not found: {}", memory_id_for_err))
                }
                other => Error::Database(other),
            })
    }

    pub async fn get_vote_counts(&self, memory_id: &str) -> Result<(u64, u64)> {
        let memory_id = memory_id.to_string();
        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT vote, COUNT(*) FROM memory_votes WHERE memory_id = ?1 GROUP BY vote",
                )?;
                let mut helpful: u64 = 0;
                let mut harmful: u64 = 0;
                stmt.query_map(rusqlite::params![memory_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
                })?
                .for_each(|r| {
                    if let Ok((vote, count)) = r {
                        match vote.as_str() {
                            "helpful" => helpful = count,
                            "harmful" => harmful = count,
                            _ => {}
                        }
                    }
                });
                Ok((helpful, harmful))
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn search_by_tags(&self, user_id: &str, tags: &[String], limit: usize) -> Result<Vec<Memory>> {
        if tags.is_empty() {
            return Ok(vec![]);
        }
        let tags = tags.to_vec();
        let user_id = user_id.to_string();
        self.conn
            .call(move |conn| {
                let mut param_idx = 1;
                let user_id_param = param_idx;
                param_idx += 1;
                let tag_start = param_idx;
                let placeholders: Vec<String> = (tag_start..tag_start + tags.len()).map(|i| format!("?{i}")).collect();
                param_idx += tags.len();
                let tag_count_param = param_idx;
                param_idx += 1;
                let limit_param = param_idx;
                let sql = format!(
                    "SELECT m.id, m.user_id, m.content, m.created_at, m.updated_at \
                     FROM memories m \
                     JOIN memory_tags mt ON m.id = mt.memory_id \
                     WHERE m.user_id = ?{user_id_param} AND mt.tag IN ({}) \
                     GROUP BY m.id \
                     HAVING COUNT(DISTINCT mt.tag) = ?{tag_count_param} \
                     ORDER BY m.created_at DESC \
                     LIMIT ?{limit_param}",
                    placeholders.join(", ")
                );
                let mut stmt = conn.prepare(&sql)?;
                let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                params.push(Box::new(user_id.clone()));
                for t in &tags {
                    params.push(Box::new(t.clone()));
                }
                params.push(Box::new(tags.len() as i64));
                params.push(Box::new(limit as i64));
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
                let rows = stmt.query_map(&*param_refs, |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })?;

                let raw: Vec<_> = rows.collect::<std::result::Result<Vec<_>, _>>()?;
                let mut memories = Vec::with_capacity(raw.len());
                for (id, mem_user_id, content, created_at_str, updated_at_str) in raw {
                    let tags = load_tags(conn, &id)?;
                    let created_at = parse_datetime(&created_at_str)?;
                    let updated_at = parse_datetime(&updated_at_str)?;
                    memories.push(Memory {
                        id,
                        user_id: mem_user_id,
                        content,
                        created_at,
                        updated_at,
                        tags,
                    });
                }
                Ok(memories)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn recall(
        &self,
        user_id: &str,
        query: &str,
        n: usize,
        embedder: &dyn Embedder,
        scorer: &Scorer,
    ) -> Result<Vec<ScoredMemory>> {
        let query_embedding = embedder.embed(query)?;
        let query_bytes: Vec<u8> = query_embedding.as_bytes().to_vec();
        let fetch_limit = n * 3;

        let candidates: Vec<(String, f64)> = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT memory_id, distance \
                     FROM memory_embeddings \
                     WHERE embedding MATCH ?1 \
                     ORDER BY distance \
                     LIMIT ?2",
                )?;
                let rows = stmt.query_map(
                    rusqlite::params![query_bytes, fetch_limit as i64],
                    |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                    },
                )?;
                rows.collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| e.into())
            })
            .await
            .map_err(Error::Database)?;

        let distance_map: HashMap<String, f64> = candidates.iter().cloned().collect();
        let memory_ids: Vec<String> = candidates.into_iter().map(|(id, _)| id).collect();

        let now = Utc::now();
        let ids_for_db = memory_ids.clone();
        let user_id = user_id.to_string();

        let memory_data: Vec<(Memory, u64, u64)> = self
            .conn
            .call(move |conn| {
                let mut results = Vec::new();
                for id in &ids_for_db {
                    let mut stmt = conn.prepare(
                        "SELECT user_id, content, created_at, updated_at FROM memories WHERE id = ?1 AND user_id = ?2",
                    )?;
                    let row = stmt.query_row(rusqlite::params![id, user_id], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    });

                    let (mem_user_id, content, created_at_str, updated_at_str) = match row {
                        Ok(r) => r,
                        Err(_) => continue,
                    };

                    let tags = load_tags(conn, id)?;

                    let created_at = parse_datetime(&created_at_str)?;
                    let updated_at = parse_datetime(&updated_at_str)?;

                    let helpful: u64 = conn.query_row(
                        "SELECT COUNT(*) FROM memory_votes WHERE memory_id = ?1 AND vote = 'helpful'",
                        rusqlite::params![id],
                        |row| row.get(0),
                    )?;
                    let harmful: u64 = conn.query_row(
                        "SELECT COUNT(*) FROM memory_votes WHERE memory_id = ?1 AND vote = 'harmful'",
                        rusqlite::params![id],
                        |row| row.get(0),
                    )?;

                    results.push((
                        Memory {
                            id: id.clone(),
                            user_id: mem_user_id,
                            content,
                            created_at,
                            updated_at,
                            tags,
                        },
                        helpful,
                        harmful,
                    ));
                }
                Ok(results)
            })
            .await
            .map_err(Error::Database)?;

        let mut scored: Vec<ScoredMemory> = memory_data
            .into_iter()
            .filter_map(|(memory, helpful, harmful)| {
                let distance = *distance_map.get(&memory.id)?;
                let relevance = 1.0 - distance;
                let age_days = (now - memory.created_at).num_seconds() as f64 / 86400.0;
                let confidence = if helpful + harmful == 0 {
                    0.5
                } else {
                    (helpful as f64) / ((helpful + harmful) as f64)
                };
                let score = scorer.score(relevance, helpful, harmful, age_days);
                Some(ScoredMemory {
                    memory,
                    relevance,
                    confidence,
                    score,
                })
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);

        let returned_ids: Vec<String> = scored.iter().map(|s| s.memory.id.clone()).collect();
        self.conn
            .call(move |conn| {
                let accessed_at = Utc::now().to_rfc3339();
                for id in &returned_ids {
                    conn.execute(
                        "INSERT INTO memory_access_log (memory_id, accessed_at) VALUES (?1, ?2)",
                        rusqlite::params![id, accessed_at],
                    )?;
                }
                Ok(())
            })
            .await
            .map_err(Error::Database)?;

        Ok(scored)
    }

    pub async fn admin_delete(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                // vec0 tables don't CASCADE, delete embeddings first
                tx.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                    rusqlite::params![id],
                )?;

                let changed =
                    tx.execute("DELETE FROM memories WHERE id = ?1", rusqlite::params![id])?;
                if changed == 0 {
                    return Err(rusqlite::Error::QueryReturnedNoRows.into());
                }

                tx.commit()?;
                Ok(())
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound(format!("memory not found"))
                }
                other => Error::Database(other),
            })
    }

    pub async fn admin_stats(&self) -> Result<Vec<(String, usize)>> {
        self.conn
            .call(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT user_id, COUNT(*) as cnt FROM memories GROUP BY user_id ORDER BY cnt DESC",
                )?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn admin_list(&self, filter: ListFilter) -> Result<Vec<Memory>> {
        let limit = filter.limit.unwrap_or(50);
        let offset = filter.offset.unwrap_or(0);
        let tags = filter.tags;

        self.conn
            .call(move |conn| {
                let ids: Vec<String> = if tags.is_empty() {
                    let mut stmt = conn.prepare(
                        "SELECT id FROM memories \
                         ORDER BY created_at DESC \
                         LIMIT ?1 OFFSET ?2",
                    )?;
                    let rows = stmt.query_map(
                        rusqlite::params![limit as i64, offset as i64],
                        |row| row.get(0),
                    )?;
                    rows.collect::<std::result::Result<Vec<_>, _>>()?
                } else {
                    let placeholders: Vec<String> = (1..=tags.len()).map(|i| format!("?{i}")).collect();
                    let tag_count_param = tags.len() + 1;
                    let limit_param = tags.len() + 2;
                    let offset_param = tags.len() + 3;
                    let sql = format!(
                        "SELECT m.id FROM memories m \
                         JOIN memory_tags mt ON m.id = mt.memory_id \
                         WHERE mt.tag IN ({}) \
                         GROUP BY m.id \
                         HAVING COUNT(DISTINCT mt.tag) = ?{tag_count_param} \
                         ORDER BY m.created_at DESC \
                         LIMIT ?{limit_param} OFFSET ?{offset_param}",
                        placeholders.join(", ")
                    );
                    let mut stmt = conn.prepare(&sql)?;
                    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = tags
                        .iter()
                        .map(|t| Box::new(t.clone()) as Box<dyn rusqlite::types::ToSql>)
                        .collect();
                    params.push(Box::new(tags.len() as i64));
                    params.push(Box::new(limit as i64));
                    params.push(Box::new(offset as i64));
                    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
                    let rows = stmt.query_map(&*param_refs, |row| row.get(0))?;
                    rows.collect::<std::result::Result<Vec<_>, _>>()?
                };

                let mut memories = Vec::with_capacity(ids.len());
                for id in ids {
                    let mut stmt = conn.prepare(
                        "SELECT user_id, content, created_at, updated_at FROM memories WHERE id = ?1",
                    )?;
                    let (mem_user_id, content, created_at_str, updated_at_str) = stmt.query_row(
                        rusqlite::params![id],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, String>(3)?,
                            ))
                        },
                    )?;

                    let tags = load_tags(conn, &id)?;
                    let created_at = parse_datetime(&created_at_str)?;
                    let updated_at = parse_datetime(&updated_at_str)?;

                    memories.push(Memory {
                        id,
                        user_id: mem_user_id,
                        content,
                        created_at,
                        updated_at,
                        tags,
                    });
                }

                Ok(memories)
            })
            .await
            .map_err(Error::Database)
    }
}

pub(crate) fn parse_datetime(s: &str) -> rusqlite::Result<DateTime<Utc>> {
    s.parse().map_err(|e: chrono::ParseError| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn load_tags(conn: &rusqlite::Connection, memory_id: &str) -> std::result::Result<Vec<String>, rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT tag FROM memory_tags WHERE memory_id = ?1 ORDER BY tag")?;
    let rows = stmt.query_map(rusqlite::params![memory_id], |row| row.get(0))?;
    rows.collect()
}

use rusqlite::OptionalExtension;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::LocalEmbedder;
    use crate::scoring::{Scorer, ScoringConfig};

    const TEST_USER: &str = "test-user";

    async fn setup() -> (MemoryStore, LocalEmbedder) {
        let conn = crate::db::open_in_memory().await.unwrap();
        let store = MemoryStore::new(conn);
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        (store, embedder)
    }

    #[tokio::test]
    async fn should_create_and_get_memory() {
        let (store, embedder) = setup().await;

        let created = store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "rust is a systems programming language".into(),
                    tags: vec!["rust".into(), "programming".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        assert_eq!(created.content, "rust is a systems programming language");
        assert_eq!(created.user_id, TEST_USER);
        assert_eq!(created.tags, vec!["rust", "programming"]);

        let fetched = store.get(TEST_USER, &created.id).await.unwrap();
        assert_eq!(fetched.content, created.content);
        assert_eq!(fetched.user_id, TEST_USER);
        assert_eq!(fetched.tags, vec!["programming", "rust"]); // sorted
    }

    #[tokio::test]
    async fn should_return_not_found_when_missing_id() {
        let (store, _) = setup().await;
        let result = store.get(TEST_USER, "nonexistent").await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn should_list_memories_with_tag_filter() {
        let (store, embedder) = setup().await;

        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "memory about rust".into(),
                    tags: vec!["rust".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "memory about python".into(),
                    tags: vec!["python".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        let all = store.list(TEST_USER, ListFilter::default()).await.unwrap();
        assert_eq!(all.len(), 2);

        let rust_only = store
            .list(TEST_USER, ListFilter {
                tags: vec!["rust".into()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(rust_only.len(), 1);
        assert_eq!(rust_only[0].content, "memory about rust");
    }

    #[tokio::test]
    async fn should_update_memory_content() {
        let (store, embedder) = setup().await;

        let created = store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "original content".into(),
                    tags: vec!["test".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        let updated = store
            .update(TEST_USER, &created.id, "updated content", &embedder)
            .await
            .unwrap();
        assert_eq!(updated.content, "updated content");

        let fetched = store.get(TEST_USER, &created.id).await.unwrap();
        assert_eq!(fetched.content, "updated content");
    }

    #[tokio::test]
    async fn should_delete_memory() {
        let (store, embedder) = setup().await;

        let created = store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "to be deleted".into(),
                    tags: vec![],
                },
                &embedder,
            )
            .await
            .unwrap();

        store.delete(TEST_USER, &created.id).await.unwrap();

        let result = store.get(TEST_USER, &created.id).await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn should_cast_vote_on_memory() {
        let (store, embedder) = setup().await;

        let created = store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "votable memory".into(),
                    tags: vec![],
                },
                &embedder,
            )
            .await
            .unwrap();

        let vote = store.vote(TEST_USER, &created.id, "helpful").await.unwrap();
        assert_eq!(vote.memory_id, created.id);
        assert_eq!(vote.vote, "helpful");
        assert!(!vote.id.is_empty());
    }

    #[tokio::test]
    async fn should_reject_invalid_vote() {
        let (store, embedder) = setup().await;

        let created = store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "some memory".into(),
                    tags: vec![],
                },
                &embedder,
            )
            .await
            .unwrap();

        let result = store.vote(TEST_USER, &created.id, "neutral").await;
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn should_search_by_tag() {
        let (store, embedder) = setup().await;

        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "rust and webdev".into(),
                    tags: vec!["rust".into(), "webdev".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "python and webdev".into(),
                    tags: vec!["python".into(), "webdev".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        let webdev = store.search_by_tags(TEST_USER, &["webdev".into()], 10).await.unwrap();
        assert_eq!(webdev.len(), 2);

        let rust = store.search_by_tags(TEST_USER, &["rust".into()], 10).await.unwrap();
        assert_eq!(rust.len(), 1);
        assert_eq!(rust[0].content, "rust and webdev");

        let both = store.search_by_tags(TEST_USER, &["rust".into(), "webdev".into()], 10).await.unwrap();
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].content, "rust and webdev");
    }

    #[tokio::test]
    async fn should_list_tags_with_counts() {
        let (store, embedder) = setup().await;
        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "a".into(),
                    tags: vec!["x".into(), "y".into()],
                },
                &embedder,
            )
            .await
            .unwrap();
        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "b".into(),
                    tags: vec!["x".into()],
                },
                &embedder,
            )
            .await
            .unwrap();
        let tags = store.list_tags(TEST_USER).await.unwrap();
        assert_eq!(tags[0], ("x".to_string(), 2));
        assert_eq!(tags[1], ("y".to_string(), 1));
    }

    #[tokio::test]
    async fn should_set_tags_replacing_existing() {
        let (store, embedder) = setup().await;
        let m = store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "a".into(),
                    tags: vec!["old".into()],
                },
                &embedder,
            )
            .await
            .unwrap();
        store
            .set_tags(TEST_USER, &m.id, vec!["new1".into(), "new2".into()])
            .await
            .unwrap();
        let fetched = store.get(TEST_USER, &m.id).await.unwrap();
        assert!(fetched.tags.contains(&"new1".to_string()));
        assert!(fetched.tags.contains(&"new2".to_string()));
        assert!(!fetched.tags.contains(&"old".to_string()));
    }

    #[tokio::test]
    async fn should_get_vote_counts() {
        let (store, embedder) = setup().await;
        let m = store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "votable".into(),
                    tags: vec![],
                },
                &embedder,
            )
            .await
            .unwrap();
        store.vote(TEST_USER, &m.id, "helpful").await.unwrap();
        store.vote(TEST_USER, &m.id, "helpful").await.unwrap();
        store.vote(TEST_USER, &m.id, "harmful").await.unwrap();
        let (helpful, harmful) = store.get_vote_counts(&m.id).await.unwrap();
        assert_eq!(helpful, 2);
        assert_eq!(harmful, 1);
    }

    #[tokio::test]
    async fn should_recall_similar_memories() {
        let (store, embedder) = setup().await;
        let scorer = Scorer::new(ScoringConfig::default());

        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "rust is great for systems programming and memory safety".into(),
                    tags: vec!["programming".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "python is popular for data science and machine learning".into(),
                    tags: vec!["programming".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        store
            .create(
                TEST_USER,
                CreateMemory {
                    content: "chocolate cake recipe with butter and sugar".into(),
                    tags: vec!["cooking".into()],
                },
                &embedder,
            )
            .await
            .unwrap();

        let results = store.recall(TEST_USER, "rust code", 3, &embedder, &scorer).await.unwrap();
        assert!(!results.is_empty());
        assert!(
            results[0].memory.content.contains("rust"),
            "expected first result to be about rust, got: {}",
            results[0].memory.content
        );
        // cooking should rank lower than programming
        let cooking_idx = results.iter().position(|r| r.memory.content.contains("cake"));
        let rust_idx = results.iter().position(|r| r.memory.content.contains("rust"));
        assert!(
            rust_idx.unwrap() < cooking_idx.unwrap(),
            "rust memory should rank higher than cooking"
        );
    }

    #[tokio::test]
    async fn should_isolate_memories_by_user() {
        let (store, embedder) = setup().await;
        store.create("user-a", CreateMemory { content: "a's memory".into(), tags: vec![] }, &embedder).await.unwrap();
        store.create("user-b", CreateMemory { content: "b's memory".into(), tags: vec![] }, &embedder).await.unwrap();
        let a_list = store.list("user-a", ListFilter::default()).await.unwrap();
        assert_eq!(a_list.len(), 1);
        assert_eq!(a_list[0].content, "a's memory");
        let b_list = store.list("user-b", ListFilter::default()).await.unwrap();
        assert_eq!(b_list.len(), 1);
        assert_eq!(b_list[0].content, "b's memory");
    }
}
