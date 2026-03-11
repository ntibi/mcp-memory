use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;

use rusqlite::OptionalExtension;

use crate::embed::Embedder;
use crate::error::{Error, Result};
use crate::memory::{CreateMemory, MemoryStore};

#[derive(Debug, Clone, Serialize)]
pub struct RunProgress {
    pub run_id: String,
    pub status: ProgressStatus,
    pub total_groups: usize,
    pub processed_groups: usize,
    pub current_group_label: String,
    pub suggestions_created: usize,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum ProgressStatus {
    Running,
    Cancelling,
}

pub type ProgressMap = Arc<DashMap<String, RunProgress>>;

pub fn new_progress_map() -> ProgressMap {
    Arc::new(DashMap::new())
}

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

pub async fn find_duplicates(conn: &Connection, user_id: &str, threshold: f64) -> Result<Vec<DuplicateCandidate>> {
    let user_id = user_id.to_string();
    conn.call(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT memory_id, embedding FROM memory_embeddings \
             WHERE user_id = ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![user_id], |row| {
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

        tracing::info!(
            embeddings = entries.len(),
            pairs_above_threshold = candidates.len(),
            threshold,
            "duplicate scan complete"
        );

        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates)
    })
    .await
    .map_err(Error::Database)
}

fn uf_find(parent: &mut HashMap<String, String>, x: &str) -> String {
    let p = parent[x].clone();
    if p != x {
        let root = uf_find(parent, &p);
        parent.insert(x.to_string(), root.clone());
        root
    } else {
        p
    }
}

fn uf_union(
    parent: &mut HashMap<String, String>,
    rank: &mut HashMap<String, usize>,
    a: &str,
    b: &str,
) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    let rank_a = rank[&ra];
    let rank_b = rank[&rb];
    if rank_a < rank_b {
        parent.insert(ra, rb);
    } else if rank_a > rank_b {
        parent.insert(rb, ra);
    } else {
        parent.insert(rb, ra.clone());
        *rank.get_mut(&ra).unwrap() += 1;
    }
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
    user_id: &str,
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
    let user_id = user_id.to_string();
    let suggestion_type = suggestion_type.to_string();
    let suggestion = suggestion.to_string();
    let source = source.to_string();

    conn.call(move |conn| {
        conn.execute(
            "INSERT INTO curation_suggestions (id, user_id, type, memory_ids, suggestion, source, status, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7)",
            rusqlite::params![id_clone, user_id, suggestion_type, memory_ids_json, suggestion, source, now],
        )?;
        Ok(())
    })
    .await
    .map_err(Error::Database)?;

    Ok(id)
}

pub async fn list_suggestions(
    conn: &Connection,
    user_id: &str,
    status: Option<&str>,
) -> Result<Vec<CurationSuggestion>> {
    let user_id = user_id.to_string();
    let status = status.map(|s| s.to_string());

    conn.call(move |conn| {
        let suggestions = if let Some(ref status) = status {
            let mut stmt = conn.prepare(
                "SELECT id, type, memory_ids, suggestion, source, status, created_at \
                 FROM curation_suggestions WHERE user_id = ?1 AND status = ?2 ORDER BY created_at DESC",
            )?;
            let rows = stmt.query_map(rusqlite::params![user_id, status], parse_suggestion_row)?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, type, memory_ids, suggestion, source, status, created_at \
                 FROM curation_suggestions WHERE user_id = ?1 ORDER BY created_at DESC",
            )?;
            let rows = stmt.query_map(rusqlite::params![user_id], parse_suggestion_row)?;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleWindow {
    pub days: Vec<u8>,
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurationSettings {
    pub user_id: String,
    pub provider: String,
    pub api_key: Option<String>,
    pub schedule_windows: Vec<ScheduleWindow>,
    pub similarity_threshold: f64,
    pub budget_limit_usd: Option<f64>,
    pub model: String,
    pub enabled: bool,
}

pub async fn get_settings(conn: &Connection, user_id: &str) -> Result<CurationSettings> {
    let user_id = user_id.to_string();
    match conn.call({
        let user_id = user_id.clone();
        move |conn| {
            Ok(conn.query_row(
                "SELECT api_key, schedule_windows, similarity_threshold, budget_limit_usd, model, enabled, provider \
                 FROM curation_settings WHERE user_id = ?1",
                rusqlite::params![user_id],
                |row| {
                    let api_key: Option<String> = row.get(0)?;
                    let windows_json: String = row.get(1)?;
                    let similarity_threshold: f64 = row.get(2)?;
                    let budget_limit_usd: Option<f64> = row.get(3)?;
                    let model: String = row.get(4)?;
                    let enabled_int: i64 = row.get(5)?;
                    let provider: String = row.get(6)?;
                    let schedule_windows: Vec<ScheduleWindow> =
                        serde_json::from_str(&windows_json).map_err(|e| {
                            rusqlite::Error::FromSqlConversionFailure(
                                1,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            )
                        })?;
                    Ok(CurationSettings {
                        user_id: user_id.clone(),
                        provider,
                        api_key,
                        schedule_windows,
                        similarity_threshold,
                        budget_limit_usd,
                        model,
                        enabled: enabled_int != 0,
                    })
                },
            ))
        }
    })
    .await {
        Ok(Ok(settings)) => Ok(settings),
        Ok(Err(rusqlite::Error::QueryReturnedNoRows)) => {
            Ok(CurationSettings {
                user_id,
                provider: "anthropic".to_string(),
                api_key: None,
                schedule_windows: vec![],
                similarity_threshold: 0.85,
                budget_limit_usd: None,
                model: "claude-sonnet-4-6".to_string(),
                enabled: false,
            })
        }
        Ok(Err(e)) => Err(Error::Rusqlite(e)),
        Err(e) => Err(Error::Database(e)),
    }
}

pub async fn upsert_settings(conn: &Connection, settings: &CurationSettings) -> Result<()> {
    let windows_json = serde_json::to_string(&settings.schedule_windows)
        .map_err(|e| Error::Curation(format!("failed to serialize schedule_windows: {e}")))?;
    let user_id = settings.user_id.clone();
    let provider = settings.provider.clone();
    let api_key = settings.api_key.clone();
    let similarity_threshold = settings.similarity_threshold;
    let budget_limit_usd = settings.budget_limit_usd;
    let model = settings.model.clone();
    let enabled = settings.enabled as i64;

    conn.call(move |conn| {
        conn.execute(
            "INSERT INTO curation_settings (user_id, provider, api_key, schedule_windows, similarity_threshold, budget_limit_usd, model, enabled) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
             ON CONFLICT(user_id) DO UPDATE SET \
                provider = excluded.provider, \
                api_key = excluded.api_key, \
                schedule_windows = excluded.schedule_windows, \
                similarity_threshold = excluded.similarity_threshold, \
                budget_limit_usd = excluded.budget_limit_usd, \
                model = excluded.model, \
                enabled = excluded.enabled",
            rusqlite::params![user_id, provider, api_key, windows_json, similarity_threshold, budget_limit_usd, model, enabled],
        )?;
        Ok(())
    })
    .await
    .map_err(Error::Database)
}

pub async fn update_suggestion_status(
    conn: &Connection,
    user_id: &str,
    id: &str,
    status: &str,
) -> Result<()> {
    if status != "applied" && status != "dismissed" {
        return Err(Error::InvalidInput(format!(
            "status must be 'applied' or 'dismissed', got '{status}'"
        )));
    }

    let id = id.to_string();
    let user_id = user_id.to_string();
    let status = status.to_string();

    let changed = conn
        .call(move |conn| {
            let changed = conn.execute(
                "UPDATE curation_suggestions SET status = ?1 WHERE id = ?2 AND user_id = ?3",
                rusqlite::params![status, id, user_id],
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurationRun {
    pub id: String,
    pub user_id: String,
    pub status: String,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub total_groups: i64,
    pub processed_groups: i64,
    pub current_group_label: Option<String>,
    pub suggestions_created: i64,
    pub tokens_used: i64,
    pub cost_usd: f64,
    pub error: Option<String>,
    pub processed_memory_ids: Vec<String>,
}

pub async fn create_run(conn: &Connection, user_id: &str) -> Result<CurationRun> {
    let run = CurationRun {
        id: ulid::Ulid::new().to_string(),
        user_id: user_id.to_string(),
        status: "running".to_string(),
        started_at: chrono::Utc::now().to_rfc3339(),
        completed_at: None,
        total_groups: 0,
        processed_groups: 0,
        current_group_label: None,
        suggestions_created: 0,
        tokens_used: 0,
        cost_usd: 0.0,
        error: None,
        processed_memory_ids: vec![],
    };

    let r = run.clone();
    conn.call(move |conn| {
        conn.execute(
            "INSERT INTO curation_runs (id, user_id, status, started_at, processed_memory_ids) \
             VALUES (?1, ?2, ?3, ?4, '[]')",
            rusqlite::params![r.id, r.user_id, r.status, r.started_at],
        )?;
        Ok(())
    })
    .await
    .map_err(Error::Database)?;

    Ok(run)
}

pub async fn update_run(conn: &Connection, run: &CurationRun) -> Result<()> {
    let memory_ids_json = serde_json::to_string(&run.processed_memory_ids)
        .map_err(|e| Error::Curation(format!("failed to serialize processed_memory_ids: {e}")))?;

    let id = run.id.clone();
    let user_id = run.user_id.clone();
    let status = run.status.clone();
    let completed_at = run.completed_at.clone();
    let total_groups = run.total_groups;
    let processed_groups = run.processed_groups;
    let current_group_label = run.current_group_label.clone();
    let suggestions_created = run.suggestions_created;
    let tokens_used = run.tokens_used;
    let cost_usd = run.cost_usd;
    let error = run.error.clone();

    conn.call(move |conn| {
        conn.execute(
            "UPDATE curation_runs SET \
             status = ?1, completed_at = ?2, total_groups = ?3, processed_groups = ?4, \
             current_group_label = ?5, suggestions_created = ?6, tokens_used = ?7, \
             cost_usd = ?8, error = ?9, processed_memory_ids = ?10 \
             WHERE id = ?11 AND user_id = ?12",
            rusqlite::params![
                status, completed_at, total_groups, processed_groups,
                current_group_label, suggestions_created, tokens_used,
                cost_usd, error, memory_ids_json, id, user_id,
            ],
        )?;
        Ok(())
    })
    .await
    .map_err(Error::Database)?;

    Ok(())
}

pub async fn get_active_run(conn: &Connection, user_id: &str) -> Result<Option<CurationRun>> {
    let user_id = user_id.to_string();
    conn.call(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT id, user_id, status, started_at, completed_at, total_groups, processed_groups, \
             current_group_label, suggestions_created, tokens_used, cost_usd, error, processed_memory_ids \
             FROM curation_runs WHERE user_id = ?1 AND status IN ('pending', 'running') LIMIT 1",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![user_id], parse_run_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    })
    .await
    .map_err(Error::Database)
}

pub async fn list_runs(conn: &Connection, user_id: &str, limit: usize) -> Result<Vec<CurationRun>> {
    let user_id = user_id.to_string();
    conn.call(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT id, user_id, status, started_at, completed_at, total_groups, processed_groups, \
             current_group_label, suggestions_created, tokens_used, cost_usd, error, processed_memory_ids \
             FROM curation_runs WHERE user_id = ?1 ORDER BY started_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![user_id, limit], parse_run_row)?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    })
    .await
    .map_err(Error::Database)
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateGroup {
    pub memory_ids: Vec<String>,
    pub source: String,
    pub label: String,
}

pub async fn find_tag_groups(
    conn: &Connection,
    user_id: &str,
    min_shared_tags: usize,
    exclude_memory_ids: &[String],
) -> Result<Vec<Vec<String>>> {
    let user_id = user_id.to_string();
    let exclude = exclude_memory_ids.to_vec();

    conn.call(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT mt.memory_id, mt.tag FROM memory_tags mt \
             JOIN memories m ON m.id = mt.memory_id \
             WHERE m.user_id = ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![user_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut tags_by_memory: HashMap<String, Vec<String>> = HashMap::new();
        for row in rows {
            let (memory_id, tag) = row?;
            if exclude.contains(&memory_id) {
                continue;
            }
            tags_by_memory.entry(memory_id).or_default().push(tag);
        }

        let memory_ids: Vec<&String> = tags_by_memory.keys().collect();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut rank: HashMap<String, usize> = HashMap::new();

        for id in &memory_ids {
            parent.insert((*id).clone(), (*id).clone());
            rank.insert((*id).clone(), 0);
        }

        for i in 0..memory_ids.len() {
            for j in (i + 1)..memory_ids.len() {
                let tags_a = &tags_by_memory[memory_ids[i]];
                let tags_b = &tags_by_memory[memory_ids[j]];
                let shared = tags_a.iter().filter(|t| tags_b.contains(t)).count();
                if shared >= min_shared_tags {
                    uf_union(&mut parent, &mut rank, memory_ids[i], memory_ids[j]);
                }
            }
        }

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for id in &memory_ids {
            let root = uf_find(&mut parent, id);
            groups.entry(root).or_default().push((*id).clone());
        }

        let result: Vec<Vec<String>> = groups
            .into_values()
            .filter(|g| g.len() >= 2)
            .collect();

        Ok(result)
    })
    .await
    .map_err(Error::Database)
}

pub async fn select_candidate_groups(
    conn: &Connection,
    user_id: &str,
    similarity_threshold: f64,
    exclude_memory_ids: &[String],
) -> Result<Vec<CandidateGroup>> {
    let duplicates = find_duplicates(conn, user_id, similarity_threshold).await?;

    let mut parent: HashMap<String, String> = HashMap::new();
    let mut rank: HashMap<String, usize> = HashMap::new();

    for dup in &duplicates {
        for id in [&dup.memory_id_a, &dup.memory_id_b] {
            parent.entry(id.clone()).or_insert_with(|| id.clone());
            rank.entry(id.clone()).or_insert(0);
        }
        uf_union(&mut parent, &mut rank, &dup.memory_id_a, &dup.memory_id_b);
    }

    let mut embedding_groups: HashMap<String, Vec<String>> = HashMap::new();
    for id in parent.keys().cloned().collect::<Vec<_>>() {
        let root = uf_find(&mut parent, &id);
        embedding_groups.entry(root).or_default().push(id);
    }

    let mut all_covered: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result: Vec<CandidateGroup> = Vec::new();

    for (_, mut ids) in embedding_groups {
        if ids.len() < 2 {
            continue;
        }
        ids.sort();
        for id in &ids {
            all_covered.insert(id.clone());
        }
        result.push(CandidateGroup {
            label: format!("embedding-similar ({})", ids.len()),
            source: "embedding".to_string(),
            memory_ids: ids,
        });
    }

    let mut tag_exclude: Vec<String> = exclude_memory_ids.to_vec();
    tag_exclude.extend(all_covered.iter().cloned());

    let tag_groups = find_tag_groups(conn, user_id, 2, &tag_exclude).await?;

    for mut ids in tag_groups {
        ids.retain(|id| !all_covered.contains(id));
        if ids.len() < 2 {
            continue;
        }
        ids.sort();
        for id in &ids {
            all_covered.insert(id.clone());
        }
        result.push(CandidateGroup {
            label: format!("tag-cooccurrence ({})", ids.len()),
            source: "tags".to_string(),
            memory_ids: ids,
        });
    }

    Ok(result)
}

pub async fn get_suggestion(
    conn: &Connection,
    user_id: &str,
    id: &str,
) -> Result<CurationSuggestion> {
    let user_id = user_id.to_string();
    let id = id.to_string();
    let id_for_err = id.clone();

    conn.call(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT id, type, memory_ids, suggestion, source, status, created_at \
             FROM curation_suggestions WHERE id = ?1 AND user_id = ?2",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![id, user_id], parse_suggestion_row)?;
        match rows.next() {
            Some(row) => Ok(row?),
            None => Err(rusqlite::Error::QueryReturnedNoRows.into()),
        }
    })
    .await
    .map_err(|e| match e {
        tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
            Error::NotFound(format!("suggestion not found: {id_for_err}"))
        }
        other => Error::Database(other),
    })
}

#[derive(Debug, Deserialize)]
struct SuggestionPayload {
    action: String,
    memory_ids: Vec<String>,
    content: String,
    tags: Vec<String>,
}

pub async fn apply_suggestion(
    conn: &Connection,
    store: &MemoryStore,
    embedder: &dyn Embedder,
    user_id: &str,
    suggestion_id: &str,
) -> Result<()> {
    let suggestion = get_suggestion(conn, user_id, suggestion_id).await?;

    let payload: SuggestionPayload = serde_json::from_str(&suggestion.suggestion)
        .map_err(|e| Error::Curation(format!("failed to parse suggestion json: {e}")))?;

    match payload.action.as_str() {
        "merge" => {
            let new_memory = store
                .create(
                    user_id,
                    CreateMemory {
                        content: payload.content,
                        tags: payload.tags,
                    },
                    embedder,
                )
                .await?;
            for source_id in &payload.memory_ids {
                store.reassign_votes(source_id, &new_memory.id).await?;
                store.reassign_access_log(source_id, &new_memory.id).await?;
                store.delete(user_id, source_id).await?;
            }
        }
        other => {
            return Err(Error::Curation(format!("unknown suggestion action: {other}")));
        }
    }

    update_suggestion_status(conn, user_id, suggestion_id, "applied").await?;
    invalidate_overlapping_suggestions(conn, user_id, suggestion_id, &payload.memory_ids).await?;
    Ok(())
}

async fn invalidate_overlapping_suggestions(
    conn: &Connection,
    user_id: &str,
    applied_id: &str,
    memory_ids: &[String],
) -> Result<()> {
    let user_id = user_id.to_string();
    let applied_id = applied_id.to_string();
    let memory_ids_json = serde_json::to_string(memory_ids)
        .map_err(|e| Error::Curation(format!("failed to serialize memory_ids: {e}")))?;

    conn.call(move |conn| {
        conn.execute(
            "UPDATE curation_suggestions \
             SET status = 'dismissed' \
             WHERE user_id = ?1 \
               AND id != ?2 \
               AND status = 'pending' \
               AND EXISTS ( \
                   SELECT 1 FROM json_each(curation_suggestions.memory_ids) AS j \
                   WHERE j.value IN (SELECT value FROM json_each(?3)) \
               )",
            rusqlite::params![user_id, applied_id, memory_ids_json],
        )?;
        Ok(())
    })
    .await
    .map_err(Error::Database)
}

pub async fn store_dismissed_pair(
    conn: &Connection,
    user_id: &str,
    memory_id_a: &str,
    memory_id_b: &str,
) -> Result<()> {
    let user_id = user_id.to_string();
    let (a, b) = normalize_pair(memory_id_a, memory_id_b);
    let now = chrono::Utc::now().to_rfc3339();

    conn.call(move |conn| {
        conn.execute(
            "INSERT OR IGNORE INTO curation_dismissed_pairs (user_id, memory_id_a, memory_id_b, dismissed_at) \
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![user_id, a, b, now],
        )?;
        Ok(())
    })
    .await
    .map_err(Error::Database)
}

pub async fn is_dismissed_pair(
    conn: &Connection,
    user_id: &str,
    memory_id_a: &str,
    memory_id_b: &str,
) -> Result<bool> {
    let user_id = user_id.to_string();
    let (a, b) = normalize_pair(memory_id_a, memory_id_b);

    conn.call(move |conn| {
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM curation_dismissed_pairs WHERE user_id = ?1 AND memory_id_a = ?2 AND memory_id_b = ?3",
                rusqlite::params![user_id, a, b],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        Ok(exists)
    })
    .await
    .map_err(Error::Database)
}

pub async fn get_dismissed_pairs(
    conn: &Connection,
    user_id: &str,
) -> Result<Vec<(String, String)>> {
    let user_id = user_id.to_string();

    conn.call(move |conn| {
        let mut stmt = conn.prepare(
            "SELECT memory_id_a, memory_id_b FROM curation_dismissed_pairs WHERE user_id = ?1",
        )?;
        let rows = stmt.query_map(rusqlite::params![user_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    })
    .await
    .map_err(Error::Database)
}

fn normalize_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn parse_run_row(row: &rusqlite::Row) -> rusqlite::Result<CurationRun> {
    let memory_ids_raw: String = row.get(12)?;
    let processed_memory_ids: Vec<String> = serde_json::from_str(&memory_ids_raw).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(
            12,
            rusqlite::types::Type::Text,
            Box::new(e),
        )
    })?;
    Ok(CurationRun {
        id: row.get(0)?,
        user_id: row.get(1)?,
        status: row.get(2)?,
        started_at: row.get(3)?,
        completed_at: row.get(4)?,
        total_groups: row.get(5)?,
        processed_groups: row.get(6)?,
        current_group_label: row.get(7)?,
        suggestions_created: row.get(8)?,
        tokens_used: row.get(9)?,
        cost_usd: row.get(10)?,
        error: row.get(11)?,
        processed_memory_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::LocalEmbedder;
    use crate::memory::{CreateMemory, MemoryStore};

    const TEST_USER: &str = "test-user";

    #[tokio::test]
    async fn should_detect_duplicate_memories() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        store.create(TEST_USER, CreateMemory { content: "rust is a systems programming language".into(), tags: vec![] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "rust is a systems-level programming language".into(), tags: vec![] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "chocolate cake recipe with frosting".into(), tags: vec![] }, &embedder).await.unwrap();

        let dupes = find_duplicates(&conn, TEST_USER, 0.8).await.unwrap();
        assert!(!dupes.is_empty());
        assert!(dupes[0].similarity > 0.8);
    }

    #[tokio::test]
    async fn should_not_find_duplicates_when_dissimilar() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        store.create(TEST_USER, CreateMemory { content: "rust is a systems programming language".into(), tags: vec![] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "chocolate cake recipe with frosting".into(), tags: vec![] }, &embedder).await.unwrap();

        let dupes = find_duplicates(&conn, TEST_USER, 0.9).await.unwrap();
        assert!(dupes.is_empty());
    }

    #[tokio::test]
    async fn should_store_and_list_suggestions() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let ids = vec!["id_a".to_string(), "id_b".to_string()];
        let suggestion_id = store_suggestion(&conn, TEST_USER, "merge", &ids, "merge these two", "auto").await.unwrap();
        assert!(!suggestion_id.is_empty());

        let all = list_suggestions(&conn, TEST_USER, None).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].suggestion_type, "merge");
        assert_eq!(all[0].memory_ids, ids);
        assert_eq!(all[0].status, "pending");

        let pending = list_suggestions(&conn, TEST_USER, Some("pending")).await.unwrap();
        assert_eq!(pending.len(), 1);

        let applied = list_suggestions(&conn, TEST_USER, Some("applied")).await.unwrap();
        assert!(applied.is_empty());
    }

    #[tokio::test]
    async fn should_update_suggestion_status() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let ids = vec!["id_a".to_string()];
        let suggestion_id = store_suggestion(&conn, TEST_USER, "merge", &ids, "remove stale memory", "llm").await.unwrap();

        update_suggestion_status(&conn, TEST_USER, &suggestion_id, "applied").await.unwrap();

        let all = list_suggestions(&conn, TEST_USER, Some("applied")).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, suggestion_id);
    }

    #[tokio::test]
    async fn should_reject_invalid_suggestion_status() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let ids = vec!["id_a".to_string()];
        let suggestion_id = store_suggestion(&conn, TEST_USER, "merge", &ids, "remove it", "auto").await.unwrap();

        let result = update_suggestion_status(&conn, TEST_USER, &suggestion_id, "invalid").await;
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn should_return_not_found_for_missing_suggestion() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let result = update_suggestion_status(&conn, TEST_USER, "nonexistent", "applied").await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn should_return_defaults_when_no_settings() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let settings = get_settings(&conn, "user-1").await.unwrap();
        assert_eq!(settings.similarity_threshold, 0.85);
        assert!(!settings.enabled);
        assert!(settings.api_key.is_none());
    }

    #[tokio::test]
    async fn should_upsert_and_get_settings() {
        let conn = crate::db::open_in_memory().await.unwrap();
        conn.call(|c| {
            c.execute(
                "INSERT INTO users (id, name, created_at) VALUES ('u1', 'test', '2026-01-01T00:00:00Z')",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();

        let settings = CurationSettings {
            user_id: "u1".to_string(),
            provider: "anthropic".to_string(),
            api_key: Some("sk-test-123".to_string()),
            schedule_windows: vec![ScheduleWindow {
                days: vec![1, 2, 3],
                start: "02:00".to_string(),
                end: "05:00".to_string(),
            }],
            similarity_threshold: 0.9,
            budget_limit_usd: Some(1.0),
            model: "claude-haiku-4-5".to_string(),
            enabled: true,
        };
        upsert_settings(&conn, &settings).await.unwrap();

        let loaded = get_settings(&conn, "u1").await.unwrap();
        assert_eq!(loaded.api_key.as_deref(), Some("sk-test-123"));
        assert_eq!(loaded.schedule_windows.len(), 1);
        assert_eq!(loaded.similarity_threshold, 0.9);
        assert!(loaded.enabled);
    }

    async fn setup_run_test() -> Connection {
        let conn = crate::db::open_in_memory().await.unwrap();
        conn.call(|c| {
            c.execute(
                "INSERT INTO users (id, name, created_at) VALUES ('test-user', 'test', '2026-01-01T00:00:00Z')",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
        conn
    }

    #[tokio::test]
    async fn should_create_run_with_running_status() {
        let conn = setup_run_test().await;
        let run = create_run(&conn, TEST_USER).await.unwrap();
        assert_eq!(run.status, "running");
        assert_eq!(run.user_id, TEST_USER);
        assert_eq!(run.total_groups, 0);
        assert_eq!(run.processed_groups, 0);
        assert_eq!(run.suggestions_created, 0);
        assert_eq!(run.tokens_used, 0);
        assert_eq!(run.cost_usd, 0.0);
        assert!(run.completed_at.is_none());
        assert!(run.error.is_none());
        assert!(run.processed_memory_ids.is_empty());
    }

    #[tokio::test]
    async fn should_update_run_status_and_counters() {
        let conn = setup_run_test().await;
        let mut run = create_run(&conn, TEST_USER).await.unwrap();

        run.status = "completed".to_string();
        run.completed_at = Some(chrono::Utc::now().to_rfc3339());
        run.total_groups = 5;
        run.processed_groups = 5;
        run.suggestions_created = 3;
        run.tokens_used = 1000;
        run.cost_usd = 0.05;
        run.processed_memory_ids = vec!["m1".to_string(), "m2".to_string()];
        update_run(&conn, &run).await.unwrap();

        let runs = list_runs(&conn, TEST_USER, 10).await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, "completed");
        assert_eq!(runs[0].total_groups, 5);
        assert_eq!(runs[0].processed_groups, 5);
        assert_eq!(runs[0].suggestions_created, 3);
        assert_eq!(runs[0].tokens_used, 1000);
        assert_eq!(runs[0].cost_usd, 0.05);
        assert_eq!(runs[0].processed_memory_ids, vec!["m1", "m2"]);
        assert!(runs[0].completed_at.is_some());
    }

    #[tokio::test]
    async fn should_return_none_when_no_active_run() {
        let conn = setup_run_test().await;
        let active = get_active_run(&conn, TEST_USER).await.unwrap();
        assert!(active.is_none());
    }

    #[tokio::test]
    async fn should_return_active_run_when_running() {
        let conn = setup_run_test().await;
        let run = create_run(&conn, TEST_USER).await.unwrap();

        let active = get_active_run(&conn, TEST_USER).await.unwrap();
        assert!(active.is_some());
        assert_eq!(active.unwrap().id, run.id);
    }

    #[tokio::test]
    async fn should_not_return_completed_run_as_active() {
        let conn = setup_run_test().await;
        let mut run = create_run(&conn, TEST_USER).await.unwrap();
        run.status = "completed".to_string();
        update_run(&conn, &run).await.unwrap();

        let active = get_active_run(&conn, TEST_USER).await.unwrap();
        assert!(active.is_none());
    }

    #[tokio::test]
    async fn should_list_runs_in_descending_order() {
        let conn = setup_run_test().await;

        let run1 = create_run(&conn, TEST_USER).await.unwrap();
        let mut run1_updated = run1.clone();
        run1_updated.status = "completed".to_string();
        update_run(&conn, &run1_updated).await.unwrap();

        let run2 = create_run(&conn, TEST_USER).await.unwrap();

        let runs = list_runs(&conn, TEST_USER, 10).await.unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].id, run2.id);
        assert_eq!(runs[1].id, run1.id);
    }

    #[tokio::test]
    async fn should_find_tag_groups_with_shared_tags() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        let m1 = store.create(TEST_USER, CreateMemory { content: "rust async patterns".into(), tags: vec!["rust".into(), "async".into(), "patterns".into()] }, &embedder).await.unwrap();
        let m2 = store.create(TEST_USER, CreateMemory { content: "rust async error handling".into(), tags: vec!["rust".into(), "async".into(), "errors".into()] }, &embedder).await.unwrap();
        let _m3 = store.create(TEST_USER, CreateMemory { content: "python web framework".into(), tags: vec!["python".into(), "web".into()] }, &embedder).await.unwrap();

        let groups = find_tag_groups(&conn, TEST_USER, 2, &[]).await.unwrap();
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert!(group.contains(&m1.id));
        assert!(group.contains(&m2.id));
    }

    #[tokio::test]
    async fn should_not_group_when_insufficient_shared_tags() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        store.create(TEST_USER, CreateMemory { content: "rust systems".into(), tags: vec!["rust".into(), "systems".into()] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "rust web".into(), tags: vec!["rust".into(), "web".into()] }, &embedder).await.unwrap();

        let groups = find_tag_groups(&conn, TEST_USER, 2, &[]).await.unwrap();
        assert!(groups.is_empty());
    }

    #[tokio::test]
    async fn should_exclude_memory_ids_from_tag_groups() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        let m1 = store.create(TEST_USER, CreateMemory { content: "rust async patterns".into(), tags: vec!["rust".into(), "async".into()] }, &embedder).await.unwrap();
        let _m2 = store.create(TEST_USER, CreateMemory { content: "rust async errors".into(), tags: vec!["rust".into(), "async".into()] }, &embedder).await.unwrap();

        let groups = find_tag_groups(&conn, TEST_USER, 2, &[m1.id.clone()]).await.unwrap();
        assert!(groups.is_empty());
    }

    #[tokio::test]
    async fn should_merge_overlapping_tag_groups_via_union_find() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        let m1 = store.create(TEST_USER, CreateMemory { content: "topic alpha".into(), tags: vec!["a".into(), "b".into(), "c".into()] }, &embedder).await.unwrap();
        let m2 = store.create(TEST_USER, CreateMemory { content: "topic beta".into(), tags: vec!["a".into(), "b".into(), "d".into()] }, &embedder).await.unwrap();
        let m3 = store.create(TEST_USER, CreateMemory { content: "topic gamma".into(), tags: vec!["a".into(), "d".into(), "e".into()] }, &embedder).await.unwrap();

        let groups = find_tag_groups(&conn, TEST_USER, 2, &[]).await.unwrap();
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert!(group.contains(&m1.id));
        assert!(group.contains(&m2.id));
        assert!(group.contains(&m3.id));
    }

    #[tokio::test]
    async fn should_select_candidate_groups_from_both_passes() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        store.create(TEST_USER, CreateMemory { content: "rust is a systems programming language".into(), tags: vec!["lang".into()] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "rust is a systems-level programming language".into(), tags: vec!["lang".into()] }, &embedder).await.unwrap();

        store.create(TEST_USER, CreateMemory { content: "deployed the new auth service to production".into(), tags: vec!["deployment".into(), "auth".into(), "production".into()] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "rolled back auth service in production".into(), tags: vec!["deployment".into(), "auth".into(), "production".into()] }, &embedder).await.unwrap();

        let groups = select_candidate_groups(&conn, TEST_USER, 0.8, &[]).await.unwrap();
        assert!(groups.len() >= 1);

        let has_embedding = groups.iter().any(|g| g.source == "embedding");
        assert!(has_embedding);
    }

    #[tokio::test]
    async fn should_deduplicate_memories_across_passes() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        store.create(TEST_USER, CreateMemory { content: "rust is a systems programming language".into(), tags: vec!["rust".into(), "programming".into()] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "rust is a systems-level programming language".into(), tags: vec!["rust".into(), "programming".into()] }, &embedder).await.unwrap();

        let groups = select_candidate_groups(&conn, TEST_USER, 0.8, &[]).await.unwrap();

        let mut all_ids: Vec<&String> = groups.iter().flat_map(|g| &g.memory_ids).collect();
        let total = all_ids.len();
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(total, all_ids.len(), "no memory should appear in multiple groups");
    }

    #[tokio::test]
    async fn should_respect_exclusion_in_candidate_selection() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        let m1 = store.create(TEST_USER, CreateMemory { content: "deployed auth to prod".into(), tags: vec!["deploy".into(), "auth".into(), "prod".into()] }, &embedder).await.unwrap();
        store.create(TEST_USER, CreateMemory { content: "rolled back auth in prod".into(), tags: vec!["deploy".into(), "auth".into(), "prod".into()] }, &embedder).await.unwrap();

        let groups = select_candidate_groups(&conn, TEST_USER, 0.99, &[m1.id.clone()]).await.unwrap();
        for group in &groups {
            assert!(!group.memory_ids.contains(&m1.id));
        }
    }

    #[tokio::test]
    async fn should_get_suggestion_by_id() {
        let conn = crate::db::open_in_memory().await.unwrap();

        let ids = vec!["id_a".to_string(), "id_b".to_string()];
        let suggestion_id = store_suggestion(&conn, TEST_USER, "merge", &ids, "merge them", "auto").await.unwrap();

        let s = get_suggestion(&conn, TEST_USER, &suggestion_id).await.unwrap();
        assert_eq!(s.id, suggestion_id);
        assert_eq!(s.suggestion_type, "merge");
        assert_eq!(s.memory_ids, ids);
    }

    #[tokio::test]
    async fn should_return_not_found_for_missing_suggestion_get() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let result = get_suggestion(&conn, TEST_USER, "nonexistent").await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn should_merge_memories_when_applying_merge_suggestion() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        let m1 = store.create(TEST_USER, CreateMemory { content: "rust async".into(), tags: vec!["rust".into()] }, &embedder).await.unwrap();
        let m2 = store.create(TEST_USER, CreateMemory { content: "rust futures".into(), tags: vec!["rust".into()] }, &embedder).await.unwrap();

        store.vote(TEST_USER, &m1.id, "helpful").await.unwrap();
        store.vote(TEST_USER, &m1.id, "helpful").await.unwrap();
        store.vote(TEST_USER, &m2.id, "helpful").await.unwrap();

        let suggestion_json = serde_json::json!({
            "action": "merge",
            "memory_ids": [m1.id, m2.id],
            "content": "rust async and futures combined",
            "tags": ["rust", "async"],
            "reasoning": "overlapping content"
        }).to_string();

        let memory_ids = vec![m1.id.clone(), m2.id.clone()];
        let suggestion_id = store_suggestion(&conn, TEST_USER, "merge", &memory_ids, &suggestion_json, "llm").await.unwrap();

        apply_suggestion(&conn, &store, &embedder, TEST_USER, &suggestion_id).await.unwrap();

        assert!(matches!(store.get(TEST_USER, &m1.id).await, Err(Error::NotFound(_))));
        assert!(matches!(store.get(TEST_USER, &m2.id).await, Err(Error::NotFound(_))));

        let all = store.list(TEST_USER, crate::memory::ListFilter::default()).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].content, "rust async and futures combined");
        assert!(all[0].tags.contains(&"async".to_string()));
        assert!(all[0].tags.contains(&"rust".to_string()));

        let (helpful, harmful) = store.get_vote_counts(&all[0].id).await.unwrap();
        assert_eq!(helpful, 3);
        assert_eq!(harmful, 0);

        let s = get_suggestion(&conn, TEST_USER, &suggestion_id).await.unwrap();
        assert_eq!(s.status, "applied");
    }

    #[tokio::test]
    async fn should_invalidate_overlapping_suggestions_on_apply() {
        let conn = crate::db::open_in_memory().await.unwrap();
        let embedder = LocalEmbedder::new("all-MiniLM-L6-v2").unwrap();
        let store = MemoryStore::new(conn.clone());

        let m1 = store.create(TEST_USER, CreateMemory { content: "rust async patterns".into(), tags: vec!["rust".into()] }, &embedder).await.unwrap();
        let m2 = store.create(TEST_USER, CreateMemory { content: "rust futures guide".into(), tags: vec!["rust".into()] }, &embedder).await.unwrap();
        let m3 = store.create(TEST_USER, CreateMemory { content: "rust error handling".into(), tags: vec!["rust".into()] }, &embedder).await.unwrap();

        let suggestion_a_json = serde_json::json!({
            "action": "merge",
            "memory_ids": [m1.id, m2.id],
            "content": "merged m1+m2",
            "tags": ["rust"],
            "reasoning": "overlap"
        }).to_string();
        let suggestion_a_id = store_suggestion(&conn, TEST_USER, "merge", &[m1.id.clone(), m2.id.clone()], &suggestion_a_json, "llm").await.unwrap();

        let suggestion_b_json = serde_json::json!({
            "action": "merge",
            "memory_ids": [m2.id, m3.id],
            "content": "merged m2+m3",
            "tags": ["rust"],
            "reasoning": "overlap"
        }).to_string();
        let suggestion_b_id = store_suggestion(&conn, TEST_USER, "merge", &[m2.id.clone(), m3.id.clone()], &suggestion_b_json, "llm").await.unwrap();

        let suggestion_c_json = serde_json::json!({
            "action": "merge",
            "memory_ids": [m3.id],
            "content": "just m3",
            "tags": ["rust"],
            "reasoning": "solo"
        }).to_string();
        let suggestion_c_id = store_suggestion(&conn, TEST_USER, "merge", &[m3.id.clone()], &suggestion_c_json, "llm").await.unwrap();

        apply_suggestion(&conn, &store, &embedder, TEST_USER, &suggestion_a_id).await.unwrap();

        let sa = get_suggestion(&conn, TEST_USER, &suggestion_a_id).await.unwrap();
        assert_eq!(sa.status, "applied");

        let sb = get_suggestion(&conn, TEST_USER, &suggestion_b_id).await.unwrap();
        assert_eq!(sb.status, "dismissed", "overlapping suggestion should be dismissed");

        let sc = get_suggestion(&conn, TEST_USER, &suggestion_c_id).await.unwrap();
        assert_eq!(sc.status, "pending", "non-overlapping suggestion should remain pending");
    }

    #[tokio::test]
    async fn should_store_and_check_dismissed_pairs() {
        let conn = setup_run_test().await;

        assert!(!is_dismissed_pair(&conn, TEST_USER, "a", "b").await.unwrap());

        store_dismissed_pair(&conn, TEST_USER, "a", "b").await.unwrap();

        assert!(is_dismissed_pair(&conn, TEST_USER, "a", "b").await.unwrap());
        assert!(is_dismissed_pair(&conn, TEST_USER, "b", "a").await.unwrap());

        assert!(!is_dismissed_pair(&conn, TEST_USER, "a", "c").await.unwrap());
        assert!(!is_dismissed_pair(&conn, "other-user", "a", "b").await.unwrap());
    }

    #[tokio::test]
    async fn should_list_dismissed_pairs() {
        let conn = setup_run_test().await;

        store_dismissed_pair(&conn, TEST_USER, "b", "a").await.unwrap();
        store_dismissed_pair(&conn, TEST_USER, "c", "d").await.unwrap();

        let pairs = get_dismissed_pairs(&conn, TEST_USER).await.unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&("a".to_string(), "b".to_string())));
        assert!(pairs.contains(&("c".to_string(), "d".to_string())));
    }

    #[tokio::test]
    async fn should_not_duplicate_dismissed_pair() {
        let conn = setup_run_test().await;

        store_dismissed_pair(&conn, TEST_USER, "a", "b").await.unwrap();
        store_dismissed_pair(&conn, TEST_USER, "a", "b").await.unwrap();
        store_dismissed_pair(&conn, TEST_USER, "b", "a").await.unwrap();

        let pairs = get_dismissed_pairs(&conn, TEST_USER).await.unwrap();
        assert_eq!(pairs.len(), 1);
    }
}
