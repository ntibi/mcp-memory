use std::collections::HashMap;
use std::sync::Arc;

use chrono::Datelike;
use memory_core::curation::{self, ScheduleWindow};
use tokio_util::sync::CancellationToken;

use crate::curation_worker::{self, ProgressMap, ProgressStatus};

pub struct Scheduler {
    conn: tokio_rusqlite::Connection,
    store: Arc<memory_core::memory::MemoryStore>,
    progress_map: ProgressMap,
    cancel_tokens: HashMap<String, CancellationToken>,  // user_id -> token
}

impl Scheduler {
    pub fn spawn(
        conn: tokio_rusqlite::Connection,
        store: Arc<memory_core::memory::MemoryStore>,
        progress_map: ProgressMap,
    ) -> Arc<tokio::sync::Mutex<Self>> {
        let scheduler = Arc::new(tokio::sync::Mutex::new(Self {
            conn,
            store,
            progress_map,
            cancel_tokens: HashMap::new(),
        }));

        let sched = scheduler.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                sched.lock().await.tick().await;
            }
        });

        scheduler
    }

    async fn tick(&mut self) {
        let enabled_users = match self.get_enabled_users().await {
            Ok(users) => users,
            Err(e) => {
                tracing::error!(error = %e, "failed to query enabled curation users");
                return;
            }
        };

        for user_id in enabled_users {
            let settings = match curation::get_settings(&self.conn, &user_id).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(user_id = %user_id, error = %e, "failed to load curation settings");
                    continue;
                }
            };

            if settings.api_key.is_none() {
                continue;
            }

            if !is_within_schedule(&settings.schedule_windows) {
                continue;
            }

            if let Ok(Some(_)) = curation::get_active_run(&self.conn, &user_id).await {
                continue;
            }

            if let Err(e) = self.start_run(&user_id).await {
                tracing::warn!(user_id = %user_id, error = %e, "failed to start scheduled curation run");
            }
        }

        self.cleanup_finished_tokens().await;
    }

    pub async fn start_run(&mut self, user_id: &str) -> anyhow::Result<String> {
        let settings = curation::get_settings(&self.conn, user_id).await?;

        anyhow::ensure!(settings.api_key.is_some(), "no api key configured");

        if let Some(active) = curation::get_active_run(&self.conn, user_id).await? {
            anyhow::bail!("run {} is already active", active.id);
        }

        let cancel = CancellationToken::new();
        self.cancel_tokens
            .insert(user_id.to_string(), cancel.clone());

        let conn = self.conn.clone();
        let store = self.store.clone();
        let progress_map = self.progress_map.clone();
        let uid = user_id.to_string();

        tokio::spawn(async move {
            curation_worker::execute_run(conn, store, settings, progress_map, cancel)
                .await;
            tracing::info!(user_id = %uid, "curation run finished");
        });

        tracing::info!(user_id = %user_id, "started curation run");
        Ok(user_id.to_string())
    }

    pub async fn cancel_run(&mut self, user_id: &str, run_id: &str) -> anyhow::Result<()> {
        let token = self
            .cancel_tokens
            .get(user_id)
            .ok_or_else(|| anyhow::anyhow!("no active run for user {user_id}"))?;

        token.cancel();

        if let Some(mut entry) = self.progress_map.get_mut(run_id) {
            entry.status = ProgressStatus::Cancelling;
        }

        tracing::info!(user_id = %user_id, run_id = %run_id, "cancellation requested for curation run");
        Ok(())
    }

    async fn get_enabled_users(&self) -> anyhow::Result<Vec<String>> {
        let users = self
            .conn
            .call(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT user_id FROM curation_settings WHERE enabled = 1 AND api_key IS NOT NULL",
                )?;
                let rows = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?;
        Ok(users)
    }

    async fn cleanup_finished_tokens(&mut self) {
        let mut finished = Vec::new();
        for user_id in self.cancel_tokens.keys() {
            match curation::get_active_run(&self.conn, user_id).await {
                Ok(None) => finished.push(user_id.clone()),
                _ => {}
            }
        }
        for user_id in finished {
            self.cancel_tokens.remove(&user_id);
        }
    }
}

fn is_within_schedule(windows: &[ScheduleWindow]) -> bool {
    let now = chrono::Local::now();
    let weekday = now.weekday().num_days_from_sunday() as u8;
    let current_time = now.format("%H:%M").to_string();

    windows.iter().any(|w| {
        w.days.contains(&weekday) && current_time >= w.start && current_time < w.end
    })
}
