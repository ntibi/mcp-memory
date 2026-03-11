use std::collections::HashMap;

use askama::Template;
use askama_web::WebTemplate;
use chrono::{DateTime, Utc};
use memory_core::curation::{CurationRun, CurationSettings, RunProgress};
use memory_core::memory::Memory;
use memory_core::users::{ApiKey, User};

pub struct ScoreBreakdown {
    pub relevance: String,
    pub confidence: String,
    pub recency: String,
    pub total: String,
}

pub struct MemoryCard {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub helpful: u64,
    pub harmful: u64,
    pub vote_ratio: String,
    pub age: String,
    pub created_at: DateTime<Utc>,
    pub score: Option<ScoreBreakdown>,
}

impl MemoryCard {
    pub fn from_memory(memory: memory_core::memory::Memory, helpful: u64, harmful: u64) -> Self {
        let age = format_age(memory.created_at);
        let vote_ratio = if helpful + harmful == 0 {
            "no votes".to_string()
        } else {
            let ratio = (helpful as f64) / ((helpful + harmful) as f64);
            format!("{:.0}%", ratio * 100.0)
        };
        Self {
            id: memory.id,
            content: memory.content,
            tags: memory.tags,
            helpful,
            harmful,
            vote_ratio,
            age,
            created_at: memory.created_at,
            score: None,
        }
    }

    pub fn with_score(mut self, relevance: f64, confidence: f64, recency: f64, total: f64) -> Self {
        self.score = Some(ScoreBreakdown {
            relevance: format!("{:.2}", relevance),
            confidence: format!("{:.2}", confidence),
            recency: format!("{:.2}", recency),
            total: format!("{:.2}", total),
        });
        self
    }
}

fn format_age(dt: DateTime<Utc>) -> String {
    let dur = Utc::now() - dt;
    if dur.num_days() > 30 {
        format!("{}mo ago", dur.num_days() / 30)
    } else if dur.num_days() > 0 {
        format!("{}d ago", dur.num_days())
    } else if dur.num_hours() > 0 {
        format!("{}h ago", dur.num_hours())
    } else {
        "just now".to_string()
    }
}

#[derive(Template, WebTemplate)]
#[template(path = "layout.html")]
pub struct LayoutTemplate {
    pub is_admin: bool,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/card_grid.html")]
pub struct CardGridTemplate {
    pub memories: Vec<MemoryCard>,
    pub has_more: bool,
    pub next_cursor: String,
    pub query: String,
    pub tag: String,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/card.html")]
pub struct CardTemplate {
    pub card: MemoryCard,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/vote_buttons.html")]
pub struct VoteButtonsTemplate {
    pub card: MemoryCard,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/tag_sidebar.html")]
pub struct TagSidebarTemplate {
    pub tags: Vec<(String, usize)>,
    pub total_count: usize,
    pub active_tags: Vec<String>,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/card_edit.html")]
pub struct CardEditTemplate {
    pub card: MemoryCard,
}

#[derive(Template, WebTemplate)]
#[template(path = "login.html")]
pub struct LoginTemplate;

#[derive(Template, WebTemplate)]
#[template(path = "admin/users.html")]
pub struct AdminUsersPageTemplate;

#[derive(Template, WebTemplate)]
#[template(path = "admin/users_list.html")]
pub struct AdminUsersListTemplate {
    pub users: Vec<User>,
    pub admin_ids: Vec<String>,
    pub keys: HashMap<String, Vec<ApiKey>>,
}

#[derive(Template, WebTemplate)]
#[template(path = "admin/stats.html")]
pub struct AdminStatsPageTemplate;

#[derive(Template, WebTemplate)]
#[template(path = "admin/stats_data.html")]
pub struct AdminStatsDataTemplate {
    pub stats: Vec<(String, usize)>,
}

#[derive(Template, WebTemplate)]
#[template(path = "curation/settings.html")]
pub struct CurationSettingsTemplate {
    pub settings: CurationSettings,
    pub is_admin: bool,
    pub schedule_days: Vec<u8>,
}

#[derive(Template, WebTemplate)]
#[template(path = "curation/dashboard.html")]
pub struct CurationDashboardTemplate {
    pub is_admin: bool,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/curation_status.html")]
pub struct CurationStatusTemplate {
    pub progress: Option<RunProgress>,
    pub last_run: Option<CurationRun>,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/curation_runs.html")]
pub struct CurationRunsTemplate {
    pub runs: Vec<CurationRun>,
}

pub struct SuggestionCard {
    pub id: String,
    pub action: String,
    pub reasoning: String,
    pub proposed_content: String,
    pub proposed_tags: Vec<String>,
    pub source_memories: Vec<Memory>,
    pub added_tags: Vec<String>,
    pub removed_tags: Vec<String>,
    pub unchanged_tags: Vec<String>,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/curation_suggestions.html")]
pub struct CurationSuggestionsTemplate {
    pub suggestions: Vec<SuggestionCard>,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/curation_indicator.html")]
pub struct CurationIndicatorTemplate {
    pub is_running: bool,
    pub pending_count: usize,
}
