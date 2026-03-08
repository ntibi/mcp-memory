use std::collections::HashMap;

use askama::Template;
use askama_web::WebTemplate;
use chrono::{DateTime, Utc};
use memory_core::users::{ApiKey, User};

pub struct MemoryCard {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub helpful: u64,
    pub harmful: u64,
    pub confidence: String,
    pub age: String,
    pub created_at: DateTime<Utc>,
    pub score: String,
}

impl MemoryCard {
    pub fn from_memory(memory: memory_core::memory::Memory, helpful: u64, harmful: u64) -> Self {
        let age = format_age(memory.created_at);
        let confidence = if helpful + harmful == 0 {
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
            confidence,
            age,
            created_at: memory.created_at,
            score: String::new(),
        }
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = format!("{:.2}", score);
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
