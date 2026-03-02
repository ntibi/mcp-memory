use askama::Template;
use askama_web::WebTemplate;
use chrono::{DateTime, Utc};

pub struct MemoryCard {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub helpful: u64,
    pub harmful: u64,
    pub confidence: String,
    pub age: String,
    pub created_at: DateTime<Utc>,
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
        }
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
pub struct LayoutTemplate;

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
    pub active_tag: String,
}

#[derive(Template, WebTemplate)]
#[template(path = "fragments/card_edit.html")]
pub struct CardEditTemplate {
    pub card: MemoryCard,
}
