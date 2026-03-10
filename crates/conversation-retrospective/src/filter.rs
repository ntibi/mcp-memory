use crate::parser::Conversation;

pub enum FilterResult {
    Pass,
    Skip(String),
}

pub fn heuristic_filter(conv: &Conversation) -> FilterResult {
    let user_count = conv.messages.iter().filter(|m| m.role == "user").count();
    let assistant_count = conv.messages.iter().filter(|m| m.role == "assistant").count();

    if user_count <= 1 && assistant_count <= 1 {
        return FilterResult::Skip("too few messages (<=1 user AND <=1 assistant)".to_string());
    }

    let total_len: usize = conv.messages.iter().map(|m| m.text.len()).sum();
    if total_len < 500 {
        return FilterResult::Skip(format!("total text too short ({total_len} chars < 500)"));
    }

    FilterResult::Pass
}
