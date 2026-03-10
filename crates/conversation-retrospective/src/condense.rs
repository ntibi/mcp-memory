use crate::parser::Conversation;

const MAX_CONDENSED_CHARS: usize = 30_000;

pub fn condense(conv: &Conversation) -> String {
    let formatted: String = conv
        .messages
        .iter()
        .map(|m| {
            let label = match m.role.as_str() {
                "user" => "User",
                "assistant" => "Assistant",
                other => other,
            };
            format!("{label}: {}\n\n", m.text)
        })
        .collect();

    if formatted.len() <= MAX_CONDENSED_CHARS {
        return formatted;
    }

    let head_budget = MAX_CONDENSED_CHARS * 60 / 100;
    let tail_budget = MAX_CONDENSED_CHARS * 40 / 100;

    let head_end = formatted
        .char_indices()
        .take_while(|(i, _)| *i < head_budget)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(head_budget);

    let tail_start_raw = formatted.len().saturating_sub(tail_budget);
    let tail_start = formatted[tail_start_raw..]
        .char_indices()
        .next()
        .map(|(i, _)| tail_start_raw + i)
        .unwrap_or(tail_start_raw);

    let head = &formatted[..head_end];
    let tail = &formatted[tail_start..];

    format!("{head}\n\n[...truncated...]\n\n{tail}")
}
