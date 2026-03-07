pub fn parse_comma_separated(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}
