use std::io::{Write, stdout};

pub struct ProjectProgress {
    pub name: String,
    pub total: usize,
    pub handled: usize,
    pub skipped: usize,
    pub interesting: usize,
    pub learnings: usize,
}

pub struct Display {
    has_active_line: bool,
}

fn clean_project_name(name: &str) -> String {
    if name == "-home-ntibi" {
        return "~".to_string();
    }
    if let Some(rest) = name.strip_prefix("-home-ntibi-ws-") {
        return rest.to_string();
    }
    if let Some(rest) = name.strip_prefix("-home-ntibi-") {
        return rest.to_string();
    }
    name.to_string()
}

impl Display {
    pub fn new() -> Self {
        Self { has_active_line: false }
    }

    pub fn print_completed(&mut self, p: &ProjectProgress) {
        let name = clean_project_name(&p.name);
        if self.has_active_line {
            print!("\x1b[2K\r");
        }
        println!(
            "\u{2713} {:<30} {:>3} conv  {:>3} skipped  {:>3} interesting  {:>3} learnings",
            name, p.total, p.skipped, p.interesting, p.learnings
        );
        self.has_active_line = false;
    }

    pub fn print_active(&mut self, p: &ProjectProgress) {
        let name = clean_project_name(&p.name);
        print!("\x1b[2K\r");
        print!(
            "  {:<30} {}/{} | {} skipped | {} interesting | {} learnings",
            name, p.handled, p.total, p.skipped, p.interesting, p.learnings
        );
        let _ = stdout().flush();
        self.has_active_line = true;
    }
}
