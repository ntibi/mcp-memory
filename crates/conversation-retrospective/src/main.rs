mod claude;
mod condense;
mod display;
mod filter;
mod memory_extract;
mod parser;
mod state;
mod store;
mod vote;

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Parser;

#[derive(Parser)]
#[command(name = "retro")]
enum Cli {
    Run(RunArgs),
    Store(StoreArgs),
    Vote(VoteArgs),
}

#[derive(clap::Args)]
struct RunArgs {
    #[arg(long, default_value = "10")]
    concurrency: usize,
    #[arg(long)]
    project: Option<String>,
    #[arg(long, default_value = "state.json")]
    state: String,
}

#[derive(clap::Args)]
struct StoreArgs {
    #[arg(long, default_value = "state.json")]
    state: String,
}

#[derive(clap::Args)]
struct VoteArgs {
    #[arg(long)]
    project: Option<String>,
    #[arg(long, default_value = "vote-state.json")]
    state: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli {
        Cli::Run(args) => run(args).await,
        Cli::Store(args) => store(args).await,
        Cli::Vote(args) => run_vote_cmd(args).await,
    }
}

async fn run(args: RunArgs) -> anyhow::Result<()> {
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("no home directory"))?
        .join(".claude/projects");

    let mut state = state::State::load(Path::new(&args.state))?;
    let projects = parser::discover_projects(&base)?;
    let mut display = display::Display::new();
    let state_path = PathBuf::from(&args.state);

    for (project_name, conv_paths) in &projects {
        if let Some(ref filter) = args.project {
            if !project_name.contains(filter) {
                continue;
            }
        }

        let mut progress = display::ProjectProgress {
            name: project_name.clone(),
            total: conv_paths.len(),
            handled: 0,
            skipped: 0,
            interesting: 0,
            learnings: 0,
        };

        for path in conv_paths {
            let session_id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if !state.is_processed(session_id) {
                continue;
            }
            progress.handled += 1;
            if let Some(cs) = state.conversations.get(session_id) {
                match cs.status {
                    state::Status::HeuristicSkipped | state::Status::NotInteresting => {
                        progress.skipped += 1;
                    }
                    _ => {
                        progress.interesting += 1;
                        progress.learnings += cs.learnings.len();
                    }
                }
            }
        }

        if progress.handled == progress.total {
            display.print_completed(&progress);
            continue;
        }

        display.print_active(&progress);

        let pending: Vec<_> = conv_paths
            .iter()
            .filter(|p| {
                let sid = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                !state.is_processed(sid)
            })
            .cloned()
            .collect();

        for path in &pending {
            let session_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            let conv = match parser::parse_conversation(path, project_name) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(session = %session_id, error = %e, "failed to parse conversation");
                    state.conversations.insert(
                        session_id.clone(),
                        state::ConversationState {
                            project: project_name.clone(),
                            session_id: session_id.clone(),
                            status: state::Status::Failed,
                            filter_reason: Some(format!("parse error: {e}")),
                            learnings: vec![],
                        },
                    );
                    progress.handled += 1;
                    progress.skipped += 1;
                    state.save(&state_path)?;
                    display.print_active(&progress);
                    continue;
                }
            };

            match filter::heuristic_filter(&conv) {
                filter::FilterResult::Skip(reason) => {
                    state.conversations.insert(
                        session_id.clone(),
                        state::ConversationState {
                            project: project_name.clone(),
                            session_id: session_id.clone(),
                            status: state::Status::HeuristicSkipped,
                            filter_reason: Some(reason),
                            learnings: vec![],
                        },
                    );
                    progress.handled += 1;
                    progress.skipped += 1;
                    state.save(&state_path)?;
                    display.print_active(&progress);
                    continue;
                }
                filter::FilterResult::Pass => {}
            }

            let condensed = condense::condense(&conv);

            match claude::filter_conversation(&condensed).await {
                Ok(response) if !response.interesting => {
                    state.conversations.insert(
                        session_id.clone(),
                        state::ConversationState {
                            project: project_name.clone(),
                            session_id: session_id.clone(),
                            status: state::Status::NotInteresting,
                            filter_reason: Some(response.reason),
                            learnings: vec![],
                        },
                    );
                    progress.handled += 1;
                    progress.skipped += 1;
                    state.save(&state_path)?;
                    display.print_active(&progress);
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(session = %session_id, error = %e, "haiku filter failed");
                    progress.handled += 1;
                    display.print_active(&progress);
                    continue;
                }
            }

            match claude::extract_learnings(&condensed).await {
                Ok(extraction) => {
                    let learnings: Vec<state::Learning> = extraction
                        .learnings
                        .into_iter()
                        .map(|l| state::Learning {
                            content: l.content,
                            tags: l.tags,
                            category: l.category,
                            approved: false,
                        })
                        .collect();

                    let learning_count = learnings.len();

                    state.conversations.insert(
                        session_id.clone(),
                        state::ConversationState {
                            project: project_name.clone(),
                            session_id: session_id.clone(),
                            status: state::Status::Extracted,
                            filter_reason: None,
                            learnings,
                        },
                    );

                    progress.handled += 1;
                    progress.interesting += 1;
                    progress.learnings += learning_count;
                }
                Err(e) => {
                    tracing::warn!(session = %session_id, error = %e, "sonnet extraction failed");
                    progress.handled += 1;
                }
            }

            state.save(&state_path)?;
            display.print_active(&progress);
        }

        display.print_completed(&progress);
    }

    println!();
    Ok(())
}

async fn run_vote_cmd(args: VoteArgs) -> anyhow::Result<()> {
    let memory_url =
        std::env::var("MEMORY_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let memory_key = std::env::var("MEMORY_API_KEY").context("MEMORY_API_KEY not set")?;
    let state_path = PathBuf::from(&args.state);

    vote::run_vote(
        &state_path,
        args.project.as_deref(),
        &memory_url,
        &memory_key,
    )
    .await
}

async fn store(args: StoreArgs) -> anyhow::Result<()> {
    let mut state = state::State::load(Path::new(&args.state))?;
    let memory_url =
        std::env::var("MEMORY_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let memory_key = std::env::var("MEMORY_API_KEY").context("MEMORY_API_KEY not set")?;
    let client = reqwest::Client::new();

    let mut stored = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for conv_state in state.conversations.values_mut() {
        if conv_state.status != state::Status::Extracted {
            continue;
        }

        let mut approved_count = 0;
        let mut success_count = 0;

        for learning in &conv_state.learnings {
            if !learning.approved {
                skipped += 1;
                continue;
            }
            approved_count += 1;
            match store::store_learnings(
                &client,
                &memory_url,
                &memory_key,
                &learning.content,
                &learning.tags,
            )
            .await
            {
                Ok(()) => {
                    stored += 1;
                    success_count += 1;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to store learning");
                    failed += 1;
                }
            }
        }

        if approved_count > 0 && success_count == approved_count {
            conv_state.status = state::Status::Stored;
        }
    }

    state.save(Path::new(&args.state))?;
    println!("stored {stored}, skipped {skipped} unapproved, {failed} failed");
    Ok(())
}
