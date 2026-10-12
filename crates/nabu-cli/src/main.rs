//! `nabu`: the editor from a terminal, a script or an AI agent.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(nabu_cli::run(&args, &nabu_cli::Launcher::FromEnvironment));
}
