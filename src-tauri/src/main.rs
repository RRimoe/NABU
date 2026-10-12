// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // `nabu mcp`, `nabu batch` and the other subcommands run the command line
    // in this same executable, so installing the editor installs the CLI too.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if nabu_cli::wants_cli(&args) {
        std::process::exit(nabu_cli::run(&args, &nabu_cli::Launcher::CurrentExe));
    }
    nabu_lib::run()
}
