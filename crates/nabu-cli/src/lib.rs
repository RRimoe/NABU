//! Nabu outside its window: the `nabu` command line and the MCP server.
//!
//! Everything except panoramas goes through the running editor over the local
//! bridge ([`bridge`]), so a script gets exactly the pipeline, models and
//! results the interface gives. When the editor is not running, it is started
//! in the background and closed again when the command is done.
//!
//! The desktop application calls [`run`] itself when it is started with one of
//! these subcommands, so the installed app doubles as the command line.

pub mod bridge;
pub mod mcp;
mod panorama;

use bridge::Client;
use serde_json::{json, Map, Value};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// The subcommands; anything else starts the editor normally.
pub const SUBCOMMANDS: &[&str] = &["mcp", "commands", "call", "batch", "panorama", "help", "--help", "-h"];

/// Whether these arguments ask for the command line rather than the window.
pub fn wants_cli(args: &[String]) -> bool {
    args.first().is_some_and(|a| SUBCOMMANDS.contains(&a.as_str()))
}

/// How to start the editor when it is not running.
pub enum Launcher {
    /// This executable is the editor: start another copy of it.
    CurrentExe,
    /// A separate `nabu` binary: the editor is wherever `NABU_APP` points, or
    /// in the usual install location.
    FromEnvironment,
}

/// Argument that starts the editor without showing its window.
pub const BACKGROUND_FLAG: &str = "--background";

impl Launcher {
    fn executable(&self) -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("NABU_APP") {
            return Some(PathBuf::from(path));
        }
        match self {
            Launcher::CurrentExe => std::env::current_exe().ok(),
            Launcher::FromEnvironment => [
                "/Applications/nabu.app/Contents/MacOS/nabu",
                "/usr/bin/nabu-app",
                "/usr/local/bin/nabu-app",
            ]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists()),
        }
    }
}

/// A connection to the editor, and the editor itself when this process
/// started it, so it can be closed afterwards.
pub struct Session {
    pub client: Client,
    started: Option<Child>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // An editor started only to run this command has no window anyone is
        // using; leaving it behind would be a process the user never asked for.
        if let Some(mut child) = self.started.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn bridge_file() -> io::Result<PathBuf> {
    bridge::info_path().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no home folder"))
}

/// Connects to the running editor, starting it when it is not running.
pub fn connect(launcher: &Launcher, background: bool) -> io::Result<Session> {
    let path = bridge_file()?;
    if let Ok(client) = Client::connect(&path) {
        return Ok(Session { client, started: None });
    }

    let executable = launcher.executable().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Nabu is not running. Open it, or set NABU_APP to the path of the application.",
        )
    })?;
    let mut command = Command::new(&executable);
    if background {
        command.arg(BACKGROUND_FLAG);
    }
    let mut child = command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .spawn()
        .map_err(|err| io::Error::new(err.kind(), format!("{}: {err}", executable.display())))?;

    // The editor writes the bridge file once its window has loaded the
    // registry; a cold start with models to index can take a while.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Ok(client) = Client::connect(&path) {
            return Ok(Session { client, started: background.then_some(child) });
        }
        if let Ok(Some(status)) = child.try_wait() {
            return Err(io::Error::other(format!("Nabu exited during start-up ({status})")));
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            return Err(io::Error::new(io::ErrorKind::TimedOut, "Nabu did not start in time"));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

const USAGE: &str = "Nabu from the command line

USAGE
  nabu commands                         List what the editor can do
  nabu call <command> [key=value ...]   Run one command on the editor
  nabu batch [options] <files...>       Edit and export many photos
  nabu panorama [options] <out> <in...> Stitch a panorama (no editor needed)
  nabu mcp                              Serve the editor to AI agents over MCP (stdio)

CALL
  Values are read as JSON when they parse (20, true, [\"a\"], {...}), as text
  otherwise. Relative paths are resolved against the current folder.
    nabu call open_image path=photo.jpg
    nabu call adjust property=exposure amount=15
    nabu call export_image path=out.jpg quality=0.9

BATCH
  --steps <file.json|json>   Commands run on each photo, e.g.
                             '[{\"command\":\"apply_look\",\"args\":{\"look\":\"portra\"}}]'
  --look <id>                Shorthand for an apply_look step
  --out <folder>             Where results go (required; created if missing)
  --format jpeg|png|webp     Output format (default jpeg)
  --quality <0..1>           JPEG/WebP quality (default 0.92)
  --scale <0..1>             Output size relative to each photo (default 1)
  --suffix <text>            Added to output names (default -nabu)

PANORAMA
  --projection auto|rectilinear|cylindrical|spherical
  --no-crop                  Keep the ragged edges
  --max-mp <n>               Largest output in megapixels (default 60)

MCP
  Add to an MCP client (Claude Desktop, Claude Code, ...) as a stdio server
  whose command is this program with the argument `mcp`.
";

/// Runs a subcommand and returns the process exit code.
pub fn run(args: &[String], launcher: &Launcher) -> i32 {
    let Some((command, rest)) = args.split_first() else {
        print!("{USAGE}");
        return 0;
    };
    let outcome = match command.as_str() {
        "mcp" => run_mcp(launcher),
        "commands" => list_commands(launcher),
        "call" => call(rest, launcher),
        "batch" => batch(rest, launcher),
        "panorama" => panorama::run(rest),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(0)
        }
        other => Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unknown command \"{other}\"; try `nabu help`"))),
    };
    match outcome {
        Ok(code) => code,
        Err(err) => {
            eprintln!("nabu: {err}");
            2
        }
    }
}

fn run_mcp(launcher: &Launcher) -> io::Result<i32> {
    // The agent's edits are worth watching: the editor opens with its window.
    let mut session: Option<Session> = None;
    let connector = || -> io::Result<Client> {
        if session.is_none() {
            session = Some(connect(launcher, false)?);
        }
        Ok(session.as_ref().map(|s| s.client.clone()).expect("connected"))
    };
    let stdin = io::stdin();
    mcp::Server::new(connector).run(stdin.lock(), io::stdout().lock())?;
    Ok(0)
}

fn list_commands(launcher: &Launcher) -> io::Result<i32> {
    let session = connect(launcher, true)?;
    let commands = session.client.commands()?;
    for command in commands.as_array().into_iter().flatten() {
        let name = command.get("name").and_then(Value::as_str).unwrap_or_default();
        let description = command.get("description").and_then(Value::as_str).unwrap_or_default();
        let params: Vec<String> = command
            .pointer("/inputSchema/properties")
            .and_then(Value::as_object)
            .map(|p| p.keys().cloned().collect())
            .unwrap_or_default();
        println!("{name}({})\n    {description}\n", params.join(", "));
    }
    Ok(0)
}

/// Keys whose values are paths, made absolute because the editor does not
/// share this process's working folder.
const PATH_KEYS: &[&str] = &["path", "paths", "files", "output_dir"];

fn absolute(path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        return path.to_string();
    }
    let joined = std::env::current_dir().map(|dir| dir.join(p)).unwrap_or_else(|_| p.to_path_buf());
    // `../photos/a.jpg` reads better in messages as the path it names.
    std::fs::canonicalize(&joined)
        .unwrap_or(joined)
        .to_string_lossy()
        .into_owned()
}

fn absolutize(key: &str, value: Value) -> Value {
    if !PATH_KEYS.contains(&key) {
        return value;
    }
    match value {
        Value::String(s) => Value::String(absolute(&s)),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|v| match v {
                    Value::String(s) => Value::String(absolute(&s)),
                    other => other,
                })
                .collect(),
        ),
        other => other,
    }
}

/// `key=value` pairs as command arguments.
pub fn parse_arguments(pairs: &[String]) -> io::Result<Map<String, Value>> {
    let mut arguments = Map::new();
    for pair in pairs {
        let (key, raw) = pair
            .split_once('=')
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("expected key=value, got \"{pair}\"")))?;
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        arguments.insert(key.to_string(), absolutize(key, value));
    }
    Ok(arguments)
}

fn print_result(result: &Value) -> i32 {
    let ok = result.get("ok").and_then(Value::as_bool).unwrap_or(false);
    if let Some(message) = result.get("message").and_then(Value::as_str) {
        if ok {
            eprintln!("{message}");
        } else {
            eprintln!("nabu: {message}");
        }
    }
    // Data goes to stdout on its own, so it can be piped into jq.
    if let Some(data) = result.get("data").filter(|d| !d.is_null()) {
        println!("{}", serde_json::to_string_pretty(data).unwrap_or_default());
    }
    if ok {
        0
    } else {
        1
    }
}

fn call(args: &[String], launcher: &Launcher) -> io::Result<i32> {
    let (name, pairs) = args
        .split_first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "which command? try `nabu commands`"))?;
    let arguments = parse_arguments(pairs)?;
    let session = connect(launcher, true)?;
    let result = session.client.call(name, Value::Object(arguments))?;
    Ok(print_result(&result))
}

/// Options of `nabu batch`, parsed.
#[derive(Debug, PartialEq)]
pub struct BatchArgs {
    pub arguments: Value,
}

pub fn parse_batch(args: &[String]) -> io::Result<BatchArgs> {
    let invalid = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
    let mut steps: Vec<Value> = Vec::new();
    let mut files = Vec::new();
    let mut extra = Map::new();
    let mut out = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| iter.next().cloned().ok_or_else(|| invalid(format!("{name} needs a value")));
        match arg.as_str() {
            "--steps" => {
                let raw = value("--steps")?;
                let text = if Path::new(&raw).is_file() { std::fs::read_to_string(&raw)? } else { raw };
                let parsed: Value = serde_json::from_str(&text).map_err(|e| invalid(format!("--steps: {e}")))?;
                match parsed {
                    Value::Array(items) => steps.extend(items),
                    Value::Object(ref o) if o.contains_key("steps") => {
                        steps.extend(o["steps"].as_array().cloned().unwrap_or_default())
                    }
                    _ => return Err(invalid("--steps must be a JSON array of steps".into())),
                }
            }
            "--look" => steps.push(json!({ "command": "apply_look", "args": { "look": value("--look")? } })),
            "--out" => out = Some(absolute(&value("--out")?)),
            "--format" => {
                extra.insert("format".into(), Value::String(value("--format")?));
            }
            "--suffix" => {
                extra.insert("suffix".into(), Value::String(value("--suffix")?));
            }
            "--quality" | "--scale" => {
                let raw = value(arg)?;
                let number: f64 = raw.parse().map_err(|_| invalid(format!("{arg}: \"{raw}\" is not a number")))?;
                extra.insert(arg.trim_start_matches("--").into(), json!(number));
            }
            flag if flag.starts_with("--") => return Err(invalid(format!("unknown option {flag}"))),
            file => files.push(Value::String(absolute(file))),
        }
    }
    let out = out.ok_or_else(|| invalid("--out <folder> is required".into()))?;
    if files.is_empty() {
        return Err(invalid("no photos given".into()));
    }
    let mut arguments = Map::new();
    arguments.insert("files".into(), Value::Array(files));
    arguments.insert("steps".into(), Value::Array(steps));
    arguments.insert("output_dir".into(), Value::String(out));
    arguments.extend(extra);
    Ok(BatchArgs { arguments: Value::Object(arguments) })
}

fn batch(args: &[String], launcher: &Launcher) -> io::Result<i32> {
    let parsed = parse_batch(args)?;
    if let Some(dir) = parsed.arguments.get("output_dir").and_then(Value::as_str) {
        std::fs::create_dir_all(dir)?;
    }
    let session = connect(launcher, true)?;
    let result = session.client.call("batch", parsed.arguments)?;
    for item in result.get("data").and_then(Value::as_array).into_iter().flatten() {
        let file = item.get("file").and_then(Value::as_str).unwrap_or_default();
        match item.get("output").and_then(Value::as_str) {
            Some(output) if item.get("ok").and_then(Value::as_bool) == Some(true) => println!("{file} -> {output}"),
            _ => println!("{file}: {}", item.get("error").and_then(Value::as_str).unwrap_or("failed")),
        }
    }
    let ok = result.get("ok").and_then(Value::as_bool).unwrap_or(false);
    if let Some(message) = result.get("message").and_then(Value::as_str) {
        eprintln!("{message}");
    }
    Ok(if ok { 0 } else { 1 })
}
