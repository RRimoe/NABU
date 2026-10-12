//! The local bridge between the running editor and everything outside it.
//!
//! The editor listens on a random port of the loopback interface and writes the
//! port, with a secret token, to `bridge.json` in its data folder. Only a
//! process that can read that file (the same user) can call it, and nothing
//! outside the machine can reach the port at all.
//!
//! The protocol is two requests of plain HTTP/1.1 with JSON bodies:
//! `GET /commands` lists the command registry, `POST /call` runs one command.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Identifier of the desktop application: its data folder is named after it.
pub const APP_IDENTIFIER: &str = "com.editor.nabu";

/// Largest request body the bridge accepts. Commands carry arguments and
/// paths, never pixels, so this is generous.
const MAX_BODY: usize = 16 * 1024 * 1024;

/// What the editor publishes so others can find it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BridgeInfo {
    pub port: u16,
    pub token: String,
    pub pid: u32,
}

/// What the bridge does with a call. The editor implements it by handing the
/// call to its interface; tests implement it with a fake.
pub trait Executor: Send + Sync + 'static {
    /// The command registry, as a JSON array of descriptors.
    fn commands(&self) -> Value;
    /// Runs one command and returns `{ ok, message, data?, image? }`.
    fn call(&self, name: &str, arguments: Value) -> Value;
}

/// The application's data folder, where Tauri also keeps its own files.
pub fn data_dir() -> Option<PathBuf> {
    let home = || std::env::var_os("HOME").map(PathBuf::from);
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library").join("Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| home().map(|h| h.join(".local").join("share")))
    }?;
    Some(base.join(APP_IDENTIFIER))
}

/// Where the bridge file lives. `NABU_BRIDGE_FILE` overrides it, which is how
/// tests and side-by-side installs keep apart.
pub fn info_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("NABU_BRIDGE_FILE") {
        return Some(PathBuf::from(path));
    }
    data_dir().map(|d| d.join("bridge.json"))
}

/// A token nobody can guess: 256 bits from the operating system's randomness,
/// which is what seeds every `RandomState`.
fn new_token() -> String {
    (0..4)
        .map(|i| {
            let mut hasher = RandomState::new().build_hasher();
            hasher.write_u64(i);
            format!("{:016x}", hasher.finish())
        })
        .collect()
}

fn write_info(path: &Path, info: &BridgeInfo) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Readable by this user only: the token is the whole of the security.
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(serde_json::to_string_pretty(info)?.as_bytes())
}

/// Removes the bridge file, if it is still ours, when the editor quits.
pub fn withdraw(path: &Path, info: &BridgeInfo) {
    let ours = std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<BridgeInfo>(&text).ok())
        .is_some_and(|current| current == *info);
    if ours {
        let _ = std::fs::remove_file(path);
    }
}

/// Starts listening and publishes where. Each connection gets its own thread;
/// the executor decides whether calls may overlap.
pub fn serve(executor: Arc<dyn Executor>, path: &Path) -> io::Result<BridgeInfo> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let info = BridgeInfo {
        port: listener.local_addr()?.port(),
        token: new_token(),
        pid: std::process::id(),
    };
    write_info(path, &info)?;

    let token = info.token.clone();
    std::thread::Builder::new()
        .name("nabu-bridge".into())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                let executor = executor.clone();
                let token = token.clone();
                let _ = std::thread::Builder::new()
                    .name("nabu-bridge-call".into())
                    .spawn(move || {
                        if let Err(err) = handle(stream, executor.as_ref(), &token) {
                            eprintln!("[nabu] bridge request failed: {err}");
                        }
                    });
            }
        })?;
    Ok(info)
}

struct Request {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> io::Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut length = 0usize;
    let mut authorization = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            let value = value.trim();
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => {
                    length = value
                        .parse()
                        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad length"))?
                }
                "authorization" => authorization = Some(value.to_string()),
                _ => {}
            }
        }
    }
    if length > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "body too large"));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Request { method, path, authorization, body })
}

fn respond(stream: &mut TcpStream, status: &str, body: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(body)?;
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    stream.flush()
}

fn handle(mut stream: TcpStream, executor: &dyn Executor, token: &str) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let request = read_request(&mut stream)?;

    // A web page can make the browser send requests to localhost; without the
    // token, which it cannot read from disk, it gets nowhere.
    let expected = format!("Bearer {token}");
    if request.authorization.as_deref() != Some(expected.as_str()) {
        return respond(&mut stream, "401 Unauthorized", &json!({ "error": "unauthorized" }));
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/commands") => respond(&mut stream, "200 OK", &executor.commands()),
        ("POST", "/call") => {
            let call: Value = match serde_json::from_slice(&request.body) {
                Ok(value) => value,
                Err(err) => {
                    return respond(&mut stream, "400 Bad Request", &json!({ "error": err.to_string() }))
                }
            };
            let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
            let arguments = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let result = executor.call(name, arguments);
            respond(&mut stream, "200 OK", &result)
        }
        _ => respond(&mut stream, "404 Not Found", &json!({ "error": "not found" })),
    }
}

/// A connection to the running editor.
#[derive(Debug, Clone)]
pub struct Client {
    pub info: BridgeInfo,
}

impl Client {
    /// Connects through the bridge file, or fails when the editor is not
    /// running (no file, or a file left behind by one that crashed).
    pub fn connect(path: &Path) -> io::Result<Client> {
        let text = std::fs::read_to_string(path)?;
        let info: BridgeInfo = serde_json::from_str(&text)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let client = Client { info };
        client.request("GET", "/commands", None, Duration::from_secs(5))?;
        Ok(client)
    }

    fn request(&self, method: &str, path: &str, body: Option<&Value>, timeout: Duration) -> io::Result<Value> {
        let address = SocketAddr::from((Ipv4Addr::LOCALHOST, self.info.port));
        let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(3))?;
        stream.set_read_timeout(Some(timeout))?;
        let body = body.map(serde_json::to_vec).transpose()?.unwrap_or_default();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.info.token,
            body.len()
        )?;
        stream.write_all(&body)?;
        stream.flush()?;

        let mut response = Vec::new();
        stream.read_to_end(&mut response)?;
        let split = response
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed response"))?;
        let status = String::from_utf8_lossy(&response[..split]);
        let ok = status.lines().next().is_some_and(|l| l.split_whitespace().nth(1) == Some("200"));
        let value: Value = serde_json::from_slice(&response[split + 4..])
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        if !ok {
            let message = value.get("error").and_then(Value::as_str).unwrap_or("request refused");
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, message.to_string()));
        }
        Ok(value)
    }

    pub fn commands(&self) -> io::Result<Value> {
        self.request("GET", "/commands", None, Duration::from_secs(30))
    }

    /// Runs one command. Batches and AI edits take a while, so the wait is long.
    pub fn call(&self, name: &str, arguments: Value) -> io::Result<Value> {
        self.request(
            "POST",
            "/call",
            Some(&json!({ "name": name, "arguments": arguments })),
            Duration::from_secs(6 * 60 * 60),
        )
    }
}
