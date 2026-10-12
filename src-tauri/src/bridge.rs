// src-tauri/src/bridge.rs
//! The editor's end of the local command bridge.
//!
//! Calls from the `nabu` command line and from MCP clients arrive on the
//! loopback server in `nabu_cli::bridge`, are handed to the interface as
//! events, and run through the same command registry as the editor's own
//! buttons. The interface answers with `command_result`, which wakes the
//! waiting request.
//!
//! Also here: reading and writing files by path, which those commands need and
//! the webview cannot do on its own.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use nabu_cli::bridge::Executor;
use serde_json::{json, Value};
use tauri::ipc::{InvokeBody, Request, Response};
use tauri::{AppHandle, Emitter, State};

use crate::ai::error::{self, NabuError};

/// How long a call waits for the interface to load its registry.
const READY_TIMEOUT: Duration = Duration::from_secs(90);
/// How long one command may run: long enough for a big batch.
const CALL_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

pub struct EditorBridge {
    app: AppHandle,
    /// The registry as the interface last published it; `None` until loaded.
    commands: Mutex<Option<Value>>,
    ready: Condvar,
    pending: Mutex<HashMap<u64, mpsc::Sender<Value>>>,
    next_id: AtomicU64,
}

impl EditorBridge {
    pub fn new(app: AppHandle) -> Self {
        Self {
            app,
            commands: Mutex::new(None),
            ready: Condvar::new(),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    fn registry(&self) -> Option<Value> {
        let guard = self.commands.lock().unwrap_or_else(|e| e.into_inner());
        let (guard, _) = self
            .ready
            .wait_timeout_while(guard, READY_TIMEOUT, |c| c.is_none())
            .unwrap_or_else(|e| e.into_inner());
        guard.clone()
    }
}

fn failure(message: impl Into<String>) -> Value {
    json!({ "ok": false, "message": message.into() })
}

impl Executor for EditorBridge {
    fn commands(&self) -> Value {
        self.registry().unwrap_or_else(|| json!([]))
    }

    fn call(&self, name: &str, arguments: Value) -> Value {
        if self.registry().is_none() {
            return failure("The editor window has not finished loading.");
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::channel();
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, sender);

        let sent = self.app.emit_to(
            "main",
            "nabu-command",
            json!({ "id": id, "name": name, "arguments": arguments }),
        );
        let result = match sent {
            Ok(()) => receiver
                .recv_timeout(CALL_TIMEOUT)
                .unwrap_or_else(|_| failure("The editor did not answer.")),
            Err(err) => failure(format!("The editor could not be reached: {err}")),
        };
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        result
    }
}

/// The interface has loaded and publishes its command registry.
#[tauri::command]
pub fn bridge_ready(bridge: State<'_, Arc<EditorBridge>>, commands: Value) {
    *bridge.commands.lock().unwrap_or_else(|e| e.into_inner()) = Some(commands);
    bridge.ready.notify_all();
}

/// The interface finished a command.
#[tauri::command]
pub fn command_result(bridge: State<'_, Arc<EditorBridge>>, id: u64, result: Value) {
    if let Some(sender) = bridge
        .pending
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id)
    {
        let _ = sender.send(result);
    }
}

/// Paths come from scripts run in another folder: a relative one would be
/// resolved against wherever the editor happened to start, which is never
/// what was meant.
fn absolute(path: &str) -> Result<&Path, NabuError> {
    let p = Path::new(path);
    if p.is_absolute() {
        Ok(p)
    } else {
        Err(error::relative_path(path))
    }
}

/// Reads a file for a command, returned as raw bytes.
#[tauri::command]
pub async fn read_file(path: String) -> Result<Response, NabuError> {
    let file = absolute(&path)?.to_path_buf();
    tauri::async_runtime::spawn_blocking(move || std::fs::read(&file))
        .await
        .map_err(error::unknown)?
        .map(Response::new)
        .map_err(|e| error::file_unreadable(&path, e))
}

/// Writes a file for a command. The bytes are the raw request body and the
/// path travels in a header, so a 50-megapixel export is not base64-encoded
/// on its way to the disk.
#[tauri::command]
pub async fn write_file(request: Request<'_>) -> Result<(), NabuError> {
    let InvokeBody::Raw(body) = request.body() else {
        return Err(error::unknown("expected a binary message"));
    };
    let path = request
        .headers()
        .get("x-nabu-path")
        .and_then(|v| v.to_str().ok())
        .map(percent_decode)
        .ok_or_else(|| error::unknown("missing x-nabu-path"))?;
    let file = absolute(&path)?.to_path_buf();
    let body = body.clone();
    tauri::async_runtime::spawn_blocking(move || {
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&file, body)
    })
    .await
    .map_err(error::unknown)?
    .map_err(|e| error::file_unwritable(&path, e))
}

/// Undoes `encodeURIComponent`, which is how a path with any characters fits
/// in an HTTP header.
fn percent_decode(text: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push(hi << 4 | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::percent_decode;

    #[test]
    fn decodes_encoded_paths() {
        assert_eq!(percent_decode("%2Fhome%2Fio%2Ffoto%20mare.jpg"), "/home/io/foto mare.jpg");
        assert_eq!(percent_decode("C%3A%5CFoto%5Cperch%C3%A9.jpg"), "C:\\Foto\\perché.jpg");
        assert_eq!(percent_decode("100%"), "100%");
    }
}
