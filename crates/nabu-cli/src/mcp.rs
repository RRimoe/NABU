//! Nabu as an MCP server, so an agent such as Claude can edit photos with the
//! same commands the editor's own buttons run.
//!
//! Model Context Protocol over stdio: one JSON-RPC message per line in, one per
//! line out. The tools are the editor's command registry, fetched from the
//! running app, so a command added there shows up here without touching this
//! file.

use crate::bridge::Client;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

/// Protocol revisions this server speaks, newest first.
const VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "Nabu is a local photo editor. Every command works on the active document: \
open a photo with open_image, look at it with get_preview, find objects with analyze_scene, edit, \
then export_image to save the result. Edits are undoable with undo. Paths must be absolute. \
To remove something analyze_scene did not name, select it with select_region (coordinates from \
get_preview, scaled to the document) and call remove_selection.";

/// Where the server gets its connection to the editor. Called lazily, on the
/// first request that needs the app, so `initialize` answers even before the
/// editor has finished starting.
pub trait Connector {
    fn connect(&mut self) -> io::Result<Client>;
}

impl<F: FnMut() -> io::Result<Client>> Connector for F {
    fn connect(&mut self) -> io::Result<Client> {
        self()
    }
}

pub struct Server<C: Connector> {
    connector: C,
    client: Option<Client>,
}

fn error(id: &Value, code: i64, message: impl Into<String>) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message.into() } })
}

fn reply(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// Turns a registry result into MCP tool content: the sentence, the data as
/// JSON for the agent to reason on, and the picture when there is one.
pub fn tool_result(result: &Value) -> Value {
    let ok = result.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let mut text = result.get("message").and_then(Value::as_str).unwrap_or_default().to_string();
    if let Some(data) = result.get("data").filter(|d| !d.is_null()) {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&serde_json::to_string_pretty(data).unwrap_or_default());
    }
    let mut content = vec![json!({ "type": "text", "text": text })];
    if let Some(image) = result.get("image").filter(|i| i.is_object()) {
        content.push(json!({
            "type": "image",
            "data": image.get("base64").cloned().unwrap_or_default(),
            "mimeType": image.get("mimeType").cloned().unwrap_or_else(|| json!("image/jpeg")),
        }));
    }
    json!({ "content": content, "isError": !ok })
}

/// Registry descriptors as MCP tools.
pub fn tools(commands: &Value) -> Value {
    let list: Vec<Value> = commands
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|c| {
                    let read_only = c.get("readOnly").and_then(Value::as_bool).unwrap_or(false);
                    json!({
                        "name": c.get("name"),
                        "description": c.get("description"),
                        "inputSchema": c.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object" })),
                        "annotations": { "readOnlyHint": read_only },
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json!({ "tools": list })
}

impl<C: Connector> Server<C> {
    pub fn new(connector: C) -> Self {
        Server { connector, client: None }
    }

    fn client(&mut self) -> io::Result<Client> {
        if let Some(client) = &self.client {
            return Ok(client.clone());
        }
        let client = self.connector.connect()?;
        self.client = Some(client.clone());
        Ok(client)
    }

    /// Runs a request through the editor, reconnecting once if it went away
    /// (closed and reopened by the user, say) since the last call.
    fn with_client<T>(&mut self, f: impl Fn(&Client) -> io::Result<T>) -> io::Result<T> {
        let client = self.client()?;
        match f(&client) {
            Ok(value) => Ok(value),
            Err(_) => {
                self.client = None;
                let client = self.client()?;
                f(&client)
            }
        }
    }

    /// Answers one message; `None` for notifications, which get no answer.
    pub fn handle(&mut self, message: &Value) -> Option<Value> {
        let id = message.get("id").cloned()?;
        let method = message.get("method").and_then(Value::as_str).unwrap_or_default();
        let params = message.get("params").cloned().unwrap_or_else(|| json!({}));

        Some(match method {
            "initialize" => {
                let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or_default();
                let version = VERSIONS.iter().find(|v| **v == asked).unwrap_or(&VERSIONS[0]);
                reply(
                    &id,
                    json!({
                        "protocolVersion": version,
                        "capabilities": { "tools": { "listChanged": false } },
                        "serverInfo": { "name": "nabu", "version": env!("CARGO_PKG_VERSION") },
                        "instructions": INSTRUCTIONS,
                    }),
                )
            }
            "ping" => reply(&id, json!({})),
            "tools/list" => match self.with_client(|c| c.commands()) {
                Ok(commands) => reply(&id, tools(&commands)),
                Err(err) => error(&id, -32603, format!("Nabu is not running and could not be started: {err}")),
            },
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
                let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
                match self.with_client(|c| c.call(&name, arguments.clone())) {
                    Ok(result) => reply(&id, tool_result(&result)),
                    Err(err) => reply(
                        &id,
                        json!({
                            "content": [{ "type": "text", "text": format!("Nabu could not be reached: {err}") }],
                            "isError": true,
                        }),
                    ),
                }
            }
            _ => error(&id, -32601, format!("Method not found: {method}")),
        })
    }

    /// Serves until the client closes stdin.
    pub fn run(&mut self, input: impl BufRead, mut output: impl Write) -> io::Result<()> {
        for line in input.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let answer = match serde_json::from_str::<Value>(&line) {
                Ok(Value::Array(batch)) => {
                    let answers: Vec<Value> = batch.iter().filter_map(|m| self.handle(m)).collect();
                    (!answers.is_empty()).then(|| Value::Array(answers))
                }
                Ok(message) => self.handle(&message),
                Err(err) => Some(error(&Value::Null, -32700, format!("Parse error: {err}"))),
            };
            if let Some(answer) = answer {
                writeln!(output, "{}", serde_json::to_string(&answer)?)?;
                output.flush()?;
            }
        }
        Ok(())
    }
}
