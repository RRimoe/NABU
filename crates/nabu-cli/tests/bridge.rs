//! The bridge, the MCP server and argument parsing, against a fake editor.

use nabu_cli::bridge::{serve, Client, Executor};
use nabu_cli::mcp::Server;
use nabu_cli::{parse_arguments, parse_batch};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;

struct Fake;

impl Executor for Fake {
    fn commands(&self) -> Value {
        json!([{
            "name": "adjust",
            "description": "Change one adjustment.",
            "readOnly": false,
            "inputSchema": { "type": "object", "properties": { "amount": { "type": "number" } }, "required": ["amount"] }
        }, {
            "name": "get_preview",
            "description": "Look at the photo.",
            "readOnly": true,
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        }])
    }

    fn call(&self, name: &str, arguments: Value) -> Value {
        match name {
            "adjust" => json!({ "ok": true, "message": "Exposure: +20", "data": { "echo": arguments } }),
            "get_preview" => json!({ "ok": true, "message": "Preview", "image": { "mimeType": "image/jpeg", "base64": "AAAA" } }),
            _ => json!({ "ok": false, "message": format!("Unknown command \"{name}\".") }),
        }
    }
}

fn bridge_file(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nabu-cli-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("bridge.json")
}

fn start(name: &str) -> Client {
    let path = bridge_file(name);
    serve(Arc::new(Fake), &path).unwrap();
    Client::connect(&path).unwrap()
}

#[test]
fn calls_reach_the_editor_and_come_back() {
    let client = start("calls");
    let commands = client.commands().unwrap();
    assert_eq!(commands.as_array().unwrap().len(), 2);

    let result = client.call("adjust", json!({ "amount": 20 })).unwrap();
    assert_eq!(result["ok"], true);
    assert_eq!(result["data"]["echo"]["amount"], 20);

    let missing = client.call("nope", json!({})).unwrap();
    assert_eq!(missing["ok"], false);
}

#[test]
fn requests_without_the_token_are_refused() {
    let client = start("token");
    let mut stream = TcpStream::connect(("127.0.0.1", client.info.port)).unwrap();
    let body = r#"{"name":"adjust","arguments":{"amount":1}}"#;
    write!(
        stream,
        "POST /call HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer wrong\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 401"), "{response}");
}

#[test]
fn the_bridge_file_is_private() {
    let path = bridge_file("private");
    let info = serve(Arc::new(Fake), &path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains(&info.token));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "bridge file readable by others: {mode:o}");
    }
}

fn exchange(server: &mut Server<impl nabu_cli::mcp::Connector>, lines: &[Value]) -> Vec<Value> {
    let input: String = lines.iter().map(|l| format!("{l}\n")).collect();
    let mut output = Vec::new();
    server.run(input.as_bytes(), &mut output).unwrap();
    String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn mcp_session_lists_and_calls_tools() {
    let client = start("mcp");
    let mut server = Server::new(move || Ok(client.clone()));
    let answers = exchange(
        &mut server,
        &[
            json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": { "name": "test", "version": "1" } } }),
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
            json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": "adjust", "arguments": { "amount": 20 } } }),
            json!({ "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": { "name": "get_preview", "arguments": {} } }),
            json!({ "jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": { "name": "nope" } }),
            json!({ "jsonrpc": "2.0", "id": 6, "method": "resources/list" }),
        ],
    );

    // The notification gets no answer; everything else does, in order.
    assert_eq!(answers.len(), 6);
    assert_eq!(answers[0]["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(answers[0]["result"]["serverInfo"]["name"], "nabu");

    let tools = answers[1]["result"]["tools"].as_array().unwrap();
    assert_eq!(tools[0]["name"], "adjust");
    assert_eq!(tools[0]["inputSchema"]["required"][0], "amount");
    assert_eq!(tools[1]["annotations"]["readOnlyHint"], true);

    assert_eq!(answers[2]["result"]["isError"], false);
    let text = answers[2]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("Exposure: +20"), "{text}");
    assert!(text.contains("\"amount\": 20"), "{text}");

    assert_eq!(answers[3]["result"]["content"][1]["type"], "image");
    assert_eq!(answers[3]["result"]["content"][1]["data"], "AAAA");

    assert_eq!(answers[4]["result"]["isError"], true);
    assert_eq!(answers[5]["error"]["code"], -32601);
}

#[test]
fn mcp_reports_an_editor_that_cannot_start() {
    let mut server = Server::new(|| Err(std::io::Error::other("not installed")));
    let answers = exchange(
        &mut server,
        &[
            json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} }),
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
            json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": "adjust" } }),
        ],
    );
    // Initialising never needs the editor, so the client can show the error.
    assert_eq!(answers[0]["result"]["protocolVersion"], "2025-06-18");
    assert!(answers[1]["error"]["message"].as_str().unwrap().contains("not installed"));
    assert_eq!(answers[2]["result"]["isError"], true);
}

#[test]
fn arguments_parse_as_json_or_text() {
    let args = parse_arguments(&[
        "amount=15".into(),
        "property=exposure".into(),
        "values={\"contrast\":10}".into(),
        "path=photos/a.jpg".into(),
    ])
    .unwrap();
    assert_eq!(args["amount"], 15);
    assert_eq!(args["property"], "exposure");
    assert_eq!(args["values"]["contrast"], 10);
    let path = args["path"].as_str().unwrap();
    assert!(std::path::Path::new(path).is_absolute() && path.ends_with("a.jpg"), "{path}");
    assert!(parse_arguments(&["nonsense".into()]).is_err());
}

#[test]
fn batch_options_become_one_call() {
    let parsed = parse_batch(&[
        "--look".into(),
        "portra".into(),
        "--steps".into(),
        r#"[{"command":"resize","args":{"width":2048}}]"#.into(),
        "--out".into(),
        "/tmp/out".into(),
        "--quality".into(),
        "0.8".into(),
        "a.jpg".into(),
        "/abs/b.jpg".into(),
    ])
    .unwrap();
    let args = parsed.arguments;
    assert_eq!(args["steps"][0]["command"], "apply_look");
    assert_eq!(args["steps"][1]["args"]["width"], 2048);
    assert_eq!(args["output_dir"], "/tmp/out");
    assert_eq!(args["quality"], 0.8);
    assert_eq!(args["files"][1], "/abs/b.jpg");
    assert!(std::path::Path::new(args["files"][0].as_str().unwrap()).is_absolute());

    assert!(parse_batch(&["a.jpg".into()]).is_err(), "missing --out");
    assert!(parse_batch(&["--out".into(), "/tmp".into()]).is_err(), "no photos");
}
