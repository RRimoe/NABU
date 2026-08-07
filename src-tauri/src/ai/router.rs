// src-tauri/src/ai/router.rs
//! Turns a sentence into a call to one of Nabu's tools.
//!
//! The model here never edits an image and never writes anything the user
//! reads. Its only job is to choose a tool and fill in its arguments, which is
//! why a small one is enough: it is pattern matching plus reliable structured
//! output, not knowledge or reasoning.
//!
//! Compiled only with the `llm` feature. Without it the interface falls back to
//! keyword matching, which understands far less but costs nothing.

use std::sync::Arc;

use mistralrs::{
    Function, GgufModelBuilder, RequestBuilder, TextMessageRole, Tool, ToolChoice, ToolType,
};
use tokio::sync::{Mutex, OnceCell};

use super::error::{self, NabuResult};
use super::tools::{self, ToolCall};

/// A tool call is a couple of dozen tokens. Without a ceiling the model can
/// drift into a paragraph of explanation, which on CPU costs minutes.
const MAX_TOKENS: usize = 128;

/// Written in English on purpose: these models follow instructions far more
/// reliably in English, while the user's sentence can be in any language.
const SYSTEM_PROMPT: &str = "\
You are the command router of an image editor. The user describes what they \
want done to a photo. Choose exactly one tool and call it.

Rules:
- Always call a tool. Never reply with prose.
- The user may write in any language. Understand it, but keep tool names and \
argument values exactly as defined.
- If the request is about brightness, darkness, colour, warmth, sharpness or \
any similar property, use `adjust` with the closest property.
- When the request points at a thing in the photo (the person, the red car, \
the dog on the left) and the context contains a SCENE list, call \
`select_object` with the id of the entry that best matches the words.
- If nothing fits, call `adjust` with property `exposure` and amount 0, which \
means 'no change'.";

/// A loaded router, kept alive between requests.
///
/// Loading a gigabyte of weights takes seconds; doing it per keystroke would
/// make the command line unusable.
static ROUTER: OnceCell<Arc<Mutex<mistralrs::Model>>> = OnceCell::const_new();

/// Builds the tool list handed to the model from the single source of truth.
fn tool_definitions() -> Vec<Tool> {
    tools::TOOLS
        .iter()
        .map(|spec| {
            let schema = tools::parameters_schema(spec);
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: spec.name.to_string(),
                    description: Some(spec.description.to_string()),
                    parameters: serde_json::from_value(schema).ok(),
                },
            }
        })
        .collect()
}

/// Loads the router model from a local GGUF file, once.
async fn router(model_dir: &str, file_name: &str) -> NabuResult<Arc<Mutex<mistralrs::Model>>> {
    ROUTER
        .get_or_try_init(|| async {
            let model = GgufModelBuilder::new(model_dir, vec![file_name])
                // Loading a gigabyte of weights is silent otherwise, which makes
                // a slow start indistinguishable from a hang.
                .with_logging()
                .build()
                .await
                .map_err(error::inference_failed)?;
            Ok(Arc::new(Mutex::new(model)))
        })
        .await
        .cloned()
}

/// Interprets one sentence and returns the tool the interface should run.
///
/// `context` describes the state the request applies to — whether something is
/// selected, for instance — so the model can prefer the tool that makes sense
/// rather than guessing.
pub async fn interpret(
    model_dir: &str,
    file_name: &str,
    phrase: &str,
    context: &str,
) -> NabuResult<ToolCall> {
    let model = router(model_dir, file_name).await?;
    let model = model.lock().await;

    let request = RequestBuilder::new()
        .add_message(TextMessageRole::System, SYSTEM_PROMPT)
        .add_message(TextMessageRole::User, format!("{context}\n\n{phrase}"))
        .set_tools(tool_definitions())
        .set_tool_choice(ToolChoice::Auto)
        .set_sampler_max_len(MAX_TOKENS)
        // Routing is a classification, not a creative task: the same sentence
        // must always reach the same tool.
        .set_deterministic_sampler();

    let response = model
        .send_chat_request(request)
        .await
        .map_err(error::inference_failed)?;

    let message = response
        .choices
        .first()
        .map(|choice| &choice.message)
        .ok_or_else(|| error::inference_failed("router returned no choice"))?;

    let parsed = match message.tool_calls.as_ref().and_then(|calls| calls.first()) {
        Some(call) => ToolCall {
            name: call.function.name.clone(),
            arguments: serde_json::from_str(&call.function.arguments)
                .unwrap_or_else(|_| serde_json::json!({})),
        },
        // The engine could not read a call, but the model may still have
        // written one: see `recover_tool_call` for why that happens.
        None => {
            let said = message.content.as_deref().unwrap_or_default();
            recover_tool_call(said).ok_or_else(|| {
                error::router_no_tool().detail(format!("model replied: {said}"))
            })?
        }
    };

    // A small model occasionally invents a tool, or names a property with the
    // everyday word rather than ours. Near-misses are rewritten; what cannot be
    // salvaged is reported, so the user is told it was not understood instead
    // of watching the application do nothing.
    let mut parsed = parsed;
    tools::normalise(&mut parsed).map_err(error::router_invalid)?;

    Ok(parsed)
}

/// Reads a tool call the engine could not parse out of the raw reply.
///
/// The chat template embedded in some GGUF builds — including the official
/// Qwen2.5 one — shows the model an example with doubled braces:
///
/// ```text
/// <tool_call>
/// {{"name": <function-name>, "arguments": <args-json-object>}}
/// </tool_call>
/// ```
///
/// The model dutifully copies that shape, and the result is not valid JSON, so
/// the engine reports no tool call at all. Rather than ship a patched template
/// per model, the malformed shape is repaired here: the model picked the right
/// tool, only the punctuation was wrong.
fn recover_tool_call(content: &str) -> Option<ToolCall> {
    // The tags are the usual wrapper, but not every model emits them: some
    // answer with the bare object, so their absence is not a dead end.
    let block = match content.split_once("<tool_call>") {
        Some((_, rest)) => rest.split_once("</tool_call>").map_or(rest, |(inner, _)| inner),
        None => content,
    }
    .trim();

    let value = first_json_object(block)?;

    let name = value.get("name")?.as_str()?.to_string();
    let mut arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    // Some replies wrap the real arguments in the schema's own `properties`
    // key, copying the shape of the tool definition instead of filling it in.
    if let Some(object) = arguments.as_object() {
        if object.len() == 1 {
            if let Some(inner) = object.get("properties") {
                if inner.is_object() {
                    arguments = inner.clone();
                }
            }
        }
    }

    Some(ToolCall { name, arguments })
}

/// Extracts the first parsable JSON object from a fragment.
///
/// Counting braces to undo the doubling is not enough: the replies are not
/// symmetric — one observed answer opened with two braces and closed with four.
/// Instead the text is scanned for a balanced object, starting at the first
/// brace and then at the second, so a doubled opening is simply stepped over
/// and any trailing rubbish is ignored.
fn first_json_object(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();

    for (start, _) in text.char_indices().filter(|(_, c)| *c == '{').take(3) {
        let Some(end) = balanced_end(bytes, start) else { continue };
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text[start..=end]) {
            if value.get("name").is_some() {
                return Some(value);
            }
        }
    }
    None
}

/// Index of the brace closing the object that opens at `start`.
///
/// Braces inside string literals do not count, or a value like `"{"` would
/// throw the depth off.
fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, byte) in bytes.iter().enumerate().skip(start) {
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_the_doubled_brace_shape_qwen_emits() {
        let reply = "<tool_call>
{{\"name\": \"remove_background\", \"arguments\": {}}}
</tool_call>";
        let call = recover_tool_call(reply).expect("recovered");
        assert_eq!(call.name, "remove_background");
    }

    #[test]
    fn recovers_arguments_and_unwraps_a_schema_artefact() {
        let reply = "<tool_call>
{{\"name\": \"adjust\", \"arguments\": {\"properties\": {\"property\": \"exposure\", \"amount\": 25}}}}
</tool_call>";
        let call = recover_tool_call(reply).expect("recovered");
        assert_eq!(call.name, "adjust");
        assert_eq!(call.arguments["property"], "exposure");
        assert_eq!(call.arguments["amount"], 25);
    }

    #[test]
    fn still_reads_a_well_formed_call() {
        let reply = "<tool_call>{\"name\": \"upscale\", \"arguments\": {}}</tool_call>";
        let call = recover_tool_call(reply).expect("recovered");
        assert_eq!(call.name, "upscale");
    }

    #[test]
    fn recovers_an_asymmetric_reply() {
        // Observed from the model: two braces open, four close.
        let reply = "<tool_call>
{{\"name\": \"adjust\", \"arguments\": {\"property\": \"exposure\", \"amount\": 25}}}}
</tool_call>";
        let call = recover_tool_call(reply).expect("recovered");
        assert_eq!(call.name, "adjust");
        assert_eq!(call.arguments["property"], "exposure");
        assert_eq!(call.arguments["amount"], 25);
    }

    #[test]
    fn ignores_braces_inside_strings() {
        let reply = r#"{"name": "adjust", "arguments": {"property": "a}b{c", "amount": 1}}"#;
        let call = recover_tool_call(reply).expect("recovered");
        assert_eq!(call.arguments["property"], "a}b{c");
    }

    #[test]
    fn returns_nothing_for_prose() {
        assert!(recover_tool_call("I am not sure what you mean.").is_none());
    }
}
