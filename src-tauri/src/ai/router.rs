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
use tokio::sync::Mutex;

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
any similar property, use `adjust` with the closest property. Complaints are \
requests too: 'too dark' means brighten, 'colours too cold' means warmer.
- When the request points at a thing in the photo (the person, the red car, \
the dog on the left) and the context contains a SCENE list, use the id of the \
entry that best matches the words: `select_object` to select it, \
`remove_object` to erase it from the photo.
- `select_object` and `remove_object` take ids from the SCENE list and from \
nowhere else. If the context has no SCENE list, or no entry matches the thing \
named, never invent an id: call `report_unsupported` instead.
- Replacing content or generating imagery is not possible: report it honestly \
with `report_unsupported` instead of forcing another tool.
- If nothing fits, call `adjust` with property `exposure` and amount 0, which \
means 'no change'.

Examples of the call for a given request:
- 'make it a bit brighter': {\"name\": \"adjust\", \"arguments\": {\"property\": \"exposure\", \"amount\": 15}}
- 'i colori sono troppo freddi': {\"name\": \"adjust\", \"arguments\": {\"property\": \"temperature\", \"amount\": 25}}
- 'blur the photo slightly': {\"name\": \"adjust\", \"arguments\": {\"property\": \"blur\", \"amount\": 10}}
- 'seleziona l'auto rossa', SCENE has '#2 car - color: red': {\"name\": \"select_object\", \"arguments\": {\"id\": 2}}
- 'remove the man on the right', SCENE has '#3 person - right': {\"name\": \"remove_object\", \"arguments\": {\"id\": 3}}
- 'remove this' while a selection exists: {\"name\": \"remove_selection\", \"arguments\": {}}
- 'elimina il cartello giallo', SCENE has no sign and nothing is selected: {\"name\": \"report_unsupported\", \"arguments\": {\"feature\": \"remove-object\"}}
- 'change the sky': {\"name\": \"report_unsupported\", \"arguments\": {\"feature\": \"replace-content\"}}";

/// The loaded router, kept alive between requests and keyed by its file.
///
/// Loading half a gigabyte of weights takes seconds; doing it per keystroke
/// would make the command line unusable. The key matters too: the preferred
/// model can change while the application runs — the user downloads the
/// smaller one, say — and a plain once-cell would keep answering with
/// whichever loaded first.
static ROUTER: Mutex<Option<(String, Arc<Mutex<mistralrs::Model>>)>> = Mutex::const_new(None);

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

/// Loads the router model from a local GGUF file, reusing it across requests.
async fn router(model_dir: &str, file_name: &str) -> NabuResult<Arc<Mutex<mistralrs::Model>>> {
    let key = format!("{model_dir}/{file_name}");

    let mut slot = ROUTER.lock().await;
    if let Some((loaded, model)) = slot.as_ref() {
        if *loaded == key {
            return Ok(model.clone());
        }
    }

    let model = GgufModelBuilder::new(model_dir, vec![file_name])
        // Loading half a gigabyte of weights is silent otherwise, which makes
        // a slow start indistinguishable from a hang.
        .with_logging()
        .build()
        .await
        .map_err(error::inference_failed)?;

    let model = Arc::new(Mutex::new(model));
    *slot = Some((key, model.clone()));
    Ok(model)
}

/// Loads the router and primes its caches without the user asking anything.
///
/// Two costs are paid here instead of on the first request: reading the
/// weights, and — much larger on CPU — processing the system prompt and tool
/// schemas, which measured in the minutes cold. Those tokens are identical
/// for every request, so one throwaway request through the prefix cache turns
/// the first real question from minutes into seconds. Called in the
/// background at startup; a failure only means the first request pays.
pub async fn preload(model_dir: &str, file_name: &str) -> NabuResult<()> {
    let model = router(model_dir, file_name).await?;
    let model = model.lock().await;

    let request = RequestBuilder::new()
        .add_message(TextMessageRole::System, SYSTEM_PROMPT)
        .add_message(TextMessageRole::User, "warm-up")
        .set_tools(tool_definitions())
        .set_tool_choice(ToolChoice::Auto)
        .set_sampler_max_len(1)
        .set_deterministic_sampler();

    let _ = model.send_chat_request(request).await;
    Ok(())
}

/// An observation plus a recipe is a short answer; a ceiling keeps a small
/// model from drifting into an essay, which on CPU costs minutes.
const ADVICE_MAX_TOKENS: usize = 96;

/// Written in English for the same reason as the router prompt: these models
/// follow instructions far more reliably in it. The observation comes back in
/// the language the caller asks for.
const ADVICE_PROMPT: &str = "\
You are the assistant of a photo editor. You are given measurements of a photo \
and a list of what a detector found in it. You never see the photo itself.

Reply with one line of JSON and nothing else:
{\"observation\": \"...\", \"recipe\": \"...\", \"target\": <id or null>}

`observation` is one short sentence, at most 18 words, about THIS photo, in the \
language named by LANGUAGE. Say something that follows from the numbers and the \
objects you were given. Never invent an object that is not listed. Never \
mention the light, the weather or the time of day: you cannot see them.

`recipe` must be exactly one of: highlight-subject, separate-subject, \
lift-shadows, recover-highlights, add-punch, warm-up, cool-down, miniature, none

`target` is the id of the object the recipe applies to, or null. Only use an id \
from the OBJECTS list.

Examples:
{\"observation\": \"The bench is the subject, but the background competes with it.\", \"recipe\": \"highlight-subject\", \"target\": 1}
{\"observation\": \"The picture is dark and its shadows have lost detail.\", \"recipe\": \"lift-shadows\", \"target\": null}";

/// Reads the scene and proposes one thing worth doing, in the user's language.
///
/// This is the *second* tier of the suggestions panel: the first is measured
/// arithmetic that appears instantly, and this arrives seconds later with a
/// sentence and a recipe. It is never on the critical path, so its 10-14s on
/// CPU costs nothing the user waits for — which is the whole reason typed
/// requests were removed and this took their place.
///
/// The model chooses from a closed list and refers to objects by id. It never
/// writes an operation, and the caller validates both the recipe name and the
/// id against what actually exists — the same rule that keeps the router
/// honest, for the same reason.
pub async fn advise(
    model_dir: &str,
    file_name: &str,
    scene: &str,
    language: &str,
) -> NabuResult<String> {
    let model = router(model_dir, file_name).await?;
    let model = model.lock().await;

    let request = RequestBuilder::new()
        .add_message(TextMessageRole::System, ADVICE_PROMPT)
        .add_message(
            TextMessageRole::User,
            format!("LANGUAGE: {language}\n\n{scene}"),
        )
        .set_sampler_max_len(ADVICE_MAX_TOKENS)
        // The same photo must always produce the same advice: an assistant that
        // changes its mind between two openings of the same file is not one.
        .set_deterministic_sampler();

    let response = model
        .send_chat_request(request)
        .await
        .map_err(error::inference_failed)?;

    let text = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .unwrap_or_default();

    Ok(extract_json_object(&text))
}

/// Pulls the first `{...}` out of whatever the model wrapped it in.
///
/// Small models like to introduce their JSON, or fence it in backticks. The
/// braces are the part that matters and the rest is politeness.
fn extract_json_object(text: &str) -> String {
    let Some(start) = text.find('{') else {
        return String::new();
    };

    let mut depth = 0usize;
    for (offset, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return text[start..start + offset + ch.len_utf8()].to_string();
                }
            }
            _ => {}
        }
    }

    String::new()
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

    let Some(value) = first_json_object(block) else {
        // No JSON at all: small models sometimes answer in function-call
        // notation instead, `adjust(property=exposure, amount=25)`.
        return recover_pseudo_call(block);
    };

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

/// Reads a call written in function notation: `adjust(property=exposure, amount=25)`.
///
/// Small models drift into this shape no matter what the template shows,
/// because it dominates their training data. The intent is fully recoverable:
/// name before the parenthesis, `key=value` or `key: value` pairs inside.
/// Whatever comes out still passes through `tools::normalise`, so an invented
/// tool or property is rejected exactly as it would be for a JSON call.
fn recover_pseudo_call(content: &str) -> Option<ToolCall> {
    let open = content.find('(')?;
    let close = content[open..].find(')')? + open;

    // The name is the identifier run ending right before the parenthesis.
    let name: String = content[..open]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if name.is_empty() {
        return None;
    }

    let mut arguments = serde_json::Map::new();
    for pair in content[open + 1..close].split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (key, value) = pair.split_once(['=', ':'])?;
        let key = key.trim().trim_matches(['"', '\'']);
        let value = value.trim().trim_matches(['"', '\'']);

        let value = match value.parse::<f64>() {
            Ok(number) => serde_json::json!(number),
            Err(_) => serde_json::json!(value),
        };
        arguments.insert(key.to_string(), value);
    }

    Some(ToolCall {
        name,
        arguments: serde_json::Value::Object(arguments),
    })
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
    fn advice_json_survives_the_prose_small_models_wrap_it_in() {
        let wrapped = "Sure! Here is my answer:\n```json\n{\"observation\": \"a\", \"recipe\": \"warm-up\", \"target\": null}\n```\nHope that helps.";
        assert_eq!(
            extract_json_object(wrapped),
            "{\"observation\": \"a\", \"recipe\": \"warm-up\", \"target\": null}"
        );
    }

    #[test]
    fn advice_json_keeps_nested_braces_together() {
        let nested = "{\"observation\": \"x\", \"meta\": {\"a\": 1}, \"recipe\": \"add-punch\"}";
        assert_eq!(extract_json_object(nested), nested);
    }

    #[test]
    fn advice_with_no_object_comes_back_empty_rather_than_half_parsed() {
        assert_eq!(extract_json_object("I could not decide."), "");
        // An unterminated object is not a usable answer either.
        assert_eq!(extract_json_object("{\"observation\": \"cut off"), "");
    }

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

    #[test]
    fn recovers_function_notation_with_equals() {
        let call = recover_tool_call("adjust(property=exposure, amount=50)").expect("recovered");
        assert_eq!(call.name, "adjust");
        assert_eq!(call.arguments["property"], "exposure");
        assert_eq!(call.arguments["amount"], 50.0);
    }

    #[test]
    fn recovers_function_notation_with_colons_and_quotes() {
        let call =
            recover_tool_call("select_object(id: 2)").expect("recovered");
        assert_eq!(call.name, "select_object");
        assert_eq!(call.arguments["id"], 2.0);

        let call = recover_tool_call("adjust(property: \"temperature\", amount: 25)")
            .expect("recovered");
        assert_eq!(call.arguments["property"], "temperature");
    }

    #[test]
    fn recovers_function_notation_without_arguments() {
        let call = recover_tool_call("Sure! remove_background()").expect("recovered");
        assert_eq!(call.name, "remove_background");
        assert!(call.arguments.as_object().unwrap().is_empty());
    }
}
