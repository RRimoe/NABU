//! End-to-end check of the language router.
//!
//! Requires the `llm` feature and the router model on disk, so it skips itself
//! when either is missing:
//!     cargo test --features llm --test router -- --ignored --nocapture
//!
//! What it asserts is deliberately loose. A 1.5B model will not phrase things
//! identically every run, and pinning exact numbers would produce a test that
//! fails for no real reason. It checks the things that must hold: the right
//! tool, the right property, and the right direction of change.

#![cfg(feature = "llm")]

use std::path::PathBuf;

#[path = "../src/ai/error.rs"]
mod error;
#[path = "../src/ai/router.rs"]
mod router;
#[path = "../src/ai/tools.rs"]
mod tools;

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.editor.nabu/models/qwen2.5-1.5b-instruct")
}

const MODEL_FILE: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";
const NO_SELECTION: &str = "Nothing is selected; the request applies to the whole photo.";

async fn ask(phrase: &str, context: &str) -> Option<tools::ToolCall> {
    let dir = model_dir();
    if !dir.join(MODEL_FILE).is_file() {
        eprintln!("router model not downloaded, skipping");
        return None;
    }

    // Timed per phrase: the first call pays for loading the weights and for
    // processing the system prompt, later ones should reuse both. Without the
    // split it is impossible to tell "slow every time" from "slow once".
    let started = std::time::Instant::now();
    match router::interpret(&dir.to_string_lossy(), MODEL_FILE, phrase, context).await {
        Ok(call) => {
            println!(
                "  {:>6.1}s  {phrase:<34} -> {} {}",
                started.elapsed().as_secs_f32(),
                call.name,
                call.arguments
            );
            Some(call)
        }
        Err(e) => panic!("router failed on {phrase:?}: {e}"),
    }
}

#[tokio::test]
#[ignore = "requires the llm feature and the router model on disk"]
async fn understands_the_same_request_in_several_languages() {
    // The point of the model over keyword matching: no per-language rules.
    for phrase in [
        "remove the background",
        "togli lo sfondo",
        "quita el fondo",
        "enlève l'arrière-plan",
    ] {
        let Some(call) = ask(phrase, NO_SELECTION).await else { return };
        assert_eq!(call.name, "remove_background", "wrong tool for {phrase:?}");
    }
}

#[tokio::test]
#[ignore = "requires the llm feature and the router model on disk"]
async fn maps_a_vague_request_onto_a_property_and_a_direction() {
    // Several properties are a defensible reading of the same vague phrase:
    // "darker" can mean lowering exposure or pulling the shadows down, and both
    // do what the person asked. Only the direction is genuinely right or wrong,
    // so that is what is asserted strictly.
    let cases: [(&str, &[&str], bool); 3] = [
        // phrase, acceptable properties, should the amount be positive
        ("make it brighter", &["exposure", "highlights", "whites"], true),
        ("rendila più scura", &["exposure", "shadows", "blacks"], false),
        ("less saturated please", &["saturation", "vibrance"], false),
    ];

    for (phrase, acceptable, positive) in cases {
        let Some(call) = ask(phrase, NO_SELECTION).await else { return };
        assert_eq!(call.name, "adjust", "wrong tool for {phrase:?}");

        let got = call.arguments["property"].as_str().unwrap_or_default();
        assert!(
            acceptable.contains(&got),
            "{phrase:?} gave property {got:?}, expected one of {acceptable:?}"
        );

        let amount = call.arguments["amount"].as_f64().unwrap_or(0.0);
        assert!(amount != 0.0, "no change requested for {phrase:?}");
        assert_eq!(
            amount > 0.0,
            positive,
            "wrong direction for {phrase:?}: amount={amount}"
        );
    }
}

#[tokio::test]
#[ignore = "requires the llm feature and the router model on disk"]
async fn a_selection_changes_which_tool_is_right() {
    let with_selection = "The user has selected a region of the photo.";
    let Some(call) = ask("cut this out", with_selection).await else { return };
    assert_eq!(call.name, "cut_out_selection");
}

#[tokio::test]
#[ignore = "requires the llm feature and the router model on disk"]
async fn resolves_an_object_reference_against_the_scene() {
    // What the vision engine is for: the model never sees the image, only the
    // scene list, and still has to land on the right id from the words alone.
    let context = "Nothing is selected; the request applies to the whole photo.\n\n\
SCENE — objects detected in the photo:\n\
#1 person — center, dominant\n\
#2 car — right, secondary, color: red\n\
#3 dog — left, minor, color: brown";

    let cases = [
        ("select the red car", 2.0),
        ("seleziona l'auto rossa", 2.0),
        ("select the dog on the left", 3.0),
    ];

    for (phrase, expected_id) in cases {
        let Some(call) = ask(phrase, context).await else { return };
        assert_eq!(call.name, "select_object", "wrong tool for {phrase:?}");
        assert_eq!(
            call.arguments["id"].as_f64(),
            Some(expected_id),
            "wrong object for {phrase:?}"
        );
    }
}

/// Known limitation, kept as a record rather than a passing assertion.
///
/// Qwen2.5 1.5B handles direct requests well, but an indirect phrasing in a
/// language other than English defeats it: "i colori sono troppo freddi" (the
/// colours are too cold) is a white-balance complaint, and the model reaches
/// for saturation instead — even with every property glossed in the schema.
///
/// It is ignored because it documents a gap, not a regression. Remove the
/// attribute when trying a larger model or few-shot examples in the prompt: if
/// it passes, the gap has closed.
#[tokio::test]
#[ignore = "known limitation of the 1.5B router"]
async fn indirect_phrasing_in_another_language_is_not_reliable_yet() {
    let Some(call) = ask("i colori sono troppo freddi", NO_SELECTION).await else { return };
    assert_eq!(call.name, "adjust");
    assert_eq!(
        call.arguments["property"].as_str(),
        Some("temperature"),
        "the model still misreads a white-balance complaint"
    );
}
