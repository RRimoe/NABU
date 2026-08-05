//! Prints the input and output signature of the SlimSAM models.
//!
//! Guessing tensor names or shapes for a two-stage model is how you end up
//! debugging a black mask for an afternoon, so this reads them from the files:
//!     cargo test --test inspect_sam -- --ignored --nocapture

use std::path::PathBuf;

use ort::session::Session;

fn models_dir() -> PathBuf {
    // Same location the application uses on Windows.
    let base = std::env::var("APPDATA").expect("APPDATA");
    PathBuf::from(base).join("com.editor.nabu/models/slimsam-77")
}

fn describe(label: &str, file: &str) {
    let path = models_dir().join(file);
    if !path.is_file() {
        println!("== {label}: not downloaded ({})", path.display());
        return;
    }

    let session = Session::builder()
        .expect("builder")
        .commit_from_file(&path)
        .expect("load");

    println!("\n== {label} ==");
    println!("  inputs:");
    for input in session.inputs() {
        println!("    {:<32} {:?}", input.name(), input.dtype());
    }
    println!("  outputs:");
    for output in session.outputs() {
        println!("    {:<32} {:?}", output.name(), output.dtype());
    }
}

#[test]
#[ignore = "requires the SlimSAM model on disk"]
fn print_slimsam_signature() {
    describe("vision encoder", "vision_encoder.onnx");
    describe("prompt encoder + mask decoder", "prompt_encoder_mask_decoder.onnx");
}
