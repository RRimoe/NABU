// src-tauri/src/ai/vision/mod.rs
//! The vision engine: models cooperate on one shared picture of the image.
//!
//! Instead of every feature calling its favourite network directly, the engine
//! defines *capabilities* — stable contracts such as "a detector receives an
//! image and returns classified boxes" — and a catalogue of models that fulfil
//! them, each described by a declarative manifest (see `manifest`). The
//! outputs meet in the scene model (see `scene`), the representation the
//! language router reasons over.
//!
//! The engine is lazy by design: the fast pass (detection plus pixel
//! statistics) runs once per image, heavier specialists only when a request
//! actually needs them, and nothing at all if no model for a capability is
//! installed — the editor keeps working, it just cannot see.

pub mod catalog;
pub mod depth;
pub mod detect;
pub mod inpaint;
pub mod manifest;
pub mod preprocess;
pub mod region;
pub mod scene;
pub mod shadow;
pub mod synth;
