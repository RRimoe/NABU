// src-tauri/src/ai/mod.rs
//! Local AI models: catalogue, downloading, on-disk state and inference.

pub mod download;
pub mod engine;
pub mod error;
pub mod imaging;
pub mod registry;
#[cfg(feature = "llm")]
pub mod router;
pub mod tasks;
pub mod tools;
pub mod vision;
