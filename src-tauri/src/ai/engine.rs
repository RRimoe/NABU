// src-tauri/src/ai/engine.rs
//! Creation and reuse of ONNX Runtime sessions.
//!
//! Loading a model is expensive (reading the file, optimising the graph,
//! allocating on the GPU), so sessions are cached per model id and reused for
//! every subsequent run.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::ep::ExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde::Serialize;
use tauri::AppHandle;

use super::download::{self, InstallState};
use super::error::{self, NabuResult};
use super::registry;

/// Accelerator actually in use, reported in the interface.
// Each variant is only constructed on the platform that supports it; elsewhere
// the compiler rightly sees it as unreachable.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Accelerator {
    DirectMl,
    CoreMl,
    Cpu,
}

impl Accelerator {
    pub fn label(self) -> &'static str {
        match self {
            Accelerator::DirectMl => "GPU (DirectML)",
            Accelerator::CoreMl => "GPU (CoreML)",
            Accelerator::Cpu => "CPU",
        }
    }
}

/// Detects the accelerator available on this machine.
///
/// `is_available()` asks ONNX Runtime, so the answer reflects the actual system
/// rather than an assumption based on the operating system.
pub fn detect_accelerator() -> Accelerator {
    #[cfg(windows)]
    {
        if ort::ep::DirectML::default().is_available().unwrap_or(false) {
            return Accelerator::DirectMl;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if ort::ep::CoreML::default().is_available().unwrap_or(false) {
            return Accelerator::CoreMl;
        }
    }
    Accelerator::Cpu
}

/// Session cache, shared as Tauri-managed state.
#[derive(Default)]
pub struct SessionCache {
    sessions: Mutex<HashMap<String, Arc<Mutex<Session>>>>,
    embedding: Mutex<Option<(String, Arc<super::tasks::SamEmbedding>)>>,
}

impl SessionCache {
    /// The segmenter's view of a picture, computed once and kept.
    ///
    /// Looking at a nine-megapixel photograph takes the encoder seconds;
    /// answering a prompt against what it saw takes fifty milliseconds. Holding
    /// the first is the whole difference between a selection you correct by
    /// pointing at it and a selection you wait for. Exactly one is kept — a
    /// person corrects the picture in front of them, and each embedding is
    /// megabytes.
    pub fn embedding_for(
        &self,
        key: &str,
        make: impl FnOnce() -> NabuResult<super::tasks::SamEmbedding>,
    ) -> NabuResult<Arc<super::tasks::SamEmbedding>> {
        if let Some((held, embedding)) = self.embedding.lock().unwrap().as_ref() {
            if held == key {
                return Ok(embedding.clone());
            }
        }
        let embedding = Arc::new(make()?);
        *self.embedding.lock().unwrap() = Some((key.to_string(), embedding.clone()));
        Ok(embedding)
    }

    /// Forgets the held embedding, for when the picture it described is gone.
    pub fn forget_embedding(&self) {
        *self.embedding.lock().unwrap() = None;
    }
}

impl SessionCache {
    /// Drops the cached sessions of a model: used when it is deleted from
    /// disk, since a session would otherwise keep pointing at a file that no
    /// longer exists. Matches by prefix because the cache is keyed per file.
    pub fn invalidate(&self, model_id: &str) {
        let prefix = format!("{model_id}/");
        self.sessions
            .lock()
            .unwrap()
            .retain(|key, _| !key.starts_with(&prefix));
    }

    /// Returns the session for a model, creating it on first request.
    pub fn get_or_load(
        &self,
        app: &AppHandle,
        model_id: &str,
        file_name: &str,
    ) -> NabuResult<Arc<Mutex<Session>>> {
        let key = format!("{model_id}/{file_name}");
        let path = resolve_model_file(app, model_id, file_name)?;
        self.get_or_load_at(&key, &path, false)
    }

    /// Returns the session for a model file at an explicit path.
    ///
    /// This is the entry point for manifest-described models. `cpu_only`
    /// comes from the manifest: some networks contain operators the GPU
    /// provider accepts and then fails at run time, and the manifest is where
    /// that knowledge lives.
    pub fn get_or_load_at(
        &self,
        key: &str,
        path: &Path,
        cpu_only: bool,
    ) -> NabuResult<Arc<Mutex<Session>>> {
        if let Some(existing) = self.sessions.lock().unwrap().get(key) {
            return Ok(existing.clone());
        }

        let session = build_session(&path.to_path_buf(), cpu_only)?;
        let session = Arc::new(Mutex::new(session));

        self.sessions
            .lock()
            .unwrap()
            .insert(key.to_string(), session.clone());

        Ok(session)
    }
}

/// Loads the primary session of a model.
///
/// For multi-file models (such as SlimSAM, split into encoder and decoder) this
/// takes the first one; the others must be requested explicitly by whichever
/// pipeline needs them.
pub fn load_for(
    app: &AppHandle,
    cache: &SessionCache,
    model_id: &str,
) -> NabuResult<Arc<Mutex<Session>>> {
    let spec = registry::find(model_id).ok_or_else(|| error::unknown_model(model_id))?;
    let file = spec
        .files
        .first()
        .ok_or_else(|| error::missing_file("", spec.name))?;

    cache.get_or_load(app, model_id, file.file_name)
}

/// Loads one specific file of a model, for models split into several stages.
pub fn load_file(
    app: &AppHandle,
    cache: &SessionCache,
    model_id: &str,
    file_name: &str,
) -> NabuResult<Arc<Mutex<Session>>> {
    cache.get_or_load(app, model_id, file_name)
}

/// Checks that the model is installed and returns the path of one of its files.
pub fn resolve_model_file(
    app: &AppHandle,
    model_id: &str,
    file_name: &str,
) -> NabuResult<PathBuf> {
    let spec = registry::find(model_id).ok_or_else(|| error::unknown_model(model_id))?;

    if download::status_of(app, spec).state != InstallState::Ready {
        return Err(error::model_not_downloaded(spec.name));
    }

    let path = download::model_dir(app, model_id)?.join(file_name);
    if !path.is_file() {
        return Err(error::missing_file(file_name, spec.name));
    }
    Ok(path)
}

/// Builds the session, registering whichever accelerator is available.
///
/// Execution providers are listed in order of preference: ONNX Runtime uses the
/// first that initialises and falls back to the next, so CPU at the end
/// guarantees the work still runs. With `cpu_only` no accelerator is offered
/// at all — for models whose operators a GPU provider accepts and then fails
/// on at run time.
pub fn build_session(path: &PathBuf, cpu_only: bool) -> NabuResult<Session> {
    let mut builder = Session::builder()
        .map_err(error::inference_failed)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(error::inference_failed)?;

    if cpu_only {
        builder = builder
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(error::inference_failed)?;
        return builder.commit_from_file(path).map_err(error::inference_failed);
    }

    #[cfg(windows)]
    {
        builder = builder
            .with_execution_providers([
                ort::ep::DirectML::default().build(),
                ort::ep::CPU::default().build(),
            ])
            .map_err(error::inference_failed)?;
    }
    #[cfg(target_os = "macos")]
    {
        builder = builder
            .with_execution_providers([
                ort::ep::CoreML::default().build(),
                ort::ep::CPU::default().build(),
            ])
            .map_err(error::inference_failed)?;
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        builder = builder
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(error::inference_failed)?;
    }

    builder
        .commit_from_file(path)
        .map_err(error::inference_failed)
}
