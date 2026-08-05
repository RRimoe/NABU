// src-tauri/src/ai/engine.rs
//! Creation and reuse of ONNX Runtime sessions.
//!
//! Loading a model is expensive (reading the file, optimising the graph,
//! allocating on the GPU), so sessions are cached per model id and reused for
//! every subsequent run.

use std::collections::HashMap;
use std::path::PathBuf;
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
}

impl SessionCache {
    /// Drops a cached session: used when a model is deleted from disk, since the
    /// session would otherwise keep pointing at a file that no longer exists.
    pub fn invalidate(&self, model_id: &str) {
        self.sessions.lock().unwrap().remove(model_id);
    }

    /// Returns the session for a model, creating it on first request.
    pub fn get_or_load(
        &self,
        app: &AppHandle,
        model_id: &str,
        file_name: &str,
    ) -> NabuResult<Arc<Mutex<Session>>> {
        let key = format!("{model_id}/{file_name}");

        if let Some(existing) = self.sessions.lock().unwrap().get(&key) {
            return Ok(existing.clone());
        }

        let path = resolve_model_file(app, model_id, file_name)?;
        let session = build_session(&path)?;
        let session = Arc::new(Mutex::new(session));

        self.sessions
            .lock()
            .unwrap()
            .insert(key, session.clone());

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
/// guarantees the work still runs.
fn build_session(path: &PathBuf) -> NabuResult<Session> {
    let mut builder = Session::builder()
        .map_err(error::inference_failed)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(error::inference_failed)?;

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
