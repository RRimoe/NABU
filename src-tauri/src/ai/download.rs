// src-tauri/src/ai/download.rs
//! Model downloading with progress, resume and integrity verification.
//!
//! Every file is written to `<name>.part` and only renamed once its SHA-256
//! matches the expected digest, so a truncated or corrupted file can never
//! appear in the models folder.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::error::{self, NabuError, NabuResult};
use super::registry::{self, ModelFile, ModelSpec};

/// Whether a model is present on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum InstallState {
    NotDownloaded,
    /// Some files present but not all: typically an interrupted download.
    Partial,
    Ready,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: String,
    pub name: String,
    pub task: registry::ModelTask,
    /// Translation key, resolved by the interface.
    pub description_key: String,
    pub license: String,
    pub commercial_use: bool,
    pub vram_required_mb: u32,
    pub total_bytes: u64,
    pub state: InstallState,
    /// Bytes already on disk, including the `.part` of an interrupted download.
    pub bytes_on_disk: u64,
    pub scale_factor: Option<u32>,
    pub dir: String,
}

/// Progress event consumed by the interface.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub model_id: String,
    pub file_name: String,
    /// Index of the current file and total count, for multi-file models.
    pub file_index: usize,
    pub file_count: usize,
    /// Bytes downloaded and expected, for the model as a whole.
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f64,
    pub bytes_per_second: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadFinished {
    pub model_id: String,
    pub success: bool,
    pub cancelled: bool,
    pub error: Option<NabuError>,
}

/// Set of downloads to interrupt, shared with the Tauri commands.
#[derive(Default)]
pub struct DownloadManager {
    cancelled: Mutex<HashSet<String>>,
    active: Mutex<HashSet<String>>,
}

impl DownloadManager {
    pub fn request_cancel(&self, model_id: &str) {
        self.cancelled.lock().unwrap().insert(model_id.to_string());
    }

    fn is_cancelled(&self, model_id: &str) -> bool {
        self.cancelled.lock().unwrap().contains(model_id)
    }

    /// Marks the download as running. Returns `false` if one is already in
    /// flight, so two clicks cannot write to the same file at once.
    fn try_begin(&self, model_id: &str) -> bool {
        let mut active = self.active.lock().unwrap();
        if active.contains(model_id) {
            return false;
        }
        active.insert(model_id.to_string());
        self.cancelled.lock().unwrap().remove(model_id);
        true
    }

    fn finish(&self, model_id: &str) {
        self.active.lock().unwrap().remove(model_id);
        self.cancelled.lock().unwrap().remove(model_id);
    }
}

/// Root folder for models: `<app data dir>/models`.
pub fn models_root(app: &AppHandle) -> NabuResult<PathBuf> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| error::disk_write(e))?;
    Ok(base.join("models"))
}

pub fn model_dir(app: &AppHandle, model_id: &str) -> NabuResult<PathBuf> {
    Ok(models_root(app)?.join(model_id))
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Inspects the disk to see which files of a model are already present.
pub fn status_of(app: &AppHandle, spec: &ModelSpec) -> ModelStatus {
    let dir = model_dir(app, spec.id).unwrap_or_default();

    let mut complete = 0usize;
    let mut bytes_on_disk = 0u64;

    for file in spec.files {
        let final_path = dir.join(file.file_name);
        if final_path.is_file() {
            complete += 1;
            bytes_on_disk += file_len(&final_path);
        } else {
            // A leftover .part still counts as progress.
            bytes_on_disk += file_len(&dir.join(format!("{}.part", file.file_name)));
        }
    }

    let state = if complete == spec.files.len() {
        InstallState::Ready
    } else if bytes_on_disk > 0 {
        InstallState::Partial
    } else {
        InstallState::NotDownloaded
    };

    ModelStatus {
        id: spec.id.to_string(),
        name: spec.name.to_string(),
        task: spec.task,
        description_key: spec.description_key.to_string(),
        license: spec.license.to_string(),
        commercial_use: spec.commercial_use,
        vram_required_mb: spec.vram_required_mb,
        total_bytes: spec.total_bytes(),
        state,
        bytes_on_disk,
        scale_factor: spec.scale_factor,
        dir: dir.to_string_lossy().to_string(),
    }
}

/// Hashes a file that is already on disk, so an interrupted `.part` can be
/// resumed without losing the ability to verify the final digest.
async fn hash_existing(path: &Path) -> NabuResult<(Sha256, u64)> {
    let mut hasher = Sha256::new();
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| error::disk_write(e))?;

    let mut buf = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf).await.map_err(|e| error::disk_write(e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hasher, total))
}

struct Cancelled;

/// Downloads one file, resuming from an existing `.part` when possible.
#[allow(clippy::too_many_arguments)]
async fn download_file(
    app: &AppHandle,
    manager: &DownloadManager,
    client: &reqwest::Client,
    spec: &ModelSpec,
    file: &ModelFile,
    file_index: usize,
    dir: &Path,
    already_done_bytes: u64,
    total_bytes: u64,
) -> NabuResult<Result<(), Cancelled>> {
    let final_path = dir.join(file.file_name);
    if final_path.is_file() {
        return Ok(Ok(()));
    }

    let part_path = dir.join(format!("{}.part", file.file_name));

    let (mut hasher, mut resume_from) = if part_path.is_file() {
        hash_existing(&part_path).await.unwrap_or((Sha256::new(), 0))
    } else {
        (Sha256::new(), 0)
    };

    if resume_from > file.size_bytes {
        // The local file is larger than expected: start over.
        let _ = tokio::fs::remove_file(&part_path).await;
        hasher = Sha256::new();
        resume_from = 0;
    }

    let mut request = client.get(file.url);
    if resume_from > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
    }

    let response = request
        .send()
        .await
        .map_err(|e| error::connection_failed(file.file_name, e))?;

    // If the server ignores Range and sends the whole file, restart the hash.
    if resume_from > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        hasher = Sha256::new();
        resume_from = 0;
    }

    if !response.status().is_success() {
        return Err(error::http_status(file.file_name, response.status()));
    }

    let mut out = if resume_from > 0 {
        tokio::fs::OpenOptions::new().append(true).open(&part_path).await
    } else {
        tokio::fs::File::create(&part_path).await
    }
    .map_err(|e| error::disk_write(e))?;

    let mut downloaded = resume_from;
    let mut last_emit = std::time::Instant::now();
    let mut last_bytes = downloaded;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        if manager.is_cancelled(spec.id) {
            let _ = out.flush().await;
            return Ok(Err(Cancelled));
        }

        let chunk = chunk.map_err(|e| error::connection_failed(file.file_name, e))?;
        hasher.update(&chunk);
        out.write_all(&chunk)
            .await
            .map_err(|e| error::disk_write(e))?;
        downloaded += chunk.len() as u64;

        // One event every ~200 ms: enough for a smooth bar without flooding IPC.
        let elapsed = last_emit.elapsed();
        if elapsed.as_millis() >= 200 {
            let speed = ((downloaded - last_bytes) as f64 / elapsed.as_secs_f64()) as u64;
            let overall = already_done_bytes + downloaded;
            let _ = app.emit(
                "model-download-progress",
                DownloadProgress {
                    model_id: spec.id.to_string(),
                    file_name: file.file_name.to_string(),
                    file_index,
                    file_count: spec.files.len(),
                    downloaded_bytes: overall,
                    total_bytes,
                    percent: if total_bytes > 0 {
                        (overall as f64 / total_bytes as f64) * 100.0
                    } else {
                        0.0
                    },
                    bytes_per_second: speed,
                },
            );
            last_emit = std::time::Instant::now();
            last_bytes = downloaded;
        }
    }

    out.flush().await.map_err(|e| error::disk_write(e))?;
    drop(out);

    // Verify integrity before promoting the .part to its final name.
    if let Some(expected) = file.sha256 {
        let actual = format!("{:x}", hasher.finalize());
        if actual != expected {
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(error::checksum_failed(file.file_name));
        }
    }

    tokio::fs::rename(&part_path, &final_path)
        .await
        .map_err(|e| error::disk_write(e))?;

    Ok(Ok(()))
}

/// Downloads every file of a model, emitting progress and the final outcome.
pub async fn download_model(
    app: AppHandle,
    manager: Arc<DownloadManager>,
    model_id: String,
) -> NabuResult<()> {
    let spec = registry::find(&model_id).ok_or_else(|| error::unknown_model(&model_id))?;

    if !manager.try_begin(&model_id) {
        return Err(error::download_in_progress());
    }

    let result = run_download(&app, &manager, spec).await;
    manager.finish(&model_id);

    let (success, cancelled, failure) = match &result {
        Ok(true) => (true, false, None),
        Ok(false) => (false, true, None),
        Err(e) => (false, false, Some(e.clone())),
    };

    let _ = app.emit(
        "model-download-finished",
        DownloadFinished {
            model_id: model_id.clone(),
            success,
            cancelled,
            error: failure,
        },
    );

    result.map(|_| ())
}

/// Returns `Ok(false)` when the user cancelled.
async fn run_download(
    app: &AppHandle,
    manager: &DownloadManager,
    spec: &ModelSpec,
) -> NabuResult<bool> {
    let dir = model_dir(app, spec.id)?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| error::disk_write(e))?;

    let client = reqwest::Client::builder()
        .user_agent("Nabu-AI-Editor")
        // Some mirrors answer with several redirects in a row.
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| error::unknown(e))?;

    let total_bytes = spec.total_bytes();
    let mut already_done = 0u64;

    for (index, file) in spec.files.iter().enumerate() {
        let outcome = download_file(
            app,
            manager,
            &client,
            spec,
            file,
            index,
            &dir,
            already_done,
            total_bytes,
        )
        .await?;

        if outcome.is_err() {
            return Ok(false);
        }
        already_done += file.size_bytes;
    }

    // Final event at 100%, so the interface does not stop at an in-between value.
    let _ = app.emit(
        "model-download-progress",
        DownloadProgress {
            model_id: spec.id.to_string(),
            file_name: String::new(),
            file_index: spec.files.len(),
            file_count: spec.files.len(),
            downloaded_bytes: total_bytes,
            total_bytes,
            percent: 100.0,
            bytes_per_second: 0,
        },
    );

    Ok(true)
}

/// Removes every file of a model from disk, including any `.part`.
pub fn delete_model(app: &AppHandle, model_id: &str) -> NabuResult<()> {
    let spec = registry::find(model_id).ok_or_else(|| error::unknown_model(model_id))?;
    let dir = model_dir(app, spec.id)?;
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| error::disk_write(e))?;
    }
    Ok(())
}
