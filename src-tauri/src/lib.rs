// src-tauri/src/lib.rs
mod ai;

use std::sync::Arc;

use ai::download::{self, DownloadManager, ModelStatus};
use ai::engine::{self, SessionCache};
use ai::error::{self, NabuError};
use ai::{imaging, registry, tasks};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

/// Lists the catalogue together with the installation state read from disk.
#[tauri::command]
fn list_models(app: AppHandle) -> Vec<ModelStatus> {
    registry::MODELS
        .iter()
        .map(|spec| download::status_of(&app, spec))
        .collect()
}

/// Starts a model download. Progress arrives through the
/// `model-download-progress` event, the outcome through `model-download-finished`.
#[tauri::command]
async fn download_model(
    app: AppHandle,
    manager: State<'_, Arc<DownloadManager>>,
    model_id: String,
) -> Result<(), NabuError> {
    download::download_model(app, manager.inner().clone(), model_id).await
}

#[tauri::command]
fn cancel_model_download(manager: State<'_, Arc<DownloadManager>>, model_id: String) {
    manager.request_cancel(&model_id);
}

#[tauri::command]
fn delete_model(
    app: AppHandle,
    cache: State<'_, Arc<SessionCache>>,
    model_id: String,
) -> Result<(), NabuError> {
    // Drop the session before the files: it would otherwise hold on to a model
    // that is about to disappear from disk.
    cache.invalidate(&model_id);
    download::delete_model(&app, &model_id)
}

/// Path of the models folder, shown in the interface.
#[tauri::command]
fn models_directory(app: AppHandle) -> Result<String, NabuError> {
    Ok(download::models_root(&app)?.to_string_lossy().to_string())
}

/// Opens the models folder in the system file manager.
#[tauri::command]
fn reveal_models_directory(app: AppHandle) -> Result<(), NabuError> {
    use tauri_plugin_opener::OpenerExt;
    let dir = download::models_root(&app)?;
    std::fs::create_dir_all(&dir).map_err(error::disk_write)?;
    app.opener()
        .open_path(dir.to_string_lossy().to_string(), None::<&str>)
        .map_err(error::unknown)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LanguageFile {
    code: String,
    content: String,
}

/// Reads translations the user dropped into `<app data>/languages`.
///
/// This is what lets someone translate Nabu without a toolchain: a JSON file in
/// that folder is picked up at startup, exactly like a built-in language.
#[tauri::command]
fn list_languages(app: AppHandle) -> Vec<LanguageFile> {
    let Ok(base) = app.path().app_data_dir() else {
        return Vec::new();
    };
    let dir = base.join("languages");

    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };

    entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? != "json" {
                return None;
            }
            Some(LanguageFile {
                code: path.file_stem()?.to_string_lossy().to_string(),
                content: std::fs::read_to_string(&path).ok()?,
            })
        })
        .collect()
}

/// Accelerator detected on this machine, shown in the status bar.
#[tauri::command]
fn accelerator_label() -> String {
    engine::detect_accelerator().label().to_string()
}

/// Progress of a running AI task, emitted to the interface.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiProgress {
    percent: u32,
    /// Stage identifier, translated by the frontend.
    stage: String,
}

/// Runs an AI task on the active layer.
///
/// Inference is blocking, CPU-heavy work: it is moved to a dedicated thread
/// with `spawn_blocking`, otherwise it would stall the async runtime and with
/// it the whole interface.
#[tauri::command]
async fn process_ai_task(
    app: AppHandle,
    cache: State<'_, Arc<SessionCache>>,
    task_type: String,
    model_id: Option<String>,
    image_data_url: Option<String>,
) -> Result<String, NabuError> {
    let image_data_url = image_data_url.ok_or_else(error::no_input_image)?;
    let model_id = model_id.ok_or_else(|| error::no_model_selected(&task_type))?;

    let spec = registry::find(&model_id).ok_or_else(|| error::unknown_model(&model_id))?;
    let scale = spec.scale_factor.unwrap_or(2);

    let cache = cache.inner().clone();
    let handle = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let source = imaging::decode_data_url(&image_data_url)?;

        let emit = |percent: u32, stage: &str| {
            let _ = handle.emit(
                "ai-task-progress",
                AiProgress {
                    percent,
                    stage: stage.to_string(),
                },
            );
        };
        emit(5, "load");

        let session = engine::load_for(&handle, &cache, &model_id)?;
        let mut session = session.lock().map_err(|_| error::session_unavailable())?;

        let result = match task_type.as_str() {
            "background-removal" => tasks::remove_background(&mut session, &source, &emit)?,
            "upscale" => tasks::upscale(&mut session, &source, scale, &emit)?,
            other => return Err(error::unsupported_task(other)),
        };

        imaging::encode_png_data_url(&result)
    })
    .await
    .map_err(error::unknown)?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            app.manage(Arc::new(DownloadManager::default()));
            app.manage(Arc::new(SessionCache::default()));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            download_model,
            cancel_model_download,
            delete_model,
            models_directory,
            reveal_models_directory,
            list_languages,
            accelerator_label,
            process_ai_task
        ])
        .run(tauri::generate_context!())
        .expect("error while starting the Nabu application");
}
