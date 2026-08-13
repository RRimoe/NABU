// src-tauri/src/lib.rs
// Public so integration tests can drive the engine through the same modules
// the application uses, instead of re-including source files by path.
pub mod ai;

use std::sync::Arc;

use ai::download::{self, DownloadManager, ModelStatus};
use ai::engine::{self, SessionCache};
use ai::error::{self, NabuError};
use ai::tools::ToolCall;
use ai::vision::{self, scene::SceneModel};
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

/// Cuts out the object inside a rectangle, using the two SlimSAM stages.
///
/// This answers a different question from background removal: not "what is the
/// subject of this photo" but "what is the object here", which is what a rough
/// selection around a car is actually asking.
#[tauri::command]
async fn segment_object(
    app: AppHandle,
    cache: State<'_, Arc<SessionCache>>,
    model_id: String,
    image_data_url: String,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> Result<String, NabuError> {
    let spec = registry::find(&model_id).ok_or_else(|| error::unknown_model(&model_id))?;
    if spec.files.len() < 2 {
        return Err(error::missing_file("prompt_encoder_mask_decoder.onnx", spec.name));
    }

    let encoder_file = spec.files[0].file_name;
    let decoder_file = spec.files[1].file_name;

    let cache = cache.inner().clone();
    let handle = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let source = imaging::decode_data_url(&image_data_url)?;

        let emit = |percent: u32, stage: &str| {
            let _ = handle.emit(
                "ai-task-progress",
                AiProgress { percent, stage: stage.to_string() },
            );
        };
        emit(5, "load");

        let encoder = engine::load_file(&handle, &cache, &model_id, encoder_file)?;
        let decoder = engine::load_file(&handle, &cache, &model_id, decoder_file)?;

        let mut encoder = encoder.lock().map_err(|_| error::session_unavailable())?;
        let mut decoder = decoder.lock().map_err(|_| error::session_unavailable())?;

        let result = tasks::segment_object(
            &mut encoder,
            &mut decoder,
            &source,
            tasks::BoxPrompt { x0, y0, x1, y1 },
            &emit,
        )?;

        imaging::encode_png_data_url(&result)
    })
    .await
    .map_err(error::unknown)?
}

/// Fast pass of the vision engine: what is in this image?
///
/// Runs the best installed detector plus cheap pixel statistics and returns
/// the scene model — the shared representation the language router reasons
/// over. It is invoked once per image in the background; requests that mention
/// objects then resolve against this without touching any pixels again.
#[tauri::command]
async fn analyze_scene(
    app: AppHandle,
    cache: State<'_, Arc<SessionCache>>,
    image_data_url: String,
) -> Result<SceneModel, NabuError> {
    let cache = cache.inner().clone();
    let handle = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let source = imaging::decode_data_url(&image_data_url)?;
        let detector = vision::catalog::best_detector(&handle)?;

        let emit = |percent: u32, stage: &str| {
            let _ = handle.emit(
                "ai-task-progress",
                AiProgress { percent, stage: stage.to_string() },
            );
        };
        emit(5, "load");

        let key = format!("{}/{}", detector.model_id, detector.manifest.file);
        let session = cache.get_or_load_at(&key, &detector.path)?;
        let mut session = session.lock().map_err(|_| error::session_unavailable())?;

        let detections = vision::detect::run(&mut session, &detector.manifest, &source, &emit)?;
        emit(100, "done");

        Ok(vision::scene::build_scene(&source, detections, &detector.model_id))
    })
    .await
    .map_err(error::unknown)?
}

/// Id of the language model the router uses.
const ROUTER_MODEL: &str = "qwen2.5-1.5b-instruct";

/// Whether a typed request can be understood rather than keyword-matched.
///
/// False when the build has no language engine, or when the model has not been
/// downloaded. The interface uses it to decide whether to offer free-form
/// requests or to stay with the phrases it can pattern-match.
#[tauri::command]
fn router_available(app: AppHandle) -> bool {
    if !cfg!(feature = "llm") {
        return false;
    }
    registry::find(ROUTER_MODEL)
        .map(|spec| download::status_of(&app, spec).state == download::InstallState::Ready)
        .unwrap_or(false)
}

/// Turns a typed sentence into a call to one of Nabu's tools.
///
/// The model chooses a tool and fills in its arguments; it never edits an image
/// and never writes anything the user reads. `has_selection` is passed along
/// because it changes which tool is the right answer for the same sentence.
#[cfg(feature = "llm")]
#[tauri::command]
async fn interpret_command(
    app: AppHandle,
    text: String,
    has_selection: bool,
    scene: Option<String>,
) -> Result<ToolCall, NabuError> {
    let spec = registry::find(ROUTER_MODEL).ok_or_else(|| error::unknown_model(ROUTER_MODEL))?;
    if download::status_of(&app, spec).state != download::InstallState::Ready {
        return Err(error::model_not_downloaded(spec.name));
    }

    let dir = download::model_dir(&app, ROUTER_MODEL)?;
    let file = spec.files[0].file_name;

    let mut context = String::from(if has_selection {
        "The user has selected a region of the photo."
    } else {
        "Nothing is selected; the request applies to the whole photo."
    });

    // The scene is how the router sees: a request like "the red car" resolves
    // against this list, so the model never needs the pixels themselves.
    if let Some(scene) = scene.filter(|s| !s.trim().is_empty()) {
        context.push_str("\n\nSCENE — objects detected in the photo:\n");
        context.push_str(&scene);
    }

    ai::router::interpret(&dir.to_string_lossy(), file, &text, &context).await
}

/// Without the language engine there is nothing to interpret with.
#[cfg(not(feature = "llm"))]
#[tauri::command]
async fn interpret_command(
    _app: AppHandle,
    _text: String,
    _has_selection: bool,
    _scene: Option<String>,
) -> Result<ToolCall, NabuError> {
    Err(error::router_unavailable())
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
            "upscale" => tasks::upscale(&mut session, &source, scale, spec.input_size, &emit)?,
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
            router_available,
            interpret_command,
            analyze_scene,
            segment_object,
            process_ai_task
        ])
        .run(tauri::generate_context!())
        .expect("error while starting the Nabu application");
}
