// src-tauri/src/lib.rs
// Public so integration tests can drive the engine through the same modules
// the application uses, instead of re-including source files by path.
pub mod ai;
pub mod panorama;

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

/// One click on the picture: this belongs to the object, or this does not.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RefinePoint {
    x: f32,
    y: f32,
    positive: bool,
}

/// Everything one refinement needs, as a single argument.
///
/// A struct rather than ten parameters, because the selection being corrected
/// has to travel with the corrections: the decoder is stateless, so "refine
/// what is already selected" is expressed by sending the original prompt and
/// the clicks together and decoding the lot.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RefineRequest {
    model_id: String,
    image_data_url: String,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    outline: Option<Vec<[f32; 2]>>,
    points: Vec<RefinePoint>,
}

/// Re-cuts a selection from a handful of clicks, reusing what the encoder
/// already saw.
///
/// The two stages of the segmenter are nothing alike in cost: looking at a
/// nine-megapixel photograph takes seconds, answering a further prompt about
/// what it saw takes fifty milliseconds. So a selection does not have to be
/// right first time. Point at the wheel the segmenter missed, or at the hedge
/// it took in by mistake, and the answer comes back inside a frame.
///
/// This is also the honest answer to a limit we measured rather than guessed:
/// a box prompt lands five points on an object, and on anything with outlying
/// parts — wheels, mirrors, a handle — those five points never mention them.
/// No automatic rule separated the missing parts from the background reliably.
/// A click does.
#[tauri::command]
async fn refine_selection(
    app: AppHandle,
    cache: State<'_, Arc<SessionCache>>,
    request: RefineRequest,
) -> Result<String, NabuError> {
    let RefineRequest {
        model_id,
        image_data_url,
        x0,
        y0,
        x1,
        y1,
        outline,
        points,
    } = request;

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

        // Keyed by the picture itself, so a different photograph is encoded
        // afresh and the same one never is.
        let key = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(image_data_url.as_bytes());
            format!("{:x}", hasher.finalize())
        };

        let embedding = cache.embedding_for(&key, || {
            let encoder = engine::load_file(&handle, &cache, &model_id, encoder_file)?;
            let mut encoder = encoder.lock().map_err(|_| error::session_unavailable())?;
            tasks::encode_for_segmentation(&mut encoder, &source, &|_, _| {})
        })?;

        let decoder = engine::load_file(&handle, &cache, &model_id, decoder_file)?;
        let mut decoder = decoder.lock().map_err(|_| error::session_unavailable())?;

        // The selection being corrected comes first, expressed the same way the
        // first cut expressed it, and the clicks follow. Working that rule out
        // a second time on the other side of the bridge is how the correction
        // and the thing it corrects drift apart.
        let (width, height) = source.dimensions();
        let region = vision::region::Region::from_polygon(
            outline.as_deref().unwrap_or(&[]),
            width,
            height,
        );
        let drawn = region
            .as_ref()
            .map(|region| region.prompt_points(5))
            .unwrap_or_default();

        let mut prompt = tasks::seed_points(tasks::BoxPrompt { x0, y0, x1, y1 }, &drawn);
        prompt.extend(points.iter().map(|point| tasks::PromptPoint {
            x: point.x,
            y: point.y,
            positive: point.positive,
        }));

        let candidates = tasks::decode_masks(&mut decoder, &embedding, &prompt)?;
        let best = tasks::pick_mask(&candidates)
            .ok_or_else(|| error::inference_failed("decoder produced no usable mask"))?;
        let mask = tasks::lift_mask(&embedding, &best.mask);

        let mut result = source.clone();
        for (x, y, pixel) in result.enumerate_pixels_mut() {
            pixel[3] = mask.get_pixel(x, y)[0];
        }
        imaging::encode_png_data_url(&result)
    })
    .await
    .map_err(error::unknown)?
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
    outline: Option<Vec<[f32; 2]>>,
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

        let result = segment_within(
            &mut encoder,
            &mut decoder,
            &source,
            tasks::BoxPrompt { x0, y0, x1, y1 },
            outline.as_deref().unwrap_or(&[]),
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
        let session = cache.get_or_load_at(&key, &detector.path, detector.manifest.cpu_only)?;
        let mut session = session.lock().map_err(|_| error::session_unavailable())?;

        let detections = vision::detect::run(&mut session, &detector.manifest, &source, &emit)?;
        emit(100, "done");

        Ok(vision::scene::build_scene(&source, detections, &detector.model_id))
    })
    .await
    .map_err(error::unknown)?
}

/// Segments an object, held to the shape the user drew.
///
/// The drawing does two jobs and the segmenter alone honours neither: it says
/// *where to point* — points taken from the places furthest inside it, because
/// the centre of a bounding box is not inside anything that is not roughly
/// square — and it says *how far the answer may reach*, which is what stops a
/// misread from erasing something on the other side of the photograph.
///
/// With no drawing, both fall back to the bounding box, which is all a
/// suggestion resolved from the scene model ever has.
fn segment_within(
    encoder: &mut ort::session::Session,
    decoder: &mut ort::session::Session,
    source: &image::RgbaImage,
    prompt: tasks::BoxPrompt,
    outline: &[[f32; 2]],
    progress: tasks::ProgressFn<'_>,
) -> Result<image::RgbaImage, NabuError> {
    let (width, height) = source.dimensions();
    let region = vision::region::Region::from_polygon(outline, width, height);
    let points = region
        .as_ref()
        .map(|region| region.prompt_points(5))
        .unwrap_or_default();

    let mut result = tasks::segment_object(encoder, decoder, source, prompt, &points, progress)?;

    let Some(region) = region else { return Ok(result) };

    // The segmenter answers with the source carrying its mask as alpha, so the
    // judgement is made on that alpha and written back to it.
    let mut mask = image::GrayImage::new(width, height);
    for (x, y, px) in result.enumerate_pixels() {
        mask.put_pixel(x, y, image::Luma([px[3]]));
    }

    vision::region::constrain(&mut mask, &region);

    for (x, y, px) in result.enumerate_pixels_mut() {
        px[3] = mask.get_pixel(x, y)[0];
    }

    Ok(result)
}

/// How far away everything in the photo is, relative to the rest of it.
///
/// Returned as a greyscale PNG rather than a numeric array: it is consumed as
/// a texture by the renderer, a PNG is an order of magnitude smaller across the
/// bridge than the equivalent JSON, and every effect that reads it wants a
/// picture anyway.
///
/// Depth is optional throughout. With no model installed this reports it, and
/// the interface falls back to the geometric shapes that need no model at all.
#[tauri::command]
async fn analyze_depth(
    app: AppHandle,
    cache: State<'_, Arc<SessionCache>>,
    image_data_url: String,
) -> Result<String, NabuError> {
    let cache = cache.inner().clone();
    let handle = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let source = imaging::decode_data_url(&image_data_url)?;
        let model =
            vision::catalog::best_depth(&handle).ok_or_else(|| error::no_model_selected("depth"))?;

        let emit = |percent: u32, stage: &str| {
            let _ = handle.emit(
                "ai-task-progress",
                AiProgress { percent, stage: stage.to_string() },
            );
        };
        emit(5, "load");

        let key = format!("{}/{}", model.model_id, model.manifest.file);
        let session = cache.get_or_load_at(&key, &model.path, model.manifest.cpu_only)?;
        let mut session = session.lock().map_err(|_| error::session_unavailable())?;

        let map = vision::depth::run(&mut session, &model.manifest, &source, &emit)?;
        emit(95, "compose");

        // Greyscale out, RGBA in: the shared encoder speaks one format, and a
        // depth map with equal channels is exactly what the shader samples.
        let mut rgba = image::RgbaImage::new(map.image.width(), map.image.height());
        for (x, y, px) in map.image.enumerate_pixels() {
            rgba.put_pixel(x, y, image::Rgba([px[0], px[0], px[0], 255]));
        }

        let encoded = imaging::encode_png_data_url(&rgba)?;
        emit(100, "done");
        Ok(encoded)
    })
    .await
    .map_err(error::unknown)?
}

/// Language models the router can use, in order of preference.
///
/// Measured on the routing test suite (tests/router.rs), both models answer
/// in the same ~10-14s per phrase on CPU — the cost is dominated by prompt
/// processing, not model size — but the 1.5B routes 5/6 test groups correctly
/// against 3/6 for the 0.5B, which misreads French and vague phrasing. Equal
/// speed and better answers put the 1.5B first; the 0.5B stays as a smaller
/// download for machines where a gigabyte of RAM matters.
const ROUTER_MODELS: &[&str] = &["qwen2.5-1.5b-instruct", "qwen2.5-0.5b-instruct"];

/// The first router model in the preference order that is ready on disk.
fn installed_router_model(app: &AppHandle) -> Option<&'static registry::ModelSpec> {
    ROUTER_MODELS.iter().find_map(|id| {
        registry::find(id)
            .filter(|spec| download::status_of(app, spec).state == download::InstallState::Ready)
    })
}

/// Removes the object inside a rectangle and rebuilds the background.
///
/// The whole point of the vision engine in one command: the box comes from
/// the scene model (via `select_object`/`remove_object` routing), a segmenter
/// turns it into a pixel mask, an inpainter fills the hole. Which concrete
/// models run is the catalogue's business — anything fulfilling the
/// `segmenter` and `inpainter` contracts will do.
#[tauri::command]
async fn remove_object(
    app: AppHandle,
    cache: State<'_, Arc<SessionCache>>,
    image_data_url: String,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    outline: Option<Vec<[f32; 2]>>,
) -> Result<String, NabuError> {
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

        // Stage 1: the segmenter turns the box into a mask.
        let segmenter = vision::catalog::best_segmenter(&handle)?;
        let vision::manifest::OutputSpec::Sam { decoder } = &segmenter.manifest.output else {
            return Err(error::inference_failed(format!(
                "manifest {} is not a segmenter family",
                segmenter.manifest.id
            )));
        };

        let encoder_key = format!("{}/{}", segmenter.model_id, segmenter.manifest.file);
        let decoder_key = format!("{}/{}", segmenter.model_id, decoder);
        let cpu_only = segmenter.manifest.cpu_only;
        let encoder = cache.get_or_load_at(&encoder_key, &segmenter.path, cpu_only)?;
        let decoder = cache.get_or_load_at(&decoder_key, &segmenter.dir.join(decoder), cpu_only)?;

        let masked = {
            let mut encoder = encoder.lock().map_err(|_| error::session_unavailable())?;
            let mut decoder = decoder.lock().map_err(|_| error::session_unavailable())?;
            segment_within(
                &mut encoder,
                &mut decoder,
                &source,
                tasks::BoxPrompt { x0, y0, x1, y1 },
                outline.as_deref().unwrap_or(&[]),
                &emit,
            )?
        };

        // The segmenter contract returns the source with the mask as alpha.
        let mut mask = image::GrayImage::new(source.width(), source.height());
        for (x, y, px) in masked.enumerate_pixels() {
            mask.put_pixel(x, y, image::Luma([px[3]]));
        }

        // What the object casts goes with the object.
        //
        // A segmenter marks the bicycle and stops there, correctly: the shadow
        // on the paving is not the bicycle. But removing the one and leaving
        // the other removes nothing — the shape on the ground still says
        // bicycle. This is done here rather than inside the inpainter because
        // it is only right for a mask that means "this object": a region the
        // user drew by hand means the region they drew and nothing more.
        emit(15, "shadow");
        let mask = vision::shadow::with_shadow(&source, &mask);

        // Stage 2: the inpainter rebuilds what the mask covers.
        let inpainter = vision::catalog::best_inpainter(&handle)?;
        let key = format!("{}/{}", inpainter.model_id, inpainter.manifest.file);
        let session = cache.get_or_load_at(&key, &inpainter.path, inpainter.manifest.cpu_only)?;
        let mut session = session.lock().map_err(|_| error::session_unavailable())?;

        let result =
            vision::inpaint::remove_region(&mut session, &inpainter.manifest, &source, &mask, &emit)?;

        imaging::encode_png_data_url(&result)
    })
    .await
    .map_err(error::unknown)?
}

/// Starts loading the language router in the background.
///
/// Fire-and-forget: the interface calls it once at startup when the model is
/// installed, so the first typed request answers in seconds instead of paying
/// for a gigabyte of weights on the spot. Doing nothing when the model is
/// absent, or the build has no language engine, is the correct behaviour.
#[cfg(feature = "llm")]
#[tauri::command]
fn warm_router(app: AppHandle) {
    let Some(spec) = installed_router_model(&app) else { return };
    let Ok(dir) = download::model_dir(&app, spec.id) else { return };
    let file = spec.files[0].file_name;

    tauri::async_runtime::spawn(async move {
        if let Err(err) = ai::router::preload(&dir.to_string_lossy(), file).await {
            eprintln!("[nabu] router warm-up failed: {err}");
        }
    });
}

#[cfg(not(feature = "llm"))]
#[tauri::command]
fn warm_router(_app: AppHandle) {}

/// Whether a typed request can be understood rather than keyword-matched.
///
/// False when the build has no language engine, or when the model has not been
/// downloaded. The interface uses it to decide whether to offer free-form
/// requests or to stay with the phrases it can pattern-match.
#[tauri::command]
fn router_available(app: AppHandle) -> bool {
    cfg!(feature = "llm") && installed_router_model(&app).is_some()
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
    let spec = installed_router_model(&app)
        .ok_or_else(|| error::model_not_downloaded("Qwen2.5 Instruct"))?;

    let dir = download::model_dir(&app, spec.id)?;
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

/// Reads the scene and proposes one thing worth doing, in the user's language.
///
/// The second tier of the suggestions panel. The first tier is arithmetic and
/// is already on screen by the time this is called; this arrives when it
/// arrives and slots in above it, so its cost on CPU is never something anyone
/// waits for. Returns the model's JSON verbatim for the interface to validate —
/// the recipe name and the object id are checked against what exists, because
/// a small model asked for a word from a list will occasionally invent one.
#[cfg(feature = "llm")]
#[tauri::command]
async fn advise_scene(
    app: AppHandle,
    scene: String,
    language: String,
) -> Result<String, NabuError> {
    let spec = installed_router_model(&app)
        .ok_or_else(|| error::model_not_downloaded("Qwen2.5 Instruct"))?;

    let dir = download::model_dir(&app, spec.id)?;
    let file = spec.files[0].file_name;

    ai::router::advise(&dir.to_string_lossy(), file, &scene, &language).await
}

/// Without the language engine there is no second tier; the first still works.
#[cfg(not(feature = "llm"))]
#[tauri::command]
async fn advise_scene(
    _app: AppHandle,
    _scene: String,
    _language: String,
) -> Result<String, NabuError> {
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
            app.manage(Arc::new(panorama::PanoramaState::default()));
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
            warm_router,
            interpret_command,
            analyze_scene,
            analyze_depth,
            advise_scene,
            remove_object,
            segment_object,
            refine_selection,
            process_ai_task,
            panorama::stitch_panorama,
            panorama::cancel_panorama
        ])
        .run(tauri::generate_context!())
        .expect("error while starting the Nabu application");
}
