//! Verifica end-to-end delle pipeline di inferenza.
//!
//! Scarica i modelli in `target/test-models/` alla prima esecuzione e li
//! riusa in seguito. Richiede rete la prima volta:
//!     cargo test --test inference -- --ignored --nocapture

use std::path::{Path, PathBuf};

use image::{Rgba, RgbaImage};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

#[path = "../src/ai/imaging.rs"]
mod imaging;
#[path = "../src/ai/error.rs"]
mod error;
#[path = "../src/ai/tasks.rs"]
mod tasks;

fn cache_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test-models");
    std::fs::create_dir_all(&dir).expect("cartella cache");
    dir
}

/// Scarica il modello se assente e restituisce il percorso locale.
fn ensure_model(file_name: &str, url: &str) -> PathBuf {
    let path = cache_dir().join(file_name);
    if path.is_file() {
        return path;
    }

    eprintln!("scarico {file_name}...");
    let bytes = reqwest::blocking::Client::builder()
        .user_agent("Nabu-AI-Editor")
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .expect("client")
        .get(url)
        .send()
        .expect("richiesta")
        .error_for_status()
        .expect("stato HTTP")
        .bytes()
        .expect("corpo");

    std::fs::write(&path, &bytes).expect("scrittura del modello");
    path
}

fn open_session(path: &Path) -> Session {
    Session::builder()
        .expect("builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("optimization level")
        .commit_from_file(path)
        .expect("caricamento del modello")
}

/// Immagine di prova: disco chiaro centrato su fondo scuro, cioè un soggetto
/// che qualunque modello di matting deve saper separare.
fn test_image(size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let c = size as f32 / 2.0;
    let r = size as f32 / 3.0;
    for (x, y, px) in img.enumerate_pixels_mut() {
        let dx = x as f32 - c;
        let dy = y as f32 - c;
        *px = if dx * dx + dy * dy <= r * r {
            Rgba([235, 90, 60, 255])
        } else {
            Rgba([18, 22, 30, 255])
        };
    }
    img
}

fn noop(_: u32, _: &str) {}

#[test]
fn data_url_round_trip_preserves_pixels() {
    let original = test_image(64);
    let url = imaging::encode_png_data_url(&original).expect("codifica");
    assert!(url.starts_with("data:image/png;base64,"));

    let decoded = imaging::decode_data_url(&url).expect("decodifica");
    assert_eq!(decoded.dimensions(), original.dimensions());
    assert_eq!(decoded.as_raw(), original.as_raw(), "round-trip non fedele");
}

#[test]
fn decode_rejects_garbage() {
    assert!(imaging::decode_data_url("data:image/png;base64,non-e-base64!!").is_err());
}

#[test]
#[ignore = "richiede il modello ISNet (179 MB)"]
fn background_removal_produces_a_meaningful_mask() {
    let path = ensure_model(
        "isnet-general-use.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
    );
    let mut session = open_session(&path);

    let source = test_image(256);
    let result = tasks::remove_background(&mut session, &source, &noop).expect("matting");

    assert_eq!(result.dimensions(), source.dimensions(), "dimensioni alterate");

    // I colori RGB non devono cambiare: solo il canale alfa.
    for (x, y, px) in result.enumerate_pixels() {
        let src = source.get_pixel(x, y);
        assert_eq!([px[0], px[1], px[2]], [src[0], src[1], src[2]]);
    }

    // Il centro (soggetto) deve restare più opaco degli angoli (sfondo).
    let center = result.get_pixel(128, 128)[3];
    let corner = result.get_pixel(3, 3)[3];
    println!("alfa centro={center} angolo={corner}");
    assert!(
        center > corner,
        "il soggetto non è più opaco dello sfondo (centro={center}, angolo={corner}): \
         normalizzazione o layout del tensore probabilmente errati"
    );
    assert!(center > 128, "soggetto troppo trasparente: alfa={center}");
}

#[test]
#[ignore = "richiede il modello Swin2SR (8 MB)"]
fn upscale_doubles_the_resolution() {
    let path = ensure_model(
        "swin2sr-lightweight-x2.onnx",
        "https://huggingface.co/Xenova/swin2SR-lightweight-x2-64/resolve/main/onnx/model.onnx",
    );
    let mut session = open_session(&path);

    let source = test_image(96);
    let result = tasks::upscale(&mut session, &source, 2, None, &noop).expect("upscale");

    assert_eq!(result.dimensions(), (192, 192), "fattore di scala errato");

    // L'immagine ingrandita deve conservare la struttura: centro chiaro,
    // angolo scuro. Se il tensore fosse trasposto, questo fallirebbe.
    let center = result.get_pixel(96, 96);
    let corner = result.get_pixel(4, 4);
    println!("centro={:?} angolo={:?}", center, corner);
    assert!(
        center[0] > corner[0] + 40,
        "struttura dell'immagine non conservata dopo l'upscaling"
    );
    assert_eq!(center[3], 255, "canale alfa perso");
}

/// The fixed-input path of the upscaler: Real-ESRGAN x4plus only accepts
/// 128x128 tensors, so the tiling has to pad partial tiles up to that size and
/// crop the result back. A 200x120 source exercises both partial dimensions.
#[test]
#[ignore = "requires the Real-ESRGAN model (67 MB)"]
fn fixed_input_upscaling_quadruples_the_resolution() {
    let path = ensure_model(
        "realesrgan-x4plus-128.onnx",
        "https://huggingface.co/bukuroo/RealESRGAN-ONNX/resolve/main/real-esrgan-x4plus-128.onnx",
    );
    let mut session = open_session(&path);

    let mut source = RgbaImage::new(200, 120);
    for (x, y, px) in source.enumerate_pixels_mut() {
        let dx = x as f32 - 100.0;
        let dy = y as f32 - 60.0;
        *px = if dx * dx + dy * dy <= 40.0 * 40.0 {
            Rgba([235, 90, 60, 255])
        } else {
            Rgba([18, 22, 30, 255])
        };
    }

    let result = tasks::upscale(&mut session, &source, 4, Some(128), &noop).expect("upscale");

    assert_eq!(result.dimensions(), (800, 480), "wrong scale factor");

    // Structure preserved: bright centre, dark corner. A padding or cropping
    // mistake in the fixed-size path would shift or repeat tiles.
    let center = result.get_pixel(400, 240);
    let corner = result.get_pixel(8, 8);
    println!("centre={center:?} corner={corner:?}");
    assert!(center[0] > corner[0] + 40, "image structure lost");
    assert_eq!(center[3], 255, "alpha lost");
}

/// Every installed vision model, loaded exactly as the application loads it:
/// the manifest decides whether the accelerator is offered (`cpuOnly`), and
/// one real inference must succeed. This is the test that caught LaMa's
/// Fourier convolutions failing on DirectML with "parametro non corretto" —
/// an operator the provider accepts at load time and rejects at run time,
/// which no load-only check can see.
#[test]
#[ignore = "requires the models on disk"]
fn vision_models_run_the_way_the_application_loads_them() {
    use nabu_lib::ai::vision::manifest::ModelManifest;

    let models = std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.editor.nabu/models");

    let source = test_image(512);

    // Detector.
    let dfine = models.join("dfine-nano/model.onnx");
    if dfine.is_file() {
        let manifest =
            ModelManifest::parse(include_str!("../manifests/dfine-nano.json")).expect("manifest");
        let mut session =
            nabu_lib::ai::engine::build_session(&dfine, manifest.cpu_only).expect("dfine session");
        nabu_lib::ai::vision::detect::run(&mut session, &manifest, &source, &noop)
            .expect("dfine inference");
        println!("ok  dfine-nano (cpu_only={})", manifest.cpu_only);
    }

    // Segmenter.
    let sam_dir = models.join("slimsam-77");
    if sam_dir.join("vision_encoder.onnx").is_file() {
        let manifest =
            ModelManifest::parse(include_str!("../manifests/slimsam-77.json")).expect("manifest");
        let mut encoder =
            nabu_lib::ai::engine::build_session(&sam_dir.join("vision_encoder.onnx"), manifest.cpu_only)
                .expect("sam encoder");
        let mut decoder = nabu_lib::ai::engine::build_session(
            &sam_dir.join("prompt_encoder_mask_decoder.onnx"),
            manifest.cpu_only,
        )
        .expect("sam decoder");
        tasks::segment_object(
            &mut encoder,
            &mut decoder,
            &source,
            tasks::BoxPrompt { x0: 120.0, y0: 120.0, x1: 392.0, y1: 392.0 },
            &[],
            &noop,
        )
        .expect("sam inference");
        println!("ok  slimsam-77 (cpu_only={})", manifest.cpu_only);
    }

    // Inpainter — cpuOnly in its manifest, because of the DirectML failure.
    let lama = models.join("lama-inpainting/big-lama.onnx");
    if lama.is_file() {
        let manifest = ModelManifest::parse(include_str!("../manifests/lama-inpainting.json"))
            .expect("manifest");
        assert!(manifest.cpu_only, "LaMa must stay off the GPU providers");
        let mut session =
            nabu_lib::ai::engine::build_session(&lama, manifest.cpu_only).expect("lama session");

        let mut mask = image::GrayImage::new(512, 512);
        for (x, y, px) in mask.enumerate_pixels_mut() {
            let dx = x as f32 - 256.0;
            let dy = y as f32 - 256.0;
            px[0] = if dx * dx + dy * dy < 60.0 * 60.0 { 255 } else { 0 };
        }
        nabu_lib::ai::vision::inpaint::remove_region(&mut session, &manifest, &source, &mask, &noop)
            .expect("lama inference");
        println!("ok  lama-inpainting (cpu_only={})", manifest.cpu_only);
    }
}

/// The full object-removal chain: segmenter (SlimSAM) makes the mask,
/// inpainter (LaMa) rebuilds the background. Uses the models installed in the
/// application folder, like the segmentation test below.
///
/// A red disc on a dark textured field, a box around it: afterwards the disc
/// region must no longer be red, and the field outside must be untouched.
#[test]
#[ignore = "requires the SlimSAM and LaMa models on disk"]
fn object_removal_rebuilds_the_background() {
    let models = std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.editor.nabu/models");
    let sam_dir = models.join("slimsam-77");
    let lama = models.join("lama-inpainting/big-lama.onnx");
    if !sam_dir.join("vision_encoder.onnx").is_file() || !lama.is_file() {
        eprintln!("models not downloaded, skipping");
        return;
    }

    let mut encoder = open_session(&sam_dir.join("vision_encoder.onnx"));
    let mut decoder = open_session(&sam_dir.join("prompt_encoder_mask_decoder.onnx"));
    let mut inpainter = open_session(&lama);

    // Dark field with mild texture, red disc left of centre.
    let size = 512u32;
    let mut source = RgbaImage::new(size, size);
    for (x, y, px) in source.enumerate_pixels_mut() {
        let dx = x as f32 - 170.0;
        let dy = y as f32 - 256.0;
        let base = 24 + ((x / 32 + y / 32) % 2) as u8 * 6;
        *px = if dx * dx + dy * dy <= 70.0 * 70.0 {
            Rgba([225, 60, 50, 255])
        } else {
            Rgba([base, base + 4, base + 10, 255])
        };
    }

    let masked = tasks::segment_object(
        &mut encoder,
        &mut decoder,
        &source,
        tasks::BoxPrompt { x0: 80.0, y0: 165.0, x1: 260.0, y1: 350.0 },
        &[],
        &noop,
    )
    .expect("segmentation");

    let mut mask = image::GrayImage::new(size, size);
    for (x, y, px) in masked.enumerate_pixels() {
        mask.put_pixel(x, y, image::Luma([px[3]]));
    }

    let manifest = nabu_lib::ai::vision::manifest::ModelManifest::parse(include_str!(
        "../manifests/lama-inpainting.json"
    ))
    .expect("manifest");

    let result = nabu_lib::ai::vision::inpaint::remove_region(
        &mut inpainter,
        &manifest,
        &source,
        &mask,
        &noop,
    )
    .expect("inpainting");

    assert_eq!(result.dimensions(), source.dimensions());

    // Where the disc was: no longer red, roughly as dark as the field.
    let centre = result.get_pixel(170, 256);
    println!("centre after removal = {centre:?}");
    assert!(
        centre[0] < 120,
        "the object survived removal: centre = {centre:?}"
    );
    assert!(
        centre[0] < centre[2].saturating_add(40),
        "still distinctly red: {centre:?}"
    );

    // Far from the mask the photo must be bit-identical.
    let far = result.get_pixel(480, 30);
    let src_far = source.get_pixel(480, 30);
    assert_eq!(far, src_far, "untouched pixels were altered");
}

/// The shadow leaves with the object that casts it.
///
/// A dark post on bright ground, with the grey it throws across the paving
/// beside it. A segmenter marks the post and only the post, so unless the
/// shadow is added to that mask the removal leaves a post-shaped stain on the
/// ground — which is what happened to the bicycle in the reported photograph.
///
/// The mask is built directly rather than segmented, so this tests the shadow
/// pass and the inpainter and nothing else.
#[test]
#[ignore = "requires the LaMa model on disk"]
fn a_cast_shadow_is_removed_along_with_the_object() {
    let models = std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.editor.nabu/models");
    let lama = models.join("lama-inpainting/big-lama.onnx");
    if !lama.is_file() {
        eprintln!("models not downloaded, skipping");
        return;
    }
    let mut inpainter = open_session(&lama);

    let size = 512u32;
    let mut source = RgbaImage::new(size, size);
    for (x, y, px) in source.enumerate_pixels_mut() {
        let base = 190 + ((x / 24 + y / 24) % 2) as u8 * 12;
        *px = Rgba([base, base, base, 255]);
    }
    // The post.
    for y in 150..330u32 {
        for x in 240..272u32 {
            source.put_pixel(x, y, Rgba([40, 44, 52, 255]));
        }
    }
    // And its shadow: the same ground under half the light, which is exactly
    // what a shadow is and why it can be told apart from a dark object.
    for y in 300..340u32 {
        for x in 272..420u32 {
            let px = *source.get_pixel(x, y);
            source.put_pixel(x, y, Rgba([px[0] / 2, px[1] / 2, px[2] / 2, 255]));
        }
    }

    // The mask a segmenter would hand over: the post, nothing else.
    let mut object = image::GrayImage::new(size, size);
    for y in 150..330u32 {
        for x in 240..272u32 {
            object.put_pixel(x, y, image::Luma([255]));
        }
    }
    assert_eq!(
        object.get_pixel(350, 320)[0],
        0,
        "the segmenter does not mark the shadow, which is the whole problem"
    );

    let widened = nabu_lib::ai::vision::shadow::with_shadow(&source, &object);
    assert_eq!(
        widened.get_pixel(350, 320)[0],
        255,
        "the shadow pass must have taken it in"
    );

    let manifest = nabu_lib::ai::vision::manifest::ModelManifest::parse(include_str!(
        "../manifests/lama-inpainting.json"
    ))
    .expect("manifest");

    let result = nabu_lib::ai::vision::inpaint::remove_region(
        &mut inpainter,
        &manifest,
        &source,
        &widened,
        &noop,
    )
    .expect("inpainting");

    let after = result.get_pixel(350, 320);
    println!("where the shadow was: {after:?}");
    assert!(
        after[0] > 140,
        "the shadow survived the removal: {after:?}"
    );

    let post = result.get_pixel(256, 240);
    assert!(post[0] > 140, "the post survived the removal: {post:?}");

    let far = result.get_pixel(30, 480);
    assert_eq!(far, source.get_pixel(30, 480), "untouched pixels were altered");
}

/// Detection smoke test on a synthetic image.
///
/// A painted scene is not a photograph, so no specific class is asserted: what
/// must hold is that the whole manifest-driven pipeline — preprocessing,
/// inference, DETR decoding, coordinate mapping — runs and returns boxes that
/// lie inside the image with sane confidences.
#[test]
#[ignore = "requires the D-FINE model (15 MB)"]
fn detection_pipeline_runs_end_to_end() {
    let path = ensure_model(
        "dfine-nano.onnx",
        "https://huggingface.co/onnx-community/dfine_n_coco-ONNX/resolve/main/onnx/model.onnx",
    );
    let mut session = open_session(&path);

    let manifest = nabu_lib::ai::vision::manifest::ModelManifest::parse(include_str!(
        "../manifests/dfine-nano.json"
    ))
    .expect("manifest");

    let source = test_image(640);
    let detections = nabu_lib::ai::vision::detect::run(&mut session, &manifest, &source, &noop)
        .expect("detection");

    println!("{} detections", detections.len());
    for det in &detections {
        println!("  {:<14} {:.2}  {:?}", det.label, det.confidence, det.bbox);
        assert!((0.0..=1.0).contains(&det.confidence));
        let [x0, y0, x1, y1] = det.bbox;
        assert!(x0 >= 0.0 && y0 >= 0.0 && x1 <= 640.0 && y1 <= 640.0);
        assert!(x1 > x0 && y1 > y0, "degenerate box survived decoding");
    }
}

/// Two shapes on a plain field: a box drawn around one of them must come back
/// with that shape isolated and the other one gone. If the box prompt were
/// ignored, or the mask misaligned, the second shape would survive.
#[test]
#[ignore = "requires the SlimSAM models on disk"]
fn segmentation_isolates_the_boxed_object() {
    let dir = std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.editor.nabu/models/slimsam-77");
    if !dir.join("vision_encoder.onnx").is_file() {
        eprintln!("SlimSAM not downloaded, skipping");
        return;
    }

    let mut encoder = open_session(&dir.join("vision_encoder.onnx"));
    let mut decoder = open_session(&dir.join("prompt_encoder_mask_decoder.onnx"));

    // Left disc and right square, well separated on a dark field.
    let size = 512u32;
    let mut source = RgbaImage::new(size, size);
    for (x, y, px) in source.enumerate_pixels_mut() {
        let dx = x as f32 - 140.0;
        let dy = y as f32 - 256.0;
        let in_disc = dx * dx + dy * dy <= 80.0 * 80.0;
        let in_square = (330..=450).contains(&x) && (196..=316).contains(&y);
        *px = if in_disc {
            Rgba([240, 80, 60, 255])
        } else if in_square {
            Rgba([70, 220, 120, 255])
        } else {
            Rgba([16, 18, 24, 255])
        };
    }

    // Box loosely around the disc only.
    let prompt = tasks::BoxPrompt { x0: 45.0, y0: 165.0, x1: 235.0, y1: 350.0 };
    let result = tasks::segment_object(&mut encoder, &mut decoder, &source, prompt, &[], &noop)
        .expect("segmentation");

    assert_eq!(result.dimensions(), source.dimensions());

    let disc_alpha = result.get_pixel(140, 256)[3];
    let square_alpha = result.get_pixel(390, 256)[3];
    let field_alpha = result.get_pixel(500, 20)[3];
    println!("alpha disc={disc_alpha} square={square_alpha} field={field_alpha}");

    assert!(disc_alpha > 170, "boxed object not kept: alpha={disc_alpha}");
    assert!(
        square_alpha < disc_alpha / 2,
        "the object outside the box survived: disc={disc_alpha} square={square_alpha}"
    );
    assert!(field_alpha < 128, "background kept: alpha={field_alpha}");
}
