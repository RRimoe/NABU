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
    let result = tasks::upscale(&mut session, &source, 2, &noop).expect("upscale");

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
