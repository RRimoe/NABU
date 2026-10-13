// src-tauri/src/heic.rs
//! HEIC and HEIF photos, the format iPhones shoot in.
//!
//! The webview can show JPEG, PNG and WebP but not HEIC (only Safari's engine
//! can, and only on Apple systems), so the backend decodes it and hands the
//! interface a PNG. The decoder is heifer: pure Rust with its own HEVC
//! decoder, so nothing has to be installed. Of the pure-Rust decoders tried it
//! was the one that read every test file and came closest to libheif (within
//! half a level). The rotation, mirroring and crop stored in the file are
//! applied, so a portrait photo opens upright.

use std::io::Cursor;

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageEncoder, RgbaImage};
use tauri::ipc::{InvokeBody, Request, Response};

use crate::ai::error::{self, NabuError, NabuResult};

/// Whether `bytes` start like a HEIF file: an `ftyp` box whose brand is one of
/// the still-image HEIF brands.
pub fn is_heif(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    matches!(&bytes[8..12], b"heic" | b"heix" | b"heim" | b"heis" | b"hevc" | b"hevx" | b"mif1" | b"msf1")
}

/// Decodes a HEIC/HEIF photo to RGBA, upright.
pub fn decode(bytes: &[u8]) -> NabuResult<RgbaImage> {
    let image = heifer::decode(bytes).map_err(|e| error::undecodable_image(format!("{e:?}")))?;
    let (w, h) = (image.width as u32, image.height as u32);
    RgbaImage::from_raw(w, h, image.to_rgba8())
        .ok_or_else(|| error::undecodable_image("size does not match the pixels"))
}

/// The photo as a PNG the webview can show. Fast compression: it is decoded
/// once by the interface, and the seconds a thorough compressor would take on
/// a 48-megapixel photo buy nothing.
fn to_png(image: &RgbaImage) -> NabuResult<Vec<u8>> {
    let mut out = Vec::with_capacity(image.len() / 2);
    PngEncoder::new_with_quality(Cursor::new(&mut out), CompressionType::Fast, FilterType::Adaptive)
        .write_image(image.as_raw(), image.width(), image.height(), image::ExtendedColorType::Rgba8)
        .map_err(error::unknown)?;
    Ok(out)
}

/// The file's bytes in, a PNG out.
#[tauri::command]
pub async fn decode_heic(request: Request<'_>) -> Result<Response, NabuError> {
    let InvokeBody::Raw(body) = request.body() else {
        return Err(error::undecodable_image("expected a binary message"));
    };
    let body = body.clone();
    tauri::async_runtime::spawn_blocking(move || Ok(Response::new(to_png(&decode(&body)?)?)))
        .await
        .map_err(error::unknown)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_heif_brands_only() {
        let mut file = vec![0, 0, 0, 24];
        file.extend_from_slice(b"ftypheic");
        file.extend_from_slice(&[0; 12]);
        assert!(is_heif(&file));
        file[8..12].copy_from_slice(b"avif");
        assert!(!is_heif(&file));
        assert!(!is_heif(b"\xff\xd8\xff\xe0 jpeg"));
    }

    /// `NABU_HEIC` names a HEIC file; it is decoded and saved next to it as PNG.
    #[test]
    #[ignore]
    fn decodes_a_real_file() {
        let Ok(path) = std::env::var("NABU_HEIC") else { return };
        let bytes = std::fs::read(&path).unwrap();
        assert!(is_heif(&bytes));
        let start = std::time::Instant::now();
        let image = decode(&bytes).unwrap();
        eprintln!("{}x{} in {:?}", image.width(), image.height(), start.elapsed());
        image.save(format!("{path}.png")).unwrap();
    }
}
