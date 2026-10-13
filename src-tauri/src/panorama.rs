// src-tauri/src/panorama.rs
//! Panorama stitching, as the interface calls it.
//!
//! The photos travel as one binary message rather than as data URLs: a set of
//! ten camera files is a few hundred megabytes, and base64 inside JSON would
//! add a third to that and copy it several times on the way. The result comes
//! back the same way, a small JSON header followed by the PNG.

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageReader, RgbaImage};
use nabu_stitch::{stitch, Options, Projection, Stage, StitchError};
use serde::{Deserialize, Serialize};
use tauri::ipc::{InvokeBody, Request, Response};
use tauri::{AppHandle, Emitter, State};

use crate::ai::error::{self, NabuError, NabuResult};

/// Lets a running stitch be stopped from the interface.
#[derive(Default)]
pub struct PanoramaState {
    cancel: AtomicBool,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct PanoramaOptions {
    /// "auto", "rectilinear", "cylindrical" or "spherical".
    projection: Option<String>,
    crop: Option<bool>,
    max_megapixels: Option<f64>,
    /// The photos arrive in the order they were shot.
    in_order: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PanoramaInfo {
    width: u32,
    height: u32,
    used: Vec<usize>,
    dropped: Vec<usize>,
    projection: &'static str,
    rms: f64,
    horizontal_fov: f64,
    vertical_fov: f64,
}

#[derive(Serialize, Clone)]
struct PanoramaProgress {
    stage: &'static str,
    percent: u32,
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Features => "features",
        Stage::Matching => "matching",
        Stage::Aligning => "aligning",
        Stage::Refining => "refining",
        Stage::Exposure => "exposure",
        Stage::Seams => "seams",
        Stage::Blending => "blending",
    }
}

/// Share of the bar each stage fills, from where it starts to where it ends.
fn stage_span(stage: Stage) -> (f32, f32) {
    match stage {
        Stage::Features => (2.0, 25.0),
        Stage::Matching => (25.0, 45.0),
        Stage::Aligning => (45.0, 55.0),
        Stage::Refining => (55.0, 62.0),
        Stage::Exposure => (62.0, 66.0),
        Stage::Seams => (66.0, 72.0),
        Stage::Blending => (72.0, 95.0),
    }
}

/// Splits the message into files: each is a little-endian u32 length
/// followed by that many bytes.
fn unpack(body: &[u8]) -> NabuResult<Vec<&[u8]>> {
    let mut files = Vec::new();
    let mut rest = body;
    while !rest.is_empty() {
        if rest.len() < 4 {
            return Err(error::panorama_failed("truncated message"));
        }
        let len = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        rest = &rest[4..];
        if rest.len() < len {
            return Err(error::panorama_failed("truncated file"));
        }
        files.push(&rest[..len]);
        rest = &rest[len..];
    }
    Ok(files)
}

/// Decodes a camera file upright: phones store most photos sideways and
/// record the turn in EXIF, and stitching them sideways would fail.
fn decode(bytes: &[u8]) -> NabuResult<RgbaImage> {
    if crate::heic::is_heif(bytes) {
        return crate::heic::decode(bytes);
    }
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(error::undecodable_image)?;
    let mut decoder = reader.into_decoder().map_err(error::undecodable_image)?;
    let orientation = decoder.orientation().map_err(error::undecodable_image)?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(error::undecodable_image)?;
    image.apply_orientation(orientation);
    Ok(image.to_rgba8())
}

#[tauri::command]
pub async fn stitch_panorama(
    app: AppHandle,
    state: State<'_, Arc<PanoramaState>>,
    request: Request<'_>,
) -> Result<Response, NabuError> {
    let InvokeBody::Raw(body) = request.body() else {
        return Err(error::panorama_failed("expected a binary message"));
    };
    let body = body.clone();
    let options: PanoramaOptions = request
        .headers()
        .get("x-nabu-options")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| serde_json::from_str(v).ok())
        .unwrap_or_default();
    let state = state.inner().clone();
    state.cancel.store(false, Ordering::SeqCst);

    tauri::async_runtime::spawn_blocking(move || {
        let emit = |stage: &'static str, percent: f32| {
            let _ = app.emit(
                "panorama-progress",
                PanoramaProgress { stage, percent: percent.round().clamp(0.0, 100.0) as u32 },
            );
        };
        emit("loading", 0.0);

        let files = unpack(&body)?;
        if files.len() < 2 {
            return Err(error::panorama_too_few());
        }
        let images = files.iter().map(|f| decode(f)).collect::<NabuResult<Vec<_>>>()?;
        drop(body);

        let defaults = Options::default();
        let stitch_options = Options {
            projection: match options.projection.as_deref() {
                Some("rectilinear") => Some(Projection::Rectilinear),
                Some("cylindrical") => Some(Projection::Cylindrical),
                Some("spherical") => Some(Projection::Spherical),
                _ => None,
            },
            crop: options.crop.unwrap_or(defaults.crop),
            max_megapixels: options.max_megapixels.unwrap_or(defaults.max_megapixels).clamp(1.0, 400.0),
            in_order: options.in_order.unwrap_or(defaults.in_order),
            ..defaults
        };

        let result = stitch(&images, &stitch_options, &mut |stage, value| {
            let (from, to) = stage_span(stage);
            emit(stage_name(stage), from + (to - from) * value);
            !state.cancel.load(Ordering::SeqCst)
        })
        .map_err(|e| match e {
            StitchError::TooFewImages => error::panorama_too_few(),
            StitchError::NoOverlap => error::panorama_no_overlap(),
            StitchError::Cancelled => error::cancelled(),
            other => error::panorama_failed(other),
        })?;

        emit("encoding", 96.0);
        let info = PanoramaInfo {
            width: result.image.width(),
            height: result.image.height(),
            used: result.used.clone(),
            dropped: result.dropped.clone(),
            projection: match result.projection {
                Projection::Rectilinear => "rectilinear",
                Projection::Cylindrical => "cylindrical",
                Projection::Spherical => "spherical",
            },
            rms: result.rms,
            horizontal_fov: result.field_of_view.0,
            vertical_fov: result.field_of_view.1,
        };
        let header = serde_json::to_vec(&info).map_err(error::unknown)?;

        let mut out = Vec::with_capacity(header.len() + 4 + result.image.len() / 2);
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);
        // Fast compression: the file is decoded once by the interface and
        // re-encoded on export, so a few extra megabytes cost less than the
        // seconds a thorough compressor would take on fifty megapixels.
        PngEncoder::new_with_quality(&mut out, CompressionType::Fast, FilterType::Adaptive)
            .write_image(
                result.image.as_raw(),
                result.image.width(),
                result.image.height(),
                image::ExtendedColorType::Rgba8,
            )
            .map_err(error::unknown)?;
        emit("done", 100.0);
        Ok(Response::new(out))
    })
    .await
    .map_err(error::unknown)?
}

#[tauri::command]
pub fn cancel_panorama(state: State<'_, Arc<PanoramaState>>) {
    state.cancel.store(true, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacks_length_prefixed_files() {
        let mut body = Vec::new();
        for file in [&b"abc"[..], &b""[..], &b"hello"[..]] {
            body.extend_from_slice(&(file.len() as u32).to_le_bytes());
            body.extend_from_slice(file);
        }
        let files = unpack(&body).unwrap();
        assert_eq!(files, vec![&b"abc"[..], &b""[..], &b"hello"[..]]);
        assert!(unpack(&body[..body.len() - 1]).is_err());
    }
}
