// src-tauri/src/ai/imaging.rs
//! Conversion between the data URLs used by the frontend and in-memory images.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::{DynamicImage, ImageFormat, RgbaImage};
use std::io::Cursor;

use super::error::{self, NabuResult};

/// Decodes a data URL (`data:image/png;base64,...`) into an RGBA image.
pub fn decode_data_url(data_url: &str) -> NabuResult<RgbaImage> {
    let payload = data_url
        .split_once(";base64,")
        .map(|(_, rest)| rest)
        // Also tolerate a bare base64 string with no header.
        .unwrap_or(data_url);

    let bytes = BASE64
        .decode(payload.trim())
        .map_err(error::invalid_data_url)?;

    let image = image::load_from_memory(&bytes).map_err(error::undecodable_image)?;

    Ok(image.to_rgba8())
}

/// Encodes an RGBA image as a PNG data URL.
pub fn encode_png_data_url(image: &RgbaImage) -> NabuResult<String> {
    let mut buffer = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image.clone())
        .write_to(&mut buffer, ImageFormat::Png)
        .map_err(error::unknown)?;

    Ok(format!(
        "data:image/png;base64,{}",
        BASE64.encode(buffer.into_inner())
    ))
}
