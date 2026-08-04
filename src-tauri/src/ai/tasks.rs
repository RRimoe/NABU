// src-tauri/src/ai/tasks.rs
//! Pre- and post-processing pipelines for the installed models.

use image::imageops::FilterType;
use image::{Rgba, RgbaImage};
use ort::session::Session;
use ort::value::Tensor;

use super::error::{self, NabuResult};

/// Working resolution of RMBG-1.4 and ISNet: both derive from IS-Net and are
/// trained on 1024x1024 inputs.
const MATTING_SIZE: u32 = 1024;

/// Callback used to report progress while a task runs.
pub type ProgressFn<'a> = &'a (dyn Fn(u32, &str) + Send + Sync);

/// Runs the model and returns its first output as (shape, data).
fn run_single(session: &mut Session, tensor: Tensor<f32>) -> NabuResult<(Vec<usize>, Vec<f32>)> {
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(error::inference_failed)?;

    let first = outputs
        .iter()
        .next()
        .ok_or_else(|| error::inference_failed("the model produced no output"))?
        .1;

    let (shape, data) = first
        .try_extract_tensor::<f32>()
        .map_err(error::inference_failed)?;

    Ok((shape.iter().map(|d| *d as usize).collect(), data.to_vec()))
}

// ---------------------------------------------------------------------------
// Background removal (RMBG-1.4 / ISNet)
// ---------------------------------------------------------------------------

/// Runs matting and returns the source image with its alpha channel replaced by
/// the mask the model produced.
pub fn remove_background(
    session: &mut Session,
    source: &RgbaImage,
    progress: ProgressFn<'_>,
) -> NabuResult<RgbaImage> {
    let (width, height) = source.dimensions();

    progress(20, "prepare");
    let resized = image::imageops::resize(source, MATTING_SIZE, MATTING_SIZE, FilterType::Triangle);

    // NCHW tensor with the normalisation used during training:
    // (pixel / 255 - 0.5) / 1.0
    let plane = (MATTING_SIZE * MATTING_SIZE) as usize;
    let mut input = vec![0f32; 3 * plane];
    for (i, pixel) in resized.pixels().enumerate() {
        input[i] = pixel[0] as f32 / 255.0 - 0.5;
        input[plane + i] = pixel[1] as f32 / 255.0 - 0.5;
        input[2 * plane + i] = pixel[2] as f32 / 255.0 - 0.5;
    }

    let tensor = Tensor::from_array((
        vec![1usize, 3, MATTING_SIZE as usize, MATTING_SIZE as usize],
        input,
    ))
    .map_err(error::inference_failed)?;

    progress(45, "segment");
    let (_shape, data) = run_single(session, tensor)?;

    if data.len() < plane {
        return Err(error::inference_failed("mask smaller than expected"));
    }

    progress(80, "compose");

    // The model does not apply a sigmoid: normalise min-max over the whole map,
    // as the reference implementation does.
    let mask = &data[..plane];
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &v in mask {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    let span = if (hi - lo).abs() < f32::EPSILON { 1.0 } else { hi - lo };

    let mut mask_image = image::GrayImage::new(MATTING_SIZE, MATTING_SIZE);
    for (i, pixel) in mask_image.pixels_mut().enumerate() {
        let normalized = ((mask[i] - lo) / span).clamp(0.0, 1.0);
        pixel[0] = (normalized * 255.0).round() as u8;
    }

    // Scale the mask back to the original resolution so the image loses no
    // detail by passing through the model's 1024 px.
    let mask_full = image::imageops::resize(&mask_image, width, height, FilterType::Triangle);

    let mut output = source.clone();
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let alpha = mask_full.get_pixel(x, y)[0];
        // Combine with the existing alpha so already-transparent areas are not
        // brought back.
        pixel[3] = ((pixel[3] as u16 * alpha as u16) / 255) as u8;
    }

    progress(100, "done");
    Ok(output)
}

// ---------------------------------------------------------------------------
// Super-resolution (Swin2SR)
// ---------------------------------------------------------------------------

/// Side of the tiles processed individually. Swin2SR uses windowed attention:
/// on a whole image the memory cost grows too quickly.
const TILE: u32 = 192;
/// Extra context around each tile, discarded afterwards: prevents visible seams
/// between adjacent tiles.
const OVERLAP: u32 = 16;
/// Swin2SR requires sides that are multiples of its window size.
const WINDOW: u32 = 8;

pub fn upscale(
    session: &mut Session,
    source: &RgbaImage,
    scale: u32,
    progress: ProgressFn<'_>,
) -> NabuResult<RgbaImage> {
    let (width, height) = source.dimensions();
    let mut output = RgbaImage::new(width * scale, height * scale);

    let cols = width.div_ceil(TILE);
    let rows = height.div_ceil(TILE);
    let total = (cols * rows).max(1);
    let mut done = 0u32;

    for row in 0..rows {
        for col in 0..cols {
            // Destination region, without overlap.
            let x0 = col * TILE;
            let y0 = row * TILE;
            let w = TILE.min(width - x0);
            let h = TILE.min(height - y0);

            // Read region, widened by the context and clamped to the image.
            let cx0 = x0.saturating_sub(OVERLAP);
            let cy0 = y0.saturating_sub(OVERLAP);
            let cx1 = (x0 + w + OVERLAP).min(width);
            let cy1 = (y0 + h + OVERLAP).min(height);

            let tile = image::imageops::crop_imm(source, cx0, cy0, cx1 - cx0, cy1 - cy0).to_image();
            let sr = upscale_tile(session, &tile, scale)?;

            // In the enlarged tile, skip the context and copy the centre.
            let skip_x = (x0 - cx0) * scale;
            let skip_y = (y0 - cy0) * scale;

            for dy in 0..(h * scale) {
                for dx in 0..(w * scale) {
                    let sx = skip_x + dx;
                    let sy = skip_y + dy;
                    if sx >= sr.width() || sy >= sr.height() {
                        continue;
                    }
                    output.put_pixel(x0 * scale + dx, y0 * scale + dy, *sr.get_pixel(sx, sy));
                }
            }

            done += 1;
            progress(((done as f32 / total as f32) * 100.0) as u32, "tile");
        }
    }

    Ok(output)
}

/// Enlarges a single tile. The alpha channel does not go through the model,
/// which is trained on RGB, and is resampled separately.
fn upscale_tile(session: &mut Session, tile: &RgbaImage, scale: u32) -> NabuResult<RgbaImage> {
    let (w, h) = tile.dimensions();

    // The model needs sides that are multiples of WINDOW: extend by replicating
    // the border, so the padding introduces no dark edge.
    let pw = w.next_multiple_of(WINDOW);
    let ph = h.next_multiple_of(WINDOW);

    let plane = (pw * ph) as usize;
    let mut input = vec![0f32; 3 * plane];
    for y in 0..ph {
        for x in 0..pw {
            let src = tile.get_pixel(x.min(w - 1), y.min(h - 1));
            let i = (y * pw + x) as usize;
            input[i] = src[0] as f32 / 255.0;
            input[plane + i] = src[1] as f32 / 255.0;
            input[2 * plane + i] = src[2] as f32 / 255.0;
        }
    }

    let tensor = Tensor::from_array((vec![1usize, 3, ph as usize, pw as usize], input))
        .map_err(error::inference_failed)?;

    let (shape, data) = run_single(session, tensor)?;

    // Read the output resolution from the tensor rather than assuming it.
    let (oh, ow) = match shape.len() {
        4 => (shape[2], shape[3]),
        3 => (shape[1], shape[2]),
        _ => return Err(error::inference_failed("unexpected upscaling output shape")),
    };
    let out_plane = ow * oh;
    if data.len() < 3 * out_plane {
        return Err(error::inference_failed("incomplete upscaling output"));
    }

    let mut upscaled = RgbaImage::new(ow as u32, oh as u32);
    for y in 0..oh {
        for x in 0..ow {
            let i = y * ow + x;
            let channel =
                |c: usize| ((data[c * out_plane + i]).clamp(0.0, 1.0) * 255.0).round() as u8;
            upscaled.put_pixel(
                x as u32,
                y as u32,
                Rgba([channel(0), channel(1), channel(2), 255]),
            );
        }
    }

    // Crop away the padding and carry over the enlarged original alpha.
    let mut result = image::imageops::crop_imm(&upscaled, 0, 0, w * scale, h * scale).to_image();
    let alpha = image::imageops::resize(tile, w * scale, h * scale, FilterType::Triangle);
    for (x, y, pixel) in result.enumerate_pixels_mut() {
        pixel[3] = alpha.get_pixel(x, y)[3];
    }

    Ok(result)
}
