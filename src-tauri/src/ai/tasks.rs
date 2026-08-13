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
    fixed_input: Option<u32>,
    progress: ProgressFn<'_>,
) -> NabuResult<RgbaImage> {
    let (width, height) = source.dimensions();
    let mut output = RgbaImage::new(width * scale, height * scale);

    // Some exports fix their input resolution instead of taking any size.
    // The destination tile shrinks so that tile plus context fits it exactly,
    // and the tensor is padded up to it at the edges of the image.
    let tile = match fixed_input {
        Some(f) => f.saturating_sub(2 * OVERLAP).max(WINDOW),
        None => TILE,
    };

    let cols = width.div_ceil(tile);
    let rows = height.div_ceil(tile);
    let total = (cols * rows).max(1);
    let mut done = 0u32;

    for row in 0..rows {
        for col in 0..cols {
            // Destination region, without overlap.
            let x0 = col * tile;
            let y0 = row * tile;
            let w = tile.min(width - x0);
            let h = tile.min(height - y0);

            // Read region, widened by the context and clamped to the image.
            let cx0 = x0.saturating_sub(OVERLAP);
            let cy0 = y0.saturating_sub(OVERLAP);
            let cx1 = (x0 + w + OVERLAP).min(width);
            let cy1 = (y0 + h + OVERLAP).min(height);

            let patch = image::imageops::crop_imm(source, cx0, cy0, cx1 - cx0, cy1 - cy0).to_image();
            let sr = upscale_tile(session, &patch, scale, fixed_input)?;

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
fn upscale_tile(
    session: &mut Session,
    tile: &RgbaImage,
    scale: u32,
    fixed_input: Option<u32>,
) -> NabuResult<RgbaImage> {
    let (w, h) = tile.dimensions();

    // A fixed-input model wants exactly its resolution; the others need sides
    // that are multiples of WINDOW. Either way the tile is extended by
    // replicating the border, so the padding introduces no dark edge.
    let (pw, ph) = match fixed_input {
        Some(f) => (f, f),
        None => (w.next_multiple_of(WINDOW), h.next_multiple_of(WINDOW)),
    };

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

// ---------------------------------------------------------------------------
// Object segmentation (SlimSAM)
// ---------------------------------------------------------------------------

/// Side of the square input the vision encoder expects.
const SAM_SIZE: u32 = 1024;
/// Side of the low-resolution mask the decoder returns.
const SAM_MASK_SIZE: usize = 256;

/// Per-channel normalisation used when SAM was trained (values on a 0..255 scale).
const SAM_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const SAM_STD: [f32; 3] = [58.395, 57.12, 57.375];

/// A rectangle in source-image pixels, used to tell SAM which object is meant.
#[derive(Debug, Clone, Copy)]
pub struct BoxPrompt {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

/// Runs the two SlimSAM stages and returns the source image with its alpha
/// channel replaced by the mask of the object inside `prompt`.
///
/// Unlike background removal, this does not ask "what is the subject of this
/// picture" — it asks "what is the object here", which is the difference
/// between cutting out a car and cutting out the shape someone drew.
pub fn segment_object(
    encoder: &mut Session,
    decoder: &mut Session,
    source: &RgbaImage,
    prompt: BoxPrompt,
    progress: ProgressFn<'_>,
) -> NabuResult<RgbaImage> {
    let (width, height) = source.dimensions();

    progress(15, "prepare");

    // SAM keeps the aspect ratio: the longest side becomes 1024 and the rest
    // of the square is padded, so the prompt must be scaled the same way.
    let scale = SAM_SIZE as f32 / width.max(height) as f32;
    let resized_w = ((width as f32 * scale).round() as u32).clamp(1, SAM_SIZE);
    let resized_h = ((height as f32 * scale).round() as u32).clamp(1, SAM_SIZE);

    let resized = image::imageops::resize(source, resized_w, resized_h, FilterType::Triangle);

    let plane = (SAM_SIZE * SAM_SIZE) as usize;
    let mut pixels = vec![0f32; 3 * plane];
    for y in 0..resized_h {
        for x in 0..resized_w {
            let px = resized.get_pixel(x, y);
            let i = (y * SAM_SIZE + x) as usize;
            for c in 0..3 {
                pixels[c * plane + i] = (px[c] as f32 - SAM_MEAN[c]) / SAM_STD[c];
            }
        }
    }

    let pixel_values = Tensor::from_array((
        vec![1usize, 3, SAM_SIZE as usize, SAM_SIZE as usize],
        pixels,
    ))
    .map_err(error::inference_failed)?;

    progress(40, "segment");

    // --- Stage 1: encode the image ----------------------------------------
    let embeddings = {
        let outputs = encoder
            .run(ort::inputs![pixel_values])
            .map_err(error::inference_failed)?;

        let mut image_embeddings: Option<Vec<f32>> = None;
        let mut positional: Option<Vec<f32>> = None;

        for (name, value) in outputs.iter() {
            let (_, data) = value
                .try_extract_tensor::<f32>()
                .map_err(error::inference_failed)?;
            match name {
                "image_embeddings" => image_embeddings = Some(data.to_vec()),
                "image_positional_embeddings" => positional = Some(data.to_vec()),
                _ => {}
            }
        }

        let image_embeddings = image_embeddings
            .ok_or_else(|| error::inference_failed("encoder produced no image embeddings"))?;
        let positional = positional
            .ok_or_else(|| error::inference_failed("encoder produced no positional embeddings"))?;
        (image_embeddings, positional)
    };

    let embed_shape = vec![1usize, 256, 64, 64];
    let image_embeddings = Tensor::from_array((embed_shape.clone(), embeddings.0))
        .map_err(error::inference_failed)?;
    let positional_embeddings =
        Tensor::from_array((embed_shape, embeddings.1)).map_err(error::inference_failed)?;

    // --- Stage 2: decode a mask from positive points -----------------------
    //
    // Corner labels 2/3 (the "box" prompt) are not honoured by this export: it
    // answers with the inverse mask. Positive points, label 1, are the prompt
    // every SAM build understands, so the selection is expressed as a handful
    // of points inside it rather than as a rectangle.
    let cx = (prompt.x0 + prompt.x1) / 2.0;
    let cy = (prompt.y0 + prompt.y1) / 2.0;
    let qx = (prompt.x1 - prompt.x0) / 4.0;
    let qy = (prompt.y1 - prompt.y0) / 4.0;

    // Centre plus four points half-way to the edges: one click can land on a
    // window or a wheel and describe only that part, while several spread over
    // the region push SAM towards the whole object.
    let samples: [(f32, f32); 5] = [
        (cx, cy),
        (cx - qx, cy),
        (cx + qx, cy),
        (cx, cy - qy),
        (cx, cy + qy),
    ];

    let mut points = Vec::with_capacity(samples.len() * 2);
    for (x, y) in samples {
        points.push(x * scale);
        points.push(y * scale);
    }
    let labels = vec![1i64; samples.len()];

    let input_points = Tensor::from_array((vec![1usize, 1, samples.len(), 2], points))
        .map_err(error::inference_failed)?;
    let input_labels = Tensor::from_array((vec![1usize, 1, samples.len()], labels))
        .map_err(error::inference_failed)?;

    progress(70, "compose");

    let outputs = decoder
        .run(ort::inputs![
            "input_points" => input_points,
            "input_labels" => input_labels,
            "image_embeddings" => image_embeddings,
            "image_positional_embeddings" => positional_embeddings,
        ])
        .map_err(error::inference_failed)?;

    let mut scores: Vec<f32> = Vec::new();
    let mut masks: Vec<f32> = Vec::new();
    for (name, value) in outputs.iter() {
        let (_, data) = value
            .try_extract_tensor::<f32>()
            .map_err(error::inference_failed)?;
        match name {
            "iou_scores" => scores = data.to_vec(),
            "pred_masks" => masks = data.to_vec(),
            _ => {}
        }
    }

    let mask_plane = SAM_MASK_SIZE * SAM_MASK_SIZE;
    if masks.len() < mask_plane {
        return Err(error::inference_failed("decoder produced no usable mask"));
    }

    // The decoder proposes three masks at different scopes (part, sub-object,
    // whole object): take the one it rates highest rather than always the first.
    let candidates = (masks.len() / mask_plane).min(scores.len().max(1));
    let best = (0..candidates)
        .max_by(|a, b| {
            let sa = scores.get(*a).copied().unwrap_or(0.0);
            let sb = scores.get(*b).copied().unwrap_or(0.0);
            sa.total_cmp(&sb)
        })
        .unwrap_or(0);

    let logits = &masks[best * mask_plane..(best + 1) * mask_plane];

    // The low-resolution mask covers the padded square, so only the part that
    // held the image is meaningful.
    let valid_w = ((resized_w as f32 / SAM_SIZE as f32) * SAM_MASK_SIZE as f32).round() as u32;
    let valid_h = ((resized_h as f32 / SAM_SIZE as f32) * SAM_MASK_SIZE as f32).round() as u32;

    let mut small = image::GrayImage::new(SAM_MASK_SIZE as u32, SAM_MASK_SIZE as u32);
    for (i, pixel) in small.pixels_mut().enumerate() {
        // A logit above zero means "inside the object"; a soft ramp around it
        // keeps the edge from looking cut with scissors.
        let v = logits[i];
        pixel[0] = ((v / 4.0 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
    }

    let cropped =
        image::imageops::crop_imm(&small, 0, 0, valid_w.max(1), valid_h.max(1)).to_image();
    let mask = image::imageops::resize(&cropped, width, height, FilterType::Triangle);

    let mut output = source.clone();
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let alpha = mask.get_pixel(x, y)[0];
        pixel[3] = ((pixel[3] as u16 * alpha as u16) / 255) as u8;
    }

    progress(100, "done");
    Ok(output)
}
