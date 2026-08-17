// src-tauri/src/ai/vision/inpaint.rs
//! The inpainter capability: image plus mask in, rebuilt image out.
//!
//! This is the second half of object removal. A segmenter answers "which
//! pixels are the object"; this module hands those pixels to a
//! mask-directed generative model (the LaMa family) and composites the
//! reconstruction back into the photo at full resolution.

use std::collections::VecDeque;

use image::imageops::FilterType;
use image::{GrayImage, RgbaImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::ai::error::{self, NabuResult};
use crate::ai::tasks::ProgressFn;

use super::manifest::{ModelManifest, OutputSpec};

/// Pixels above this alpha count as "object" when a mask is binarised.
///
/// 128, not less. The segmenter writes its confidence as a ramp centred there:
/// 128 is the point where it stopped saying "background" and started saying
/// "object". Thresholding below it — this was 96 — silently takes in everything
/// the model actually rejected, down to a fair margin of negative confidence,
/// so every removal ate more of the photograph than was asked for and the
/// inpainter was handed a larger hole than it needed to fill.
pub(super) const MASK_THRESHOLD: u8 = 128;

/// Smallest growth, for a mask whose edge was drawn rather than inferred.
const MIN_DILATE: u32 = 2;

/// How far an inferred mask grows, as a fraction of the picture's shorter side.
///
/// Of the *picture*. Not of the context window, which collapses to the whole
/// frame for a large object and made the growth depend on how much of the
/// photograph happened to be in view; and not of the object, whose shape has
/// nothing to do with it. What has to be covered is the halo a thing leaves in
/// a photograph — fringing, the softness of the lens, the bloom of grain — plus
/// a segmenter's habit of stopping just inside a soft edge. Both are measured
/// in pixels of the picture.
const DILATE_OF_FRAME: f32 = 0.02;

/// Never more than this share of the object's *longest* side, so that a small
/// object is not swallowed by its own margin.
///
/// The longest, and that is the whole point. Keyed to the shorter side, a bench
/// two thousand pixels wide and three hundred tall was allowed a third of the
/// growth it needed, and the removal left a halo exactly where the bench had
/// been. A flat object is not a small one.
const MAX_DILATE_OF_OBJECT: f32 = 0.25;

/// Mask values this far from nothing and from certainty count as the soft ramp.
const RAMP_LO: u8 = 24;
const RAMP_HI: u8 = 232;

/// How much surrounding photo the model is shown, as a fraction of the hole's
/// longest side. The reconstruction is invented from what borders it, so a
/// window cropped tight to the object gives it almost nothing to go on; too
/// wide and the detail budget is spent on pixels that were never in question.
const CONTEXT_FRACTION: f32 = 0.6;

/// Smallest margin around the hole, in source pixels, for a tiny object.
const MIN_CONTEXT: u32 = 48;

/// Side of the grid the correspondence search runs on.
///
/// The search is the expensive half of the detail transfer and the half that
/// does not need full resolution: it answers *where* in the photograph a piece
/// of hedge lives, and that is a question about structure. The copying that
/// follows happens at the window's own size, so the detail that lands in the
/// hole is the photograph's, not this grid's.
const SYNTH_SEARCH_SIDE: u32 = 768;

/// Width of the band over which the rebuilt window fades into the photograph,
/// as a fraction of the window's shorter side.
const FEATHER_FRACTION: f32 = 0.006;

/// How far apart native-resolution tiles are placed, as a fraction of a tile.
const TILE_STEP: f32 = 0.5;

/// The thickest hole a tile can still hold with photograph on both sides of it.
///
/// Under half the tile, so that even at its widest the hole leaves a quarter of
/// a tile of real picture above and below — which is what the model rebuilds
/// from. Above this a tile in the middle of the hole would see almost nothing.
const TILE_MAX_HOLE: f32 = 0.45;

/// Below this magnification the reduced window is losing little enough that
/// several passes of the model would not be worth their cost.
const TILE_MIN_SCALE: u32 = 2;

/// The part of the source the model actually sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// How far the mask must grow to cover the halo the object leaves behind.
///
/// The soft ramp in a segmenter's answer is not uncertainty waiting to be
/// covered — the threshold sits exactly on the boundary the model decided, and
/// the ramp is only how it draws that boundary without scissors. So the ramp is
/// not measured, it is merely *noticed*: a mask that has one was inferred, and
/// gets the growth an inferred edge needs; a mask without one was drawn by
/// hand, and what someone drew is what they meant.
fn dilation_radius(mask: &GrayImage, frame_side: u32) -> u32 {
    let (w, h) = mask.dimensions();
    if w == 0 || h == 0 {
        return MIN_DILATE;
    }
    let raw = mask.as_raw();
    let stride = w as usize;
    let mut ramp = 0u64;
    let mut perimeter = 0u64;

    for y in 0..h {
        let row = y as usize * stride;
        for x in 0..w {
            let value = raw[row + x as usize];
            if value > RAMP_LO && value < RAMP_HI {
                ramp += 1;
            }
            if value <= MASK_THRESHOLD {
                continue;
            }
            let outside = (x == 0 || raw[row + x as usize - 1] <= MASK_THRESHOLD)
                || (x + 1 >= w || raw[row + x as usize + 1] <= MASK_THRESHOLD)
                || (y == 0 || raw[row - stride + x as usize] <= MASK_THRESHOLD)
                || (y + 1 >= h || raw[row + stride + x as usize] <= MASK_THRESHOLD);
            if outside {
                perimeter += 1;
            }
        }
    }

    if perimeter == 0 {
        return MIN_DILATE;
    }

    // A ramp narrower than a pixel is not a ramp: this edge was drawn.
    if ramp / perimeter < 1 {
        return MIN_DILATE;
    }

    let cap = match mask_bounds(mask) {
        Some((x0, y0, x1, y1)) => {
            let extent = (x1 - x0 + 1).max(y1 - y0 + 1);
            ((extent as f32 * MAX_DILATE_OF_OBJECT) as u32).max(MIN_DILATE)
        }
        None => MIN_DILATE,
    };

    ((frame_side as f32 * DILATE_OF_FRAME) as u32).clamp(MIN_DILATE, cap)
}

/// Bounding box of everything the mask marks, or `None` for an empty mask.
fn mask_bounds(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = mask.dimensions();
    let mut x0 = w;
    let mut y0 = h;
    let mut x1 = 0u32;
    let mut y1 = 0u32;
    let mut any = false;

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] > MASK_THRESHOLD {
                any = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }

    if any { Some((x0, y0, x1, y1)) } else { None }
}

/// The square of photo handed to the model around a hole.
///
/// Square on purpose. The model's input is square, so a rectangular window
/// would have to be squashed to fit it — and a network asked to imagine a leg
/// in an image stretched sideways imagines a stretched leg. Cropping square
/// means the resize that follows is a pure change of scale.
///
/// Clamped to the picture, which is why it is not always centred on the hole:
/// an object against the left edge simply gets all of its context from the
/// right.
fn context_window(
    bounds: (u32, u32, u32, u32),
    src_w: u32,
    src_h: u32,
) -> Window {
    let (x0, y0, x1, y1) = bounds;
    let box_w = x1.saturating_sub(x0) + 1;
    let box_h = y1.saturating_sub(y0) + 1;

    let margin = ((box_w.max(box_h) as f32 * CONTEXT_FRACTION) as u32).max(MIN_CONTEXT);

    // Grow to a square that holds the hole plus its margin, but never larger
    // than the picture itself.
    let side = (box_w.max(box_h) + margin * 2).min(src_w.min(src_h));

    // Centre it on the hole, then slide it back inside the frame.
    let centre_x = x0 + box_w / 2;
    let centre_y = y0 + box_h / 2;
    let x = centre_x
        .saturating_sub(side / 2)
        .min(src_w.saturating_sub(side));
    let y = centre_y
        .saturating_sub(side / 2)
        .min(src_h.saturating_sub(side));

    Window { x, y, width: side, height: side }
}

/// Removes the masked region from the image, rebuilding it from its
/// surroundings with the inpainter the manifest describes.
///
/// `mask` is an alpha map at source resolution — exactly what the segmenter
/// contract produces. Soft edges are welcome; the mask is binarised, grown
/// and re-feathered here.
pub fn remove_region(
    session: &mut Session,
    manifest: &ModelManifest,
    source: &RgbaImage,
    mask: &GrayImage,
    progress: ProgressFn<'_>,
) -> NabuResult<RgbaImage> {
    let OutputSpec::Lama {
        image_input,
        mask_input,
        output: output_name,
    } = &manifest.output
    else {
        return Err(error::inference_failed(format!(
            "manifest {} is not an inpainter family",
            manifest.id
        )));
    };

    let (src_w, src_h) = source.dimensions();
    let (w, h) = (manifest.input.width, manifest.input.height);

    progress(20, "prepare");

    // Nothing marked: the photo comes back untouched rather than being run
    // through a network that would repaint it for no reason.
    let Some(bounds) = mask_bounds(mask) else {
        progress(100, "done");
        return Ok(source.clone());
    };

    // The model sees a square window around the hole, not the whole photo.
    //
    // This is the difference between a repair and a smear. The network works at
    // a fixed 512 pixels: given a whole 12-megapixel frame it has about a
    // seventh of the linear resolution to invent with, and the result is then
    // magnified back — so the reconstruction arrives blurred by construction,
    // however good the model is. Cropping first spends those 512 pixels on the
    // part of the picture that is actually missing.
    let window = context_window(bounds, src_w, src_h);
    let cropped = image::imageops::crop_imm(source, window.x, window.y, window.width, window.height)
        .to_image();
    let cropped_mask =
        image::imageops::crop_imm(mask, window.x, window.y, window.width, window.height).to_image();

    // The mask is hardened and grown at the window's own resolution, and only
    // reduced later, by whichever path needs it — and then by taking a
    // maximum, never an average.
    //
    // The order matters far more than it looks. Shrinking a soft alpha map
    // first and thresholding afterwards is, in effect, a filter that deletes
    // thin things: a mast, a handlebar, a spoke covers a fraction of a pixel
    // once the window is down at 512, averages to a value well under the
    // threshold, and drops out of the mask altogether. The object then comes
    // back in the finished photograph as a dark splinter the removal never
    // touched — and worse, as a patch of "known" picture that the detail
    // transfer is free to copy somewhere else entirely.
    //
    // The growth is measured against the picture's own size, not the window's:
    // the window is a working detail, and the halo being covered belongs to
    // the photograph.
    let grown = dilate(
        &binarise(&cropped_mask),
        dilation_radius(&cropped_mask, src_w.min(src_h)),
    );

    let names = Names {
        image: image_input.as_str(),
        mask: mask_input.as_str(),
        output: output_name.as_str(),
    };

    // Two ways to rebuild the window, and the difference between them is
    // resolution.
    //
    // The model works at a fixed 512 pixels. Handed a whole window it must be
    // shrunk to fit, so every line it draws is drawn at a fraction of the
    // photograph's resolution and magnified back — which is the softness that
    // survives even a perfect mask. But a hole that is long and thin, which is
    // what a bench or a car or a railing leaves behind, fits inside a tile of
    // 512 *real* pixels with photograph still to spare on either side. Those
    // can be rebuilt at their own size, one tile at a time.
    //
    // A thick hole cannot: a tile placed in the middle of it would see nothing
    // but hole. Those still go through the reduced window.
    let plan = mask_bounds(&grown).and_then(|bounds| tile_origins(bounds, window, w, h));

    let guide = match plan {
        Some(origins) => reconstruct_in_tiles(
            session, manifest, names, &cropped, &grown, &origins, w, progress,
        )?,
        None => {
            progress(45, "inpaintReduced");
            let small = image::imageops::resize(&cropped, w, h, FilterType::Triangle);
            let dilated = downscale_max(&grown, w, h);
            let rebuilt = run_inpainter(session, manifest, names, &small, &dilated)?;

            // Back to the window's own size *before* the detail is put in.
            //
            // The network answered smoothly, which is the right answer for
            // structure and the wrong one for texture: the average of a
            // thousand leaves is green mud, and magnifying that mud sevenfold
            // is what turned a hedge into a smear. But the smoothness only
            // ever concerned the structure — where the light falls, how the
            // ground recedes — and structure magnifies tolerably well. So the
            // enlarged answer is kept as the *guide*, and the detail is put
            // back into it from the photograph at full size.
            image::imageops::resize(&rebuilt, window.width, window.height, FilterType::Triangle)
        }
    };

    progress(80, "compose");

    // The search grid: small enough to be affordable, and reduced with a
    // maximum so a cell it calls known holds no hole pixels at all — which is
    // what lets a match found here be copied at full size without dragging a
    // piece of the removed object along with it.
    let side = window.width.min(SYNTH_SEARCH_SIDE);
    let search_image = image::imageops::resize(&cropped, side, side, FilterType::Triangle);
    let search_mask = downscale_max(&grown, side, side);

    let filled = if super::synth::worth_synthesising(&search_image, &search_mask) {
        progress(88, "texture");
        let search_guide = image::imageops::resize(&guide, side, side, FilterType::Triangle);
        let field = super::synth::correspondence(&search_image, &search_guide, &search_mask);
        super::synth::apply_detail(&cropped, &guide, &grown, &field)
    } else {
        // Nothing textured to copy, which does not mean nothing to copy. Every
        // photograph has grain, and a hole without any of it reads as a soft
        // patch however right its colours are.
        progress(88, "grain");
        super::synth::transfer_grain(&cropped, &guide, &grown)
    };

    // Last, the levels. Everything above decided *what* goes in the hole; this
    // decides that it meets the photograph without a step.
    progress(94, "levels");
    let filled = level_match(&cropped, &filled, &grown);

    // Feathering happens where the picture is, too. A band measured on the
    // model's grid and magnified back arrives several times wider than asked
    // for, and softer than the pixels underneath it — which is its own visible
    // halo around every removal.
    let feather =
        ((window.width.min(window.height) as f32 * FEATHER_FRACTION).round() as u32).max(1);
    let feathered = feather_mask(&grown, feather);

    let mut output = source.clone();
    for wy in 0..window.height {
        for wx in 0..window.width {
            let m = feathered.get_pixel(wx, wy)[0] as u16;
            if m == 0 {
                continue;
            }
            let (x, y) = (window.x + wx, window.y + wy);
            let r = filled.get_pixel(wx, wy);
            let px = output.get_pixel_mut(x, y);
            for c in 0..3 {
                px[c] = ((px[c] as u16 * (255 - m) + r[c] as u16 * m) / 255) as u8;
            }
        }
    }

    progress(100, "done");
    Ok(output)
}

/// Sides of the grids the level correction is solved on, coarsest first.
///
/// A pyramid, because relaxation is very good at removing local error and
/// hopeless at removing a gentle tilt across a wide region — which is exactly
/// the shape of the error here. Solving small first and carrying the answer up
/// fixes the tilt where it is cheap to see.
const LEVEL_SIDES: [u32; 4] = [16, 32, 64, 128];

/// Relaxation sweeps at each level of that pyramid.
const LEVEL_SWEEPS: u32 = 120;

/// Matches the fill's levels to the photograph's, by spreading the mismatch at
/// the seam smoothly across the hole.
///
/// The inpainter answers with the right *content* and, quite often, slightly
/// the wrong brightness: the patch reads as a pale bruise on grass, or a
/// shadow that is a little too light. Fading one into the other with an alpha
/// ramp does not fix that — it averages the two, so the mismatch simply moves
/// into the fade and becomes a soft rim.
///
/// So the seam mismatch is measured where the two meet, and a *membrane* is
/// interpolated across the hole from those edge values and added to the fill.
/// At the boundary the result becomes the photograph exactly; inside, only the
/// low frequencies move, because a harmonic function has nothing else in it.
/// The fill's own detail and grain pass through untouched. This is Poisson
/// blending, arranged so that no large system has to be solved: a membrane has
/// no fine structure, so it is worked out on a small grid and carried up.
fn level_match(source: &RgbaImage, filled: &RgbaImage, hole: &GrayImage) -> RgbaImage {
    let (w, h) = source.dimensions();
    let mut out = filled.clone();
    if w == 0 || h == 0 || filled.dimensions() != (w, h) || hole.dimensions() != (w, h) {
        return out;
    }

    let mut carried: Option<(u32, Vec<f32>)> = None;

    for &side in &LEVEL_SIDES {
        let coarse_source = image::imageops::resize(source, side, side, FilterType::Triangle);
        let coarse_filled = image::imageops::resize(filled, side, side, FilterType::Triangle);
        // A maximum, so a cell the coarse grid calls known holds no hole at
        // all and the mismatch measured there is a real one.
        let coarse_hole = downscale_max(hole, side, side);

        let n = (side * side) as usize;
        let mut field = vec![0f32; n * 3];

        // Start from the level below, where the wide, gentle part of the
        // answer has already been found.
        if let Some((from_side, previous)) = &carried {
            for y in 0..side {
                for x in 0..side {
                    let sx = (x * from_side / side).min(from_side - 1);
                    let sy = (y * from_side / side).min(from_side - 1);
                    let from = (sy * from_side + sx) as usize;
                    let to = (y * side + x) as usize;
                    field[to * 3..to * 3 + 3]
                        .copy_from_slice(&previous[from * 3..from * 3 + 3]);
                }
            }
        }

        // The membrane is held up by the *inner* rim of the hole, not by the
        // photograph outside it.
        //
        // This is the part that is easy to get backwards, and silently: out
        // there the fill and the photograph are the same pixels by
        // construction, so a mismatch measured there is always zero and the
        // membrane comes out flat. The step to be removed happens *across* the
        // boundary — the fill on one side, the photograph on the other — so it
        // has to be measured on the last row of hole and cancelled there.
        let inside: Vec<bool> = (0..n).map(|i| coarse_hole.as_raw()[i] > 0).collect();
        let mut fixed = vec![true; n];

        for i in 0..n {
            let (x, y) = (i as u32 % side, i as u32 / side);
            if !inside[i] {
                for c in 0..3 {
                    field[i * 3 + c] = 0.0;
                }
                continue;
            }

            let neighbours = [
                (x > 0).then(|| i - 1),
                (x + 1 < side).then(|| i + 1),
                (y > 0).then(|| i - side as usize),
                (y + 1 < side).then(|| i + side as usize),
            ];
            let known: Vec<usize> = neighbours
                .into_iter()
                .flatten()
                .filter(|&j| !inside[j])
                .collect();

            if known.is_empty() {
                // Deeper in: solved for, not measured.
                fixed[i] = false;
                continue;
            }

            let f = coarse_filled.get_pixel(x, y);
            for c in 0..3 {
                let wanted: f32 = known
                    .iter()
                    .map(|&j| coarse_source.get_pixel(j as u32 % side, j as u32 / side)[c] as f32)
                    .sum::<f32>()
                    / known.len() as f32;
                field[i * 3 + c] = wanted - f[c] as f32;
            }
        }

        for _ in 0..LEVEL_SWEEPS {
            for y in 0..side {
                for x in 0..side {
                    let i = (y * side + x) as usize;
                    if fixed[i] {
                        continue;
                    }
                    for c in 0..3 {
                        let mut total = 0f32;
                        let mut count = 0f32;
                        if x > 0 {
                            total += field[(i - 1) * 3 + c];
                            count += 1.0;
                        }
                        if x + 1 < side {
                            total += field[(i + 1) * 3 + c];
                            count += 1.0;
                        }
                        if y > 0 {
                            total += field[(i - side as usize) * 3 + c];
                            count += 1.0;
                        }
                        if y + 1 < side {
                            total += field[(i + side as usize) * 3 + c];
                            count += 1.0;
                        }
                        if count > 0.0 {
                            field[i * 3 + c] = total / count;
                        }
                    }
                }
            }
        }

        carried = Some((side, field));
    }

    let Some((side, field)) = carried else { return out };

    // Up to the picture, bilinearly: the membrane is smooth, so nothing is
    // lost on the way.
    let raw = hole.as_raw();
    let stride = w as usize;
    for y in 0..h {
        let fy = (y as f32 + 0.5) * side as f32 / h as f32 - 0.5;
        let y0 = fy.floor().clamp(0.0, side as f32 - 1.0) as u32;
        let y1 = (y0 + 1).min(side - 1);
        let wy = (fy - y0 as f32).clamp(0.0, 1.0);

        for x in 0..w {
            if raw[y as usize * stride + x as usize] == 0 {
                continue;
            }
            let fx = (x as f32 + 0.5) * side as f32 / w as f32 - 0.5;
            let x0 = fx.floor().clamp(0.0, side as f32 - 1.0) as u32;
            let x1 = (x0 + 1).min(side - 1);
            let wx = (fx - x0 as f32).clamp(0.0, 1.0);

            let at = |gx: u32, gy: u32| (gy * side + gx) as usize * 3;
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));

            let pixel = out.get_pixel_mut(x, y);
            for channel in 0..3 {
                let top = field[a + channel] * (1.0 - wx) + field[b + channel] * wx;
                let bottom = field[c + channel] * (1.0 - wx) + field[d + channel] * wx;
                let correction = top * (1.0 - wy) + bottom * wy;
                pixel[channel] =
                    (pixel[channel] as f32 + correction).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

/// The tensor names this family uses, carried together to keep the signatures
/// of the two reconstruction paths honest.
#[derive(Clone, Copy)]
struct Names<'a> {
    image: &'a str,
    mask: &'a str,
    output: &'a str,
}

/// One pass of the inpainter over an image and mask already at its input size.
fn run_inpainter(
    session: &mut Session,
    manifest: &ModelManifest,
    names: Names<'_>,
    image: &RgbaImage,
    mask: &GrayImage,
) -> NabuResult<RgbaImage> {
    let (w, h) = image.dimensions();
    let plane = (w * h) as usize;
    let scale = manifest.input.scale;

    let mut image_data = vec![0f32; 3 * plane];
    for (i, px) in image.pixels().enumerate() {
        for c in 0..3 {
            image_data[c * plane + i] =
                (px[c] as f32 * scale - manifest.input.mean[c]) / manifest.input.std[c];
        }
    }

    let mut mask_data = vec![0f32; plane];
    for (i, px) in mask.pixels().enumerate() {
        mask_data[i] = if px[0] > 0 { 1.0 } else { 0.0 };
    }

    let image_tensor = Tensor::from_array((vec![1usize, 3, h as usize, w as usize], image_data))
        .map_err(error::inference_failed)?;
    let mask_tensor = Tensor::from_array((vec![1usize, 1, h as usize, w as usize], mask_data))
        .map_err(error::inference_failed)?;

    let outputs = session
        .run(ort::inputs![
            names.image => image_tensor,
            names.mask => mask_tensor,
        ])
        .map_err(error::inference_failed)?;

    let mut result: Option<(Vec<usize>, Vec<f32>)> = None;
    for (name, value) in outputs.iter() {
        if name == names.output {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(error::inference_failed)?;
            result = Some((shape.iter().map(|d| *d as usize).collect(), data.to_vec()));
        }
    }
    let (shape, data) = result.ok_or_else(|| {
        error::inference_failed(format!("missing output tensor: {}", names.output))
    })?;

    let (oh, ow) = match shape.as_slice() {
        [1, 3, oh, ow] => (*oh, *ow),
        other => {
            return Err(error::inference_failed(format!(
                "unexpected inpainter output shape: {other:?}"
            )));
        }
    };
    let out_plane = ow * oh;
    if data.len() < 3 * out_plane {
        return Err(error::inference_failed("incomplete inpainter output"));
    }

    // The family speaks 0..255 at the output; anything else would show up
    // immediately as a black or grey patch in the end-to-end test.
    let mut rebuilt = RgbaImage::new(ow as u32, oh as u32);
    for (i, px) in rebuilt.pixels_mut().enumerate() {
        for c in 0..3 {
            px[c] = data[c * out_plane + i].clamp(0.0, 255.0).round() as u8;
        }
        px[3] = 255;
    }
    Ok(rebuilt)
}

/// Where to place native-resolution tiles along a hole, or `None` when the hole
/// is the wrong shape for it.
///
/// The condition is simply whether a tile can hold the hole *across* its
/// narrow direction and still see photograph on both sides. A bench, a car, a
/// railing, a cable: all long and thin, all fine. A crowd, a tree, a parked
/// van seen head-on: not, and those keep the reduced window, which at least
/// sees the whole problem at once.
fn tile_origins(
    bounds: (u32, u32, u32, u32),
    window: Window,
    tile_w: u32,
    tile_h: u32,
) -> Option<Vec<(u32, u32)>> {
    // The model's input has to be square for a tile to be a plain crop of the
    // photograph rather than a squashed one.
    if tile_w != tile_h || tile_w == 0 {
        return None;
    }
    let tile = tile_w;
    if window.width < tile || window.height < tile {
        return None;
    }

    // Below this the reduced window is barely losing anything, and several
    // passes of the model would buy almost no resolution for their cost.
    if window.width < tile * TILE_MIN_SCALE {
        return None;
    }

    let (x0, y0, x1, y1) = bounds;
    let box_w = x1 - x0 + 1;
    let box_h = y1 - y0 + 1;
    if box_w.min(box_h) as f32 > tile as f32 * TILE_MAX_HOLE {
        return None;
    }

    let along_x = box_w >= box_h;
    let (cross_centre, cross_span) = if along_x {
        ((y0 + y1) / 2, window.height)
    } else {
        ((x0 + x1) / 2, window.width)
    };
    let cross = cross_centre
        .saturating_sub(tile / 2)
        .min(cross_span - tile);

    let (from, to, span) = if along_x {
        (x0, x1, window.width)
    } else {
        (y0, y1, window.height)
    };
    let step = ((tile as f32 * TILE_STEP) as u32).max(1);
    let last = span - tile;

    let mut at = from.saturating_sub(tile / 4).min(last);
    let mut origins = Vec::new();
    loop {
        origins.push(if along_x { (at, cross) } else { (cross, at) });
        if at >= last || at + tile > to {
            break;
        }
        at = (at + step).min(last);
    }
    Some(origins)
}

/// Rebuilds the hole from overlapping tiles taken at the photograph's own
/// resolution.
///
/// Each tile is a plain crop — no shrinking, so nothing is invented at a
/// coarser scale than the picture it belongs to. They overlap by half, and
/// where they meet the answers are averaged with a weight that falls off
/// towards each tile's edge, so a line crossing the join arrives continuous
/// instead of stepped.
#[allow(clippy::too_many_arguments)]
fn reconstruct_in_tiles(
    session: &mut Session,
    manifest: &ModelManifest,
    names: Names<'_>,
    cropped: &RgbaImage,
    hole: &GrayImage,
    origins: &[(u32, u32)],
    tile: u32,
    progress: ProgressFn<'_>,
) -> NabuResult<RgbaImage> {
    let mut guide = cropped.clone();

    // Only the band the tiles actually cover needs accumulating; for a long
    // thin hole that is a fraction of the window, and the window can be nine
    // megapixels.
    let (mut bx0, mut by0, mut bx1, mut by1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for &(ox, oy) in origins {
        bx0 = bx0.min(ox);
        by0 = by0.min(oy);
        bx1 = bx1.max(ox + tile);
        by1 = by1.max(oy + tile);
    }
    let band_w = (bx1 - bx0) as usize;
    let band_h = (by1 - by0) as usize;

    let mut sum = vec![0f32; band_w * band_h * 3];
    let mut weight = vec![0f32; band_w * band_h];

    // The falloff is half a step wide, so by the time one tile has faded out
    // the next is fully in.
    let ramp = ((tile as f32 * TILE_STEP) / 2.0).max(1.0);

    for (index, &(ox, oy)) in origins.iter().enumerate() {
        let mask_tile = image::imageops::crop_imm(hole, ox, oy, tile, tile).to_image();
        if mask_tile.pixels().all(|px| px[0] == 0) {
            continue;
        }

        let image_tile = image::imageops::crop_imm(cropped, ox, oy, tile, tile).to_image();
        let filled = run_inpainter(session, manifest, names, &image_tile, &mask_tile)?;

        for ty in 0..tile {
            for tx in 0..tile {
                let edge_x = tx.min(tile - 1 - tx) as f32;
                let edge_y = ty.min(tile - 1 - ty) as f32;
                // A hair above zero, so a pixel only one tile reaches is still
                // that tile's answer rather than a division by nothing.
                let w = (edge_x / ramp).min(1.0) * (edge_y / ramp).min(1.0) + 0.001;

                let bx = (ox + tx - bx0) as usize;
                let by = (oy + ty - by0) as usize;
                let at = by * band_w + bx;
                let px = filled.get_pixel(tx, ty);
                for c in 0..3 {
                    sum[at * 3 + c] += px[c] as f32 * w;
                }
                weight[at] += w;
            }
        }

        // A stage of its own, so that which of the two reconstructions ran is
        // visible from the outside. Without it there is no way to tell a
        // native-resolution rebuild from a magnified one except by squinting
        // at the result.
        progress(
            45 + (35 * (index as u32 + 1) / origins.len().max(1) as u32),
            "inpaintTiles",
        );
    }

    for by in 0..band_h {
        for bx in 0..band_w {
            let at = by * band_w + bx;
            if weight[at] <= 0.0 {
                continue;
            }
            let px = guide.get_pixel_mut(bx0 + bx as u32, by0 + by as u32);
            for c in 0..3 {
                px[c] = (sum[at * 3 + c] / weight[at]).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(guide)
}

/// The region [`remove_region`] will actually cut out for this mask: hardened
/// at the segmenter's own threshold, then grown by the same margin the removal
/// uses.
///
/// Exposed because two other things need to know it — a preview that shows
/// what is about to disappear, and the mask report that measures whether the
/// segmenter is the limiting factor — and because working it out a second time
/// somewhere else is how the answer and the act drift apart.
pub fn effective_hole(mask: &GrayImage) -> GrayImage {
    let (w, h) = mask.dimensions();
    if mask_bounds(mask).is_none() {
        return GrayImage::new(w, h);
    }
    dilate(&binarise(mask), dilation_radius(mask, w.min(h)))
}

/// Hard 0/255 mask from a soft alpha map.
fn binarise(mask: &GrayImage) -> GrayImage {
    let mut out = mask.clone();
    for px in out.pixels_mut() {
        px[0] = if px[0] > MASK_THRESHOLD { 255 } else { 0 };
    }
    out
}

/// Softens the binary mask edges for seamless alpha blending.
fn feather_mask(mask: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = mask.dimensions();
    if w == 0 || h == 0 || radius == 0 {
        return mask.clone();
    }
    let r = radius as i32;
    let span = (r * 2 + 1) as f32;
    let inv_span = 1.0 / span;
    let raw = mask.as_raw();

    let mut horizontal = vec![0u8; (w * h) as usize];
    for y in 0..h {
        let row_offset = (y * w) as usize;
        let mut acc = 0f32;
        for dx in -r..=r {
            let sx = dx.clamp(0, w as i32 - 1) as usize;
            acc += raw[row_offset + sx] as f32;
        }
        for x in 0..w {
            horizontal[row_offset + x as usize] = (acc * inv_span).round().clamp(0.0, 255.0) as u8;
            let left_x = (x as i32 - r).clamp(0, w as i32 - 1) as usize;
            let right_x = (x as i32 + r + 1).clamp(0, w as i32 - 1) as usize;
            acc += raw[row_offset + right_x] as f32 - raw[row_offset + left_x] as f32;
        }
    }

    let mut out = GrayImage::new(w, h);
    let out_raw: &mut [u8] = out.as_mut();
    for x in 0..w {
        let stride = w as usize;
        let mut acc = 0f32;
        for dy in -r..=r {
            let sy = dy.clamp(0, h as i32 - 1) as usize;
            acc += horizontal[sy * stride + x as usize] as f32;
        }
        for y in 0..h {
            let out_idx = y as usize * stride + x as usize;
            out_raw[out_idx] = (acc * inv_span).round().clamp(0.0, 255.0) as u8;
            let top_y = (y as i32 - r).clamp(0, h as i32 - 1) as usize;
            let bottom_y = (y as i32 + r + 1).clamp(0, h as i32 - 1) as usize;
            acc += horizontal[bottom_y * stride + x as usize] as f32 - horizontal[top_y * stride + x as usize] as f32;
        }
    }
    out
}

/// Maximum of `src` over the window `[i - r, i + r]`, clamped to the ends,
/// written into `dst`.
///
/// `deque` holds candidate indices in decreasing value order, so each element
/// enters and leaves it exactly once and the cost per pixel does not depend on
/// the radius. It is passed in only to be reused between rows and columns.
fn sliding_max(src: &[u8], r: usize, dst: &mut [u8], deque: &mut VecDeque<usize>) {
    let n = src.len();
    deque.clear();
    let mut next = 0usize;

    for (i, out) in dst[..n].iter_mut().enumerate() {
        // Take in everything the window has newly reached, dropping anything
        // it is now impossible to return: a value with a bigger one after it
        // can never be the maximum again.
        let last = (i + r).min(n - 1);
        while next <= last {
            while let Some(&back) = deque.back() {
                if src[back] > src[next] {
                    break;
                }
                deque.pop_back();
            }
            deque.push_back(next);
            next += 1;
        }

        let first = i.saturating_sub(r);
        while let Some(&front) = deque.front() {
            if front >= first {
                break;
            }
            deque.pop_front();
        }

        *out = src[deque[0]];
    }
}

/// Grows the white region by `radius`, as two separable sliding-maximum
/// passes: exact for a square structuring element and linear in the pixels,
/// whatever the radius.
///
/// The radius is not small any more. This runs on the mask at the window's own
/// resolution, where 2% of a three-thousand-pixel side is sixty pixels and the
/// straightforward scan it would take is a hundred and twenty reads per pixel.
pub(super) fn dilate(mask: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = mask.dimensions();
    if w == 0 || h == 0 || radius == 0 {
        return mask.clone();
    }

    let r = radius as usize;
    let stride = w as usize;
    let rows = h as usize;
    let src = mask.as_raw();
    let mut deque = VecDeque::with_capacity(stride.max(rows));

    let mut horizontal = vec![0u8; stride * rows];
    for (y, out_row) in horizontal.chunks_mut(stride).enumerate() {
        let row = y * stride;
        sliding_max(&src[row..row + stride], r, out_row, &mut deque);
    }

    let mut out = GrayImage::new(w, h);
    let out_raw: &mut [u8] = out.as_mut();
    let mut column = vec![0u8; rows];
    let mut grown = vec![0u8; rows];
    for x in 0..stride {
        for y in 0..rows {
            column[y] = horizontal[y * stride + x];
        }
        sliding_max(&column, r, &mut grown, &mut deque);
        for y in 0..rows {
            out_raw[y * stride + x] = grown[y];
        }
    }
    out
}

/// Reduces a mask to `dst_w` × `dst_h` by taking the maximum over each source
/// cell rather than its average.
///
/// A mask is not a photograph, and resampling it as though it were is how an
/// object survives its own removal: three marked pixels inside an eight-pixel
/// cell average out to ninety-five, fall under any sensible threshold, and the
/// spoke they belonged to is quietly left in the picture. A maximum keeps
/// whatever was marked marked, at every scale — the mask can round outwards,
/// never inwards.
fn downscale_max(mask: &GrayImage, dst_w: u32, dst_h: u32) -> GrayImage {
    let (src_w, src_h) = mask.dimensions();
    if (src_w, src_h) == (dst_w, dst_h) {
        return mask.clone();
    }

    let mut out = GrayImage::new(dst_w, dst_h);
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }

    let raw = mask.as_raw();
    // In u64, because a cell index times a source dimension overflows u32 on
    // any photograph worth the name.
    let span = |i: u32, dst: u32, src: u32| {
        let lo = (i as u64 * src as u64 / dst as u64) as u32;
        // At least one pixel wide, so a destination bigger than the source
        // still samples something rather than an empty range.
        let hi = (((i as u64 + 1) * src as u64 / dst as u64) as u32).max(lo + 1).min(src);
        (lo, hi)
    };

    for y in 0..dst_h {
        let (y0, y1) = span(y, dst_h, src_h);
        for x in 0..dst_w {
            let (x0, x1) = span(x, dst_w, src_w);
            let mut best = 0u8;
            'cell: for sy in y0..y1 {
                let row = sy as usize * src_w as usize;
                for sx in x0..x1 {
                    let v = raw[row + sx as usize];
                    if v > best {
                        best = v;
                        if best == 255 {
                            break 'cell;
                        }
                    }
                }
            }
            out.put_pixel(x, y, image::Luma([best]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilation_grows_a_point_into_a_square() {
        let mut mask = GrayImage::new(9, 9);
        mask.put_pixel(4, 4, image::Luma([255]));

        let grown = dilate(&mask, 2);

        assert_eq!(grown.get_pixel(2, 2)[0], 255, "corner of the square");
        assert_eq!(grown.get_pixel(6, 6)[0], 255);
        assert_eq!(grown.get_pixel(4, 1)[0], 0, "outside the radius");
        assert_eq!(grown.get_pixel(0, 0)[0], 0);
    }

    /// A mask with one white rectangle, for the window tests.
    fn mask_with(x0: u32, y0: u32, x1: u32, y1: u32, w: u32, h: u32) -> GrayImage {
        let mut mask = GrayImage::new(w, h);
        for y in y0..=y1 {
            for x in x0..=x1 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        mask
    }

    #[test]
    fn mask_bounds_finds_the_marked_rectangle() {
        let mask = mask_with(10, 20, 30, 50, 100, 100);
        assert_eq!(mask_bounds(&mask), Some((10, 20, 30, 50)));
    }

    #[test]
    fn an_empty_mask_has_no_bounds() {
        assert_eq!(mask_bounds(&GrayImage::new(16, 16)), None);
    }

    #[test]
    fn faint_marks_below_the_threshold_do_not_count() {
        let mut mask = GrayImage::new(8, 8);
        mask.put_pixel(4, 4, image::Luma([MASK_THRESHOLD]));
        assert_eq!(mask_bounds(&mask), None);
    }

    #[test]
    fn the_window_is_square_and_holds_the_hole_with_room_around_it() {
        // A tall thin hole in a large frame: 100 wide, 400 tall.
        let window = context_window((450, 300, 549, 699), 2000, 1500);

        assert_eq!(window.width, window.height, "the window must be square");
        // 400 tall plus 60% margin on each side.
        assert_eq!(window.width, 400 + (400 * 6 / 10) * 2);
        // It contains the hole.
        assert!(window.x <= 450 && window.y <= 300);
        assert!(window.x + window.width >= 550);
        assert!(window.y + window.height >= 700);
    }

    #[test]
    fn a_tiny_hole_still_gets_a_usable_amount_of_context() {
        let window = context_window((100, 100, 103, 103), 800, 800);
        assert_eq!(window.width, 4 + MIN_CONTEXT * 2);
    }

    #[test]
    fn a_hole_against_the_edge_slides_the_window_inside_the_frame() {
        // Hard against the top-left corner, which is where the legs in the
        // reported photo were: the window cannot centre, so it must shift.
        let window = context_window((0, 0, 60, 400), 1200, 1600);

        assert_eq!(window.x, 0);
        assert_eq!(window.y, 0);
        assert!(window.x + window.width <= 1200);
        assert!(window.y + window.height <= 1600);
    }

    #[test]
    fn a_hole_covering_the_frame_gives_a_window_no_larger_than_the_picture() {
        let window = context_window((0, 0, 999, 799), 1000, 800);
        assert_eq!(window.width, 800, "capped by the shorter side");
        assert!(window.x + window.width <= 1000);
        assert!(window.y + window.height <= 800);
    }

    #[test]
    fn the_window_never_leaves_the_picture_wherever_the_hole_is() {
        // The clamping is the part that would silently produce a panic in
        // `crop_imm`, so it is checked from every corner and the middle.
        for (x0, y0) in [(0, 0), (930, 0), (0, 730), (930, 730), (480, 380)] {
            let window = context_window((x0, y0, x0 + 60, y0 + 60), 1000, 800);
            assert!(
                window.x + window.width <= 1000 && window.y + window.height <= 800,
                "window {window:?} escaped the picture from ({x0}, {y0})"
            );
        }
    }

    #[test]
    fn dilation_with_a_radius_past_the_edges_fills_everything() {
        let mut mask = GrayImage::new(6, 4);
        mask.put_pixel(0, 3, image::Luma([255]));

        let grown = dilate(&mask, 99);
        assert!(grown.pixels().all(|p| p[0] == 255));
    }

    #[test]
    fn dilation_by_nothing_leaves_the_mask_alone() {
        let mask = mask_with(2, 2, 4, 4, 8, 8);
        assert_eq!(dilate(&mask, 0).as_raw(), mask.as_raw());
    }

    #[test]
    fn a_thin_mark_survives_the_reduction_that_averaging_would_erase() {
        // One column of object in a 512-wide window on its way to the model's
        // grid: the mast, the handlebar, the spoke.
        let mut mask = GrayImage::new(512, 64);
        for y in 0..64 {
            mask.put_pixel(200, y, image::Luma([255]));
        }

        // What the pipeline used to do, and why the object came back.
        let averaged = image::imageops::resize(&mask, 64, 8, FilterType::Triangle);
        let peak = averaged.pixels().map(|p| p[0]).max().unwrap();
        assert!(
            peak < MASK_THRESHOLD,
            "averaging must lose the mark — that is the bug being fixed, got {peak}"
        );

        let reduced = downscale_max(&mask, 64, 8);
        assert!(
            reduced.pixels().any(|p| p[0] == 255),
            "the maximum must carry it through"
        );
        // And it lands where it belongs: 200/512 of the way across.
        assert_eq!(reduced.get_pixel(200 * 64 / 512, 0)[0], 255);
    }

    #[test]
    fn the_reduced_mask_covers_everything_the_full_one_did() {
        // A diagonal hairline, the shape averaging is worst at.
        let mut mask = GrayImage::new(256, 256);
        for i in 0..256 {
            mask.put_pixel(i, i, image::Luma([255]));
        }

        let reduced = downscale_max(&mask, 32, 32);
        for i in 0..256u32 {
            assert_eq!(
                reduced.get_pixel(i / 8, i / 8)[0],
                255,
                "source pixel ({i}, {i}) lost its cell"
            );
        }
    }

    #[test]
    fn the_effective_hole_is_the_mask_hardened_and_grown() {
        let mut mask = GrayImage::new(400, 400);
        // A faint mark that the threshold rejects, and a solid one it keeps.
        mask.put_pixel(20, 20, image::Luma([100]));
        for y in 190..210 {
            for x in 190..210 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }

        let hole = effective_hole(&mask);

        assert_eq!(hole.get_pixel(200, 200)[0], 255, "the marked square is in");
        assert_eq!(hole.get_pixel(20, 20)[0], 0, "a mark under the threshold is not");
        assert_eq!(hole.get_pixel(188, 200)[0], 255, "a crisp edge grows by the minimum");
        assert_eq!(hole.get_pixel(185, 200)[0], 0, "and asks for no more than that");
    }

    /// A rectangle with a soft edge, the shape a segmenter returns.
    fn soft_rectangle(w: u32, h: u32, half_w: i32, half_h: i32) -> GrayImage {
        let mut mask = GrayImage::new(w, h);
        let (cx, cy) = (w as i32 / 2, h as i32 / 2);
        for y in 0..h {
            for x in 0..w {
                // Distance outside the rectangle, in pixels.
                let dx = ((x as i32 - cx).abs() - half_w).max(0);
                let dy = ((y as i32 - cy).abs() - half_h).max(0);
                let d = dx.max(dy);
                let value = match d {
                    0 => 255,
                    1..=19 => (255 - d * 255 / 20) as u8,
                    _ => 0,
                };
                mask.put_pixel(x, y, image::Luma([value]));
            }
        }
        mask
    }

    #[test]
    fn a_flat_object_grows_as_much_as_a_tall_one() {
        // The regression this exists to prevent. Growth used to be capped by
        // the object's *shorter* side, so a bench — wide and low — was allowed
        // a fraction of what it needed and the removal left a halo exactly
        // where the bench had been. Nothing about being flat makes an object's
        // halo smaller.
        let frame = 1000u32;
        let tall = effective_hole(&soft_rectangle(frame, frame, 150, 150));
        let flat = effective_hole(&soft_rectangle(frame, frame, 400, 60));

        let reach_of = |hole: &GrayImage, half: i32| {
            let mut reach = 0;
            for step in 0..90 {
                if hole.get_pixel((500 + half + step) as u32, 500)[0] > 128 {
                    reach = step;
                }
            }
            reach
        };

        let tall_reach = reach_of(&tall, 150);
        let flat_reach = reach_of(&flat, 400);
        assert_eq!(
            tall_reach, flat_reach,
            "a flat object must grow exactly as much as a tall one"
        );
        assert!(tall_reach >= 25, "and by a real margin, got {tall_reach}");
    }

    /// The window a hole of this shape would get, for the tiling tests.
    fn window_of(width: u32, height: u32) -> Window {
        Window { x: 0, y: 0, width, height }
    }

    #[test]
    fn a_long_thin_hole_is_rebuilt_in_tiles_along_its_length() {
        // A bench: wide, shallow, in a window far larger than one tile.
        let origins = tile_origins((300, 1400, 2600, 1560), window_of(3024, 3024), 512, 512)
            .expect("a thin hole must be tileable");

        assert!(origins.len() > 1, "one tile cannot cover 2300 pixels of bench");
        // Stepping along x, and all at the same height: the hole runs that way.
        assert!(origins.windows(2).all(|pair| pair[0].1 == pair[1].1));
        assert!(origins.windows(2).all(|pair| pair[0].0 < pair[1].0));

        // Every tile stays inside the window...
        for &(x, y) in &origins {
            assert!(x + 512 <= 3024 && y + 512 <= 3024, "tile ({x}, {y}) left the window");
        }
        // ...they overlap...
        assert!(
            origins.windows(2).all(|pair| pair[1].0 - pair[0].0 < 512),
            "tiles must overlap or a seam is guaranteed"
        );
        // ...they cover the hole from end to end...
        assert!(origins[0].0 <= 300, "the near end is not covered");
        assert!(origins.last().unwrap().0 + 512 >= 2600, "the far end is not covered");
        // ...and the hole sits inside them across its narrow direction.
        let top = origins[0].1;
        assert!(top <= 1400 && top + 512 >= 1560, "the hole is not inside the tile");
    }

    #[test]
    fn a_thick_hole_is_not_tiled() {
        // A tile in the middle of this would see nothing but hole, so there is
        // nothing to rebuild from and the reduced window is the better answer.
        assert!(tile_origins((500, 500, 1500, 1500), window_of(3024, 3024), 512, 512).is_none());
    }

    #[test]
    fn a_window_barely_larger_than_a_tile_is_not_tiled() {
        // Shrinking this window loses almost nothing, and several passes of the
        // model would buy that almost-nothing at a high price.
        assert!(tile_origins((100, 400, 800, 500), window_of(900, 900), 512, 512).is_none());
    }

    #[test]
    fn a_tall_thin_hole_is_tiled_down_its_length() {
        let origins = tile_origins((1400, 300, 1560, 2600), window_of(3024, 3024), 512, 512)
            .expect("a vertical pole is just as tileable");

        assert!(origins.windows(2).all(|pair| pair[0].0 == pair[1].0), "same column");
        assert!(origins.windows(2).all(|pair| pair[0].1 < pair[1].1), "stepping down");
        assert!(origins.last().unwrap().1 + 512 >= 2600);
    }

    #[test]
    fn a_hole_against_the_window_edge_keeps_its_tiles_inside_it() {
        let origins = tile_origins((0, 0, 2900, 120), window_of(3024, 3024), 512, 512)
            .expect("tileable");
        for &(x, y) in &origins {
            assert!(x + 512 <= 3024 && y + 512 <= 3024, "tile ({x}, {y}) escaped");
        }
    }

    #[test]
    fn a_model_whose_input_is_not_square_is_never_tiled() {
        // A tile has to be a plain crop of the photograph, not a squashed one.
        assert!(tile_origins((300, 1400, 2600, 1560), window_of(3024, 3024), 512, 256).is_none());
    }

    #[test]
    fn a_fill_that_is_the_wrong_brightness_is_brought_back_to_the_photograph() {
        // The pale bruise on the grass: right content, wrong level.
        let source = RgbaImage::from_pixel(256, 256, image::Rgba([120, 130, 110, 255]));
        let mut filled = source.clone();
        for y in 96..160 {
            for x in 96..160 {
                filled.put_pixel(x, y, image::Rgba([150, 160, 140, 255]));
            }
        }
        let hole = mask_with(96, 96, 159, 159, 256, 256);

        let matched = level_match(&source, &filled, &hole);

        let before = (filled.get_pixel(128, 128)[0] as i32 - 120).abs();
        let after = (matched.get_pixel(128, 128)[0] as i32 - 120).abs();
        assert!(before >= 25, "the test fill really is off by a lot");
        assert!(
            after <= 4,
            "the mismatch had to be carried across the hole, got {after} from {before}"
        );
    }

    #[test]
    fn level_matching_moves_the_level_without_flattening_the_detail() {
        // A fill that is both too bright *and* textured. The brightness must
        // move; the texture must survive, because a membrane has no fine
        // structure to impose.
        let source = RgbaImage::from_pixel(256, 256, image::Rgba([100, 100, 100, 255]));
        let mut filled = source.clone();
        for y in 96..160u32 {
            for x in 96..160u32 {
                let v = if (x / 2 + y / 2) % 2 == 0 { 170 } else { 130 };
                filled.put_pixel(x, y, image::Rgba([v, v, v, 255]));
            }
        }
        let hole = mask_with(96, 96, 159, 159, 256, 256);

        let matched = level_match(&source, &filled, &hole);

        let spread = |img: &RgbaImage| {
            let mut lo = 255i32;
            let mut hi = 0i32;
            for y in 110..146u32 {
                for x in 110..146u32 {
                    let v = img.get_pixel(x, y)[0] as i32;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            (hi - lo, (lo + hi) / 2)
        };

        let (spread_before, mid_before) = spread(&filled);
        let (spread_after, mid_after) = spread(&matched);

        assert!(
            (mid_after - 100).abs() < (mid_before - 100).abs(),
            "the level must move towards the photograph: {mid_before} -> {mid_after}"
        );
        assert!(
            spread_after >= spread_before - 4,
            "the texture must survive: {spread_before} -> {spread_after}"
        );
    }

    #[test]
    fn level_matching_leaves_a_fill_that_already_agrees_alone() {
        let source = RgbaImage::from_pixel(128, 128, image::Rgba([90, 100, 110, 255]));
        let hole = mask_with(40, 40, 88, 88, 128, 128);

        let matched = level_match(&source, &source, &hole);
        for (a, b) in matched.pixels().zip(source.pixels()) {
            for c in 0..3 {
                assert!((a[c] as i32 - b[c] as i32).abs() <= 1, "a perfect fill was disturbed");
            }
        }
    }

    #[test]
    fn a_mask_that_was_drawn_rather_than_inferred_is_barely_grown() {
        // No soft ramp means no segmenter: somebody drew this, and what they
        // drew is what they meant.
        let mask = mask_with(200, 200, 400, 400, 1000, 1000);
        let hole = effective_hole(&mask);

        assert_eq!(hole.get_pixel(198, 300)[0], 255, "grown by the minimum");
        assert_eq!(hole.get_pixel(190, 300)[0], 0, "and no further");
    }

    #[test]
    fn an_unmarked_mask_has_no_effective_hole() {
        assert!(effective_hole(&GrayImage::new(32, 32))
            .pixels()
            .all(|p| p[0] == 0));
    }

    #[test]
    fn reduction_to_the_same_size_changes_nothing() {
        let mask = mask_with(3, 3, 9, 9, 16, 16);
        assert_eq!(downscale_max(&mask, 16, 16).as_raw(), mask.as_raw());
    }

    #[test]
    fn reduction_never_invents_a_mark_where_there_was_none() {
        let mask = GrayImage::new(64, 64);
        assert!(downscale_max(&mask, 8, 8).pixels().all(|p| p[0] == 0));
    }

    #[test]
    fn binarise_applies_the_threshold() {
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(0, 0, image::Luma([MASK_THRESHOLD]));
        mask.put_pixel(1, 0, image::Luma([MASK_THRESHOLD + 1]));

        let hard = binarise(&mask);
        assert_eq!(hard.get_pixel(0, 0)[0], 0);
        assert_eq!(hard.get_pixel(1, 0)[0], 255);
    }
}
