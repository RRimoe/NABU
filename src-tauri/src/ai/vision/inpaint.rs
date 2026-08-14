// src-tauri/src/ai/vision/inpaint.rs
//! The inpainter capability: image plus mask in, rebuilt image out.
//!
//! This is the second half of object removal. A segmenter answers "which
//! pixels are the object"; this module hands those pixels to a
//! mask-directed generative model (the LaMa family) and composites the
//! reconstruction back into the photo at full resolution.

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
const MASK_THRESHOLD: u8 = 128;

/// How far the mask grows beyond the object, as a fraction of the working
/// resolution. Inpainting models reconstruct from what surrounds the hole:
/// a mask that hugs the object too tightly leaves its halo behind.
const DILATE_FRACTION: f32 = 0.02;

/// How much surrounding photo the model is shown, as a fraction of the hole's
/// longest side. The reconstruction is invented from what borders it, so a
/// window cropped tight to the object gives it almost nothing to go on; too
/// wide and the detail budget is spent on pixels that were never in question.
const CONTEXT_FRACTION: f32 = 0.6;

/// Smallest margin around the hole, in source pixels, for a tiny object.
const MIN_CONTEXT: u32 = 48;

/// The part of the source the model actually sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
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

    // Into the model's square. The window is already square, so this is a
    // change of scale and nothing else.
    let small = image::imageops::resize(&cropped, w, h, FilterType::Triangle);
    let small_mask = image::imageops::resize(&cropped_mask, w, h, FilterType::Triangle);
    let radius = ((w.min(h) as f32 * DILATE_FRACTION).round() as u32).max(2);
    let dilated = dilate(&binarise(&small_mask), radius);

    let plane = (w * h) as usize;
    let mut image_data = vec![0f32; 3 * plane];
    let scale = manifest.input.scale;
    for (i, px) in small.pixels().enumerate() {
        for c in 0..3 {
            image_data[c * plane + i] =
                (px[c] as f32 * scale - manifest.input.mean[c]) / manifest.input.std[c];
        }
    }

    let mut mask_data = vec![0f32; plane];
    for (i, px) in dilated.pixels().enumerate() {
        mask_data[i] = if px[0] > 0 { 1.0 } else { 0.0 };
    }

    let image_tensor = Tensor::from_array((vec![1usize, 3, h as usize, w as usize], image_data))
        .map_err(error::inference_failed)?;
    let mask_tensor = Tensor::from_array((vec![1usize, 1, h as usize, w as usize], mask_data))
        .map_err(error::inference_failed)?;

    progress(45, "inpaint");

    let outputs = session
        .run(ort::inputs![
            image_input.as_str() => image_tensor,
            mask_input.as_str() => mask_tensor,
        ])
        .map_err(error::inference_failed)?;

    let mut result: Option<(Vec<usize>, Vec<f32>)> = None;
    for (name, value) in outputs.iter() {
        if name == output_name {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(error::inference_failed)?;
            result = Some((shape.iter().map(|d| *d as usize).collect(), data.to_vec()));
        }
    }
    let (shape, data) = result
        .ok_or_else(|| error::inference_failed(format!("missing output tensor: {output_name}")))?;

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

    progress(80, "compose");

    // The family speaks 0..255 at the output; anything else would show up
    // immediately as a black or grey patch in the end-to-end test.
    let mut rebuilt = RgbaImage::new(ow as u32, oh as u32);
    for (i, px) in rebuilt.pixels_mut().enumerate() {
        for c in 0..3 {
            px[c] = data[c * out_plane + i].clamp(0.0, 255.0).round() as u8;
        }
        px[3] = 255;
    }

    // The network answered smoothly, which is the right answer for structure
    // and the wrong one for texture: the average of a thousand pebbles is grey
    // mud. Where the surroundings are textured, its answer is kept as the guide
    // — it knows where the light falls — and the detail is copied from the real
    // photograph a few centimetres away.
    let rebuilt = {
        let small_source = image::imageops::resize(&cropped, w, h, FilterType::Triangle);
        if super::synth::worth_synthesising(&small_source, &dilated) {
            progress(88, "texture");
            super::synth::transfer_detail(&small_source, &rebuilt, &dilated)
        } else {
            rebuilt
        }
    };

    // Back to the window's own resolution, and only the window is touched.
    // Feathering the dilated mask softens the transition band, giving a seamless
    // blending between the inpainting reconstruction and the original photo.
    let rebuilt_window =
        image::imageops::resize(&rebuilt, window.width, window.height, FilterType::Triangle);
    let feathered = feather_mask(&dilated, 3);
    let blend_window =
        image::imageops::resize(&feathered, window.width, window.height, FilterType::Triangle);

    let mut output = source.clone();
    for wy in 0..window.height {
        for wx in 0..window.width {
            let m = blend_window.get_pixel(wx, wy)[0] as u16;
            if m == 0 {
                continue;
            }
            let (x, y) = (window.x + wx, window.y + wy);
            let r = rebuilt_window.get_pixel(wx, wy);
            let px = output.get_pixel_mut(x, y);
            for c in 0..3 {
                px[c] = ((px[c] as u16 * (255 - m) + r[c] as u16 * m) / 255) as u8;
            }
        }
    }

    progress(100, "done");
    Ok(output)
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

/// Grows the white region by `radius`, as two separable sliding-maximum
/// passes: exact for a square structuring element and linear in the pixels.
fn dilate(mask: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = mask.dimensions();
    let r = radius as i64;
    let raw = mask.as_raw();

    let mut horizontal = vec![0u8; (w * h) as usize];
    for y in 0..h {
        let row_offset = (y * w) as usize;
        for x in 0..w {
            let mut best = 0u8;
            for dx in -r..=r {
                let sx = x as i64 + dx;
                if sx >= 0 && sx < w as i64 {
                    let v = raw[row_offset + sx as usize];
                    if v > best {
                        best = v;
                        if best == 255 {
                            break;
                        }
                    }
                }
            }
            horizontal[row_offset + x as usize] = best;
        }
    }

    let mut out = GrayImage::new(w, h);
    let out_raw: &mut [u8] = out.as_mut();
    let stride = w as usize;
    for x in 0..w {
        for y in 0..h {
            let mut best = 0u8;
            for dy in -r..=r {
                let sy = y as i64 + dy;
                if sy >= 0 && sy < h as i64 {
                    let v = horizontal[sy as usize * stride + x as usize];
                    if v > best {
                        best = v;
                        if best == 255 {
                            break;
                        }
                    }
                }
            }
            out_raw[y as usize * stride + x as usize] = best;
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
    fn binarise_applies_the_threshold() {
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(0, 0, image::Luma([MASK_THRESHOLD]));
        mask.put_pixel(1, 0, image::Luma([MASK_THRESHOLD + 1]));

        let hard = binarise(&mask);
        assert_eq!(hard.get_pixel(0, 0)[0], 0);
        assert_eq!(hard.get_pixel(1, 0)[0], 255);
    }
}
