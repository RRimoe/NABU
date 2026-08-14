// src-tauri/src/ai/vision/depth.rs
//! The depth capability: image in, one relative distance per pixel out.
//!
//! Models in this family predict *relative inverse* depth: larger means nearer,
//! and the scale is arbitrary — it carries no metres and cannot be compared
//! between two photographs. So the only honest thing to hand back is a map
//! normalised against this picture's own near and far ends, which is exactly
//! what the editor needs: "how far is this pixel compared to the rest of the
//! frame" answers "blur the background" and "where is the ground".

use image::{GrayImage, Luma, RgbaImage};
use image::imageops::FilterType;
use ort::session::Session;

use crate::ai::error::{self, NabuResult};
use crate::ai::tasks::ProgressFn;

use super::manifest::{ModelManifest, OutputSpec};
use super::preprocess;

/// A depth map at the source image's resolution, 0 = farthest, 255 = nearest.
pub struct DepthMap {
    pub image: GrayImage,
}

/// Runs the depth model described by the manifest.
pub fn run(
    session: &mut Session,
    manifest: &ModelManifest,
    source: &RgbaImage,
    progress: ProgressFn<'_>,
) -> NabuResult<DepthMap> {
    let OutputSpec::Dpt {
        output: output_name,
        inverse,
    } = &manifest.output
    else {
        return Err(error::inference_failed(format!(
            "manifest {} is not of the dpt family",
            manifest.id
        )));
    };

    progress(20, "prepare");
    let (tensor, _) = preprocess::prepare(source, &manifest.input)?;

    progress(45, "depth");
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(error::inference_failed)?;

    // The tensor is looked up by name when the manifest gives one, and
    // otherwise taken as "the only output there is". Export tools rename freely
    // between versions, and refusing to run because a name drifted would be a
    // poor trade for a network that has exactly one thing to say.
    let mut found: Option<(Vec<usize>, Vec<f32>)> = None;
    for (name, value) in outputs.iter() {
        if !output_name.is_empty() && name != output_name {
            continue;
        }
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(error::inference_failed)?;
        found = Some((shape.iter().map(|d| *d as usize).collect(), data.to_vec()));
        if !output_name.is_empty() {
            break;
        }
    }

    let (shape, data) = found.ok_or_else(|| {
        error::inference_failed(if output_name.is_empty() {
            "the depth model produced no output".to_string()
        } else {
            format!("missing output tensor: {output_name}")
        })
    })?;

    progress(80, "compose");
    let (width, height) = predicted_size(&shape)?;
    let normalised = normalise(&data, width * height, *inverse)?;

    // Back to the source resolution. Triangle rather than nearest: a depth map
    // is a smooth field, and blocky edges in it become blocky edges in the blur
    // that reads it.
    let small = GrayImage::from_raw(width as u32, height as u32, normalised)
        .ok_or_else(|| error::inference_failed("depth map did not fill its buffer"))?;
    let (src_w, src_h) = source.dimensions();
    let image = image::imageops::resize(&small, src_w, src_h, FilterType::Triangle);

    Ok(DepthMap { image })
}

/// Reads the spatial size out of whatever shape the export happened to use.
///
/// `[1, H, W]` is the common one; `[1, 1, H, W]` and a bare `[H, W]` show up
/// often enough in the wild to be worth accepting rather than rejecting.
fn predicted_size(shape: &[usize]) -> NabuResult<(usize, usize)> {
    match shape {
        [1, h, w] => Ok((*w, *h)),
        [1, 1, h, w] => Ok((*w, *h)),
        [h, w] => Ok((*w, *h)),
        other => Err(error::inference_failed(format!(
            "unexpected depth output shape: {other:?}"
        ))),
    }
}

/// Scales raw predictions into 0..255 with 255 meaning nearest.
///
/// A flat prediction — a photograph of a wall, or a model that failed to say
/// anything — would divide by zero, so it comes back as a uniform mid grey:
/// "everything is the same distance" is the correct answer there, and it makes
/// every downstream effect a no-op instead of a crash.
fn normalise(data: &[f32], pixels: usize, inverse: bool) -> NabuResult<Vec<u8>> {
    if data.len() < pixels {
        return Err(error::inference_failed("depth output truncated"));
    }

    let slice = &data[..pixels];
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for value in slice {
        if !value.is_finite() {
            continue;
        }
        min = min.min(*value);
        max = max.max(*value);
    }

    if !min.is_finite() || !max.is_finite() || (max - min).abs() < 1e-9 {
        return Ok(vec![128u8; pixels]);
    }

    let span = max - min;
    Ok(slice
        .iter()
        .map(|value| {
            let normalised = if value.is_finite() {
                (value - min) / span
            } else {
                0.0
            };
            // Inverse depth already has near = large; a model predicting true
            // distance has to be flipped so the contract stays one thing.
            let near = if inverse { normalised } else { 1.0 - normalised };
            (near.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect())
}

/// Mean depth inside a box, 0..1 with 1 nearest. Used to tell a subject from
/// its background without asking the user which is which.
pub fn mean_depth(map: &GrayImage, x0: u32, y0: u32, x1: u32, y1: u32) -> f32 {
    let (w, h) = map.dimensions();
    let x1 = x1.min(w);
    let y1 = y1.min(h);
    if x0 >= x1 || y0 >= y1 {
        return 0.0;
    }

    let mut sum = 0u64;
    let mut count = 0u64;
    for y in y0..y1 {
        for x in x0..x1 {
            let Luma([v]) = *map.get_pixel(x, y);
            sum += v as u64;
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum as f32 / count as f32 / 255.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_spreads_the_range_over_the_full_scale() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let out = normalise(&data, 4, true).expect("normalised");
        assert_eq!(out, vec![0, 85, 170, 255]);
    }

    #[test]
    fn true_distance_models_are_flipped_so_near_is_always_bright() {
        let data = vec![2.0, 8.0];
        let out = normalise(&data, 2, false).expect("normalised");
        assert_eq!(out, vec![255, 0], "the closest sample must end up brightest");
    }

    #[test]
    fn a_flat_prediction_becomes_uniform_grey_rather_than_a_division_by_zero() {
        let data = vec![3.0; 16];
        let out = normalise(&data, 16, true).expect("normalised");
        assert!(out.iter().all(|v| *v == 128));
    }

    #[test]
    fn non_finite_values_do_not_poison_the_range() {
        let data = vec![f32::NAN, 0.0, 10.0, f32::INFINITY];
        let out = normalise(&data, 4, true).expect("normalised");
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 255);
    }

    #[test]
    fn truncated_output_is_an_error_not_a_panic() {
        assert!(normalise(&[1.0, 2.0], 16, true).is_err());
    }

    #[test]
    fn the_common_output_shapes_are_all_accepted() {
        assert_eq!(predicted_size(&[1, 518, 392]).expect("chw"), (392, 518));
        assert_eq!(predicted_size(&[1, 1, 64, 32]).expect("nchw"), (32, 64));
        assert_eq!(predicted_size(&[8, 4]).expect("hw"), (4, 8));
        assert!(predicted_size(&[1, 3, 64, 64, 2]).is_err());
    }

    #[test]
    fn mean_depth_reads_the_box_it_was_given() {
        let mut map = GrayImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                map.put_pixel(x, y, Luma([if x < 2 { 255 } else { 0 }]));
            }
        }
        assert!((mean_depth(&map, 0, 0, 2, 4) - 1.0).abs() < 1e-4);
        assert!((mean_depth(&map, 2, 0, 4, 4) - 0.0).abs() < 1e-4);
        assert!((mean_depth(&map, 0, 0, 4, 4) - 0.5).abs() < 1e-3);
        // A degenerate box is answered, not panicked over.
        assert_eq!(mean_depth(&map, 3, 3, 1, 1), 0.0);
    }
}
