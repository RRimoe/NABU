// src-tauri/src/ai/vision/preprocess.rs
//! Image-to-tensor preparation, driven entirely by the manifest.
//!
//! Every vision model wants roughly the same thing — a resized, normalised
//! NCHW float tensor — and differs only in the numbers. Keeping this one
//! parametric implementation is what makes a manifest sufficient to describe a
//! model: there is no per-model preprocessing code to write.

use image::RgbaImage;
use image::imageops::FilterType;
use ort::value::Tensor;

use crate::ai::error::{self, NabuResult};

use super::manifest::{InputSpec, ResizePolicy};

/// Grey used to fill the letterbox padding, the convention detection models
/// are trained with.
const PAD_VALUE: f32 = 114.0;

/// Maps coordinates normalised to the model input back onto source pixels.
///
/// The mapping depends on how the image was resized: a stretched input spans
/// the whole tensor, a letterboxed one only its top-left region. Capturing it
/// here keeps the decoder family code geometry-agnostic.
#[derive(Debug, Clone, Copy)]
pub struct CoordMap {
    /// Multipliers from normalised model space to source pixels.
    fx: f32,
    fy: f32,
    src_w: f32,
    src_h: f32,
}

impl CoordMap {
    /// Converts a normalised (0..1) model-space point to source pixels,
    /// clamped to the image so padding never produces out-of-range boxes.
    pub fn to_source(&self, nx: f32, ny: f32) -> (f32, f32) {
        (
            (nx * self.fx).clamp(0.0, self.src_w),
            (ny * self.fy).clamp(0.0, self.src_h),
        )
    }
}

/// Resizes and normalises the image into NCHW data as the manifest asks.
///
/// Split from `prepare` so the arithmetic can be tested without an ONNX
/// runtime in the loop.
fn prepare_data(source: &RgbaImage, spec: &InputSpec) -> (Vec<usize>, Vec<f32>, CoordMap) {
    let (src_w, src_h) = source.dimensions();
    let (w, h) = (spec.width, spec.height);

    let (resized, map) = match spec.resize {
        ResizePolicy::Stretch => {
            let resized = image::imageops::resize(source, w, h, FilterType::Triangle);
            let map = CoordMap {
                fx: src_w as f32,
                fy: src_h as f32,
                src_w: src_w as f32,
                src_h: src_h as f32,
            };
            (resized, map)
        }
        ResizePolicy::Letterbox => {
            let scale = (w as f32 / src_w as f32).min(h as f32 / src_h as f32);
            let rw = ((src_w as f32 * scale).round() as u32).clamp(1, w);
            let rh = ((src_h as f32 * scale).round() as u32).clamp(1, h);
            let resized = image::imageops::resize(source, rw, rh, FilterType::Triangle);
            let map = CoordMap {
                // Normalised coordinates cover the padded square, so the way
                // back divides the padding out again.
                fx: w as f32 / scale,
                fy: h as f32 / scale,
                src_w: src_w as f32,
                src_h: src_h as f32,
            };
            (resized, map)
        }
    };

    let plane = (w * h) as usize;
    let mut data = vec![0f32; 3 * plane];

    // Fill with the padding value first: letterboxing leaves part of the
    // tensor untouched by real pixels.
    for c in 0..3 {
        let padded = (PAD_VALUE * spec.scale - spec.mean[c]) / spec.std[c];
        data[c * plane..(c + 1) * plane].fill(padded);
    }

    for (x, y, px) in resized.enumerate_pixels() {
        if x >= w || y >= h {
            continue;
        }
        let i = (y * w + x) as usize;
        for c in 0..3 {
            data[c * plane + i] = (px[c] as f32 * spec.scale - spec.mean[c]) / spec.std[c];
        }
    }

    (vec![1, 3, h as usize, w as usize], data, map)
}

/// Builds the input tensor and the coordinate map for one inference run.
pub fn prepare(source: &RgbaImage, spec: &InputSpec) -> NabuResult<(Tensor<f32>, CoordMap)> {
    let (shape, data, map) = prepare_data(source, spec);
    let tensor = Tensor::from_array((shape, data)).map_err(error::inference_failed)?;
    Ok((tensor, map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn spec(resize: ResizePolicy) -> InputSpec {
        InputSpec {
            width: 64,
            height: 64,
            resize,
            scale: 1.0 / 255.0,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn stretch_maps_normalised_corners_to_the_image_corners() {
        let source = RgbaImage::new(200, 100);
        let (_, _, map) = prepare_data(&source, &spec(ResizePolicy::Stretch));

        assert_eq!(map.to_source(0.0, 0.0), (0.0, 0.0));
        assert_eq!(map.to_source(1.0, 1.0), (200.0, 100.0));
        assert_eq!(map.to_source(0.5, 0.5), (100.0, 50.0));
    }

    #[test]
    fn letterbox_maps_the_content_region_not_the_padding() {
        // 200x100 into a 64x64 letterbox: scale 0.32, content fills the full
        // width but only the top 32 rows.
        let source = RgbaImage::new(200, 100);
        let (_, _, map) = prepare_data(&source, &spec(ResizePolicy::Letterbox));

        // The content's right edge sits at nx = 1.0, its bottom at ny = 0.5.
        assert_eq!(map.to_source(1.0, 0.5), (200.0, 100.0));
        // Beyond the content the mapping clamps instead of inventing pixels.
        assert_eq!(map.to_source(1.0, 1.0), (200.0, 100.0));
    }

    #[test]
    fn normalisation_follows_the_manifest_numbers() {
        let mut source = RgbaImage::new(2, 2);
        for px in source.pixels_mut() {
            *px = Rgba([255, 0, 128, 255]);
        }

        let mut spec = spec(ResizePolicy::Stretch);
        spec.width = 2;
        spec.height = 2;
        spec.mean = [0.5, 0.5, 0.5];
        spec.std = [0.5, 0.5, 0.5];

        let (shape, data, _) = prepare_data(&source, &spec);

        assert_eq!(shape, vec![1, 3, 2, 2]);
        // Channel 0: (1.0 - 0.5) / 0.5 = 1.0
        assert!((data[0] - 1.0).abs() < 1e-5);
        // Channel 1: (0.0 - 0.5) / 0.5 = -1.0
        assert!((data[4] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn letterbox_padding_area_holds_the_pad_value() {
        // A wide image letterboxed into a square: the bottom rows are padding.
        let mut source = RgbaImage::new(200, 100);
        for px in source.pixels_mut() {
            *px = Rgba([255, 255, 255, 255]);
        }

        let (_, data, _) = prepare_data(&source, &spec(ResizePolicy::Letterbox));

        let plane = 64 * 64;
        let expected_pad = 114.0 / 255.0;
        // Last row of channel 0 is well below the content region (32 rows).
        let last_row = &data[plane - 64..plane];
        for v in last_row {
            assert!((v - expected_pad).abs() < 1e-5, "padding not applied: {v}");
        }
    }
}
