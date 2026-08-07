// src-tauri/src/ai/vision/scene.rs
//! The scene model: one shared representation of what is in the image.
//!
//! Detectors, and later depth or face specialists, do not answer the interface
//! directly; they contribute to this structure. The language router receives a
//! compact rendering of it, which is why a small text model can resolve "the
//! red car behind me" without ever seeing a pixel: by the time the request
//! arrives, seeing has already happened.
//!
//! Everything here is derived — ids, positions, colour names, exposure — and
//! deliberately cheap: the fast pass must not make opening a photo feel slower.

use image::RgbaImage;
use serde::Serialize;

use super::detect::Detection;

/// One object in the scene, ready for the router and the interface.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneObject {
    /// Stable id inside one scene, what `select_object` refers to.
    pub id: u32,
    pub class: String,
    pub confidence: f32,
    /// x0, y0, x1, y1 in pixels of the analysed layer.
    pub bbox: [f32; 4],
    /// Fraction of the image the box covers, 0..1.
    pub area_fraction: f32,
    /// Horizontal third the object's centre falls into: left, center, right.
    pub position: String,
    /// dominant / secondary / minor, by covered area.
    pub prominence: String,
    /// Everyday colour name, when one colour clearly dominates the box.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// Global measurements that need no model at all.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneStats {
    pub width: u32,
    pub height: u32,
    /// Mean luminance, 0..1.
    pub mean_luma: f32,
    /// Fraction of pixels crushed to near-black.
    pub clipped_shadows: f32,
    /// Fraction of pixels blown to near-white.
    pub clipped_highlights: f32,
    /// Rough sharpness score, 0..1; low values suggest blur.
    pub sharpness: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneModel {
    /// Which detector produced the objects, for the activity log.
    pub model_id: String,
    pub objects: Vec<SceneObject>,
    pub stats: SceneStats,
}

/// Assembles the scene from raw detections.
pub fn build_scene(source: &RgbaImage, detections: Vec<Detection>, model_id: &str) -> SceneModel {
    let (width, height) = source.dimensions();
    let image_area = (width as f32 * height as f32).max(1.0);

    let objects = detections
        .into_iter()
        .enumerate()
        .map(|(index, det)| {
            let [x0, y0, x1, y1] = det.bbox;
            let area_fraction = ((x1 - x0) * (y1 - y0) / image_area).clamp(0.0, 1.0);
            let center_x = (x0 + x1) / 2.0;

            let position = if center_x < width as f32 / 3.0 {
                "left"
            } else if center_x > width as f32 * 2.0 / 3.0 {
                "right"
            } else {
                "center"
            };

            let prominence = if area_fraction >= 0.25 {
                "dominant"
            } else if area_fraction >= 0.05 {
                "secondary"
            } else {
                "minor"
            };

            SceneObject {
                id: index as u32 + 1,
                color: dominant_color_name(source, det.bbox),
                class: det.label,
                confidence: det.confidence,
                bbox: det.bbox,
                area_fraction,
                position: position.to_string(),
                prominence: prominence.to_string(),
            }
        })
        .collect();

    SceneModel {
        model_id: model_id.to_string(),
        objects,
        stats: image_stats(source),
    }
}

/// Measures exposure and sharpness by looking at the pixels, no model involved.
pub fn image_stats(source: &RgbaImage) -> SceneStats {
    let (width, height) = source.dimensions();

    // Sample on a stride: statistics on ~64k pixels are indistinguishable from
    // statistics on 40 million, and the fast pass has a latency budget.
    let stride = (((width as u64 * height as u64) / 65_536) as f64).sqrt().ceil() as u32;
    let stride = stride.max(1);

    let mut count = 0u64;
    let mut luma_sum = 0f64;
    let mut shadows = 0u64;
    let mut highlights = 0u64;

    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let px = source.get_pixel(x, y);
            let luma = luma_of(px[0], px[1], px[2]);
            luma_sum += luma as f64;
            if luma < 0.02 {
                shadows += 1;
            } else if luma > 0.98 {
                highlights += 1;
            }
            count += 1;
            x += stride;
        }
        y += stride;
    }
    let count = count.max(1);

    SceneStats {
        width,
        height,
        mean_luma: (luma_sum / count as f64) as f32,
        clipped_shadows: shadows as f32 / count as f32,
        clipped_highlights: highlights as f32 / count as f32,
        sharpness: sharpness_of(source),
    }
}

/// Sharpness as saturated Laplacian variance on a thumbnail.
///
/// The thumbnail keeps the cost flat regardless of resolution; the saturation
/// turns an open-ended variance into a stable 0..1 score. The constant was
/// picked so that crisp photos land above 0.5 and soft ones below.
fn sharpness_of(source: &RgbaImage) -> f32 {
    const SIDE: u32 = 256;
    const HALF_POINT: f32 = 150.0;

    let (w, h) = source.dimensions();
    let scale = SIDE as f32 / w.max(h).max(1) as f32;
    let thumb = if scale < 1.0 {
        image::imageops::resize(
            source,
            ((w as f32 * scale) as u32).max(3),
            ((h as f32 * scale) as u32).max(3),
            image::imageops::FilterType::Triangle,
        )
    } else {
        source.clone()
    };

    let (tw, th) = thumb.dimensions();
    let gray: Vec<f32> = thumb
        .pixels()
        .map(|px| luma_of(px[0], px[1], px[2]) * 255.0)
        .collect();

    let mut sum = 0f64;
    let mut sum_sq = 0f64;
    let mut count = 0u64;

    for y in 1..th - 1 {
        for x in 1..tw - 1 {
            let i = (y * tw + x) as usize;
            let lap = 4.0 * gray[i]
                - gray[i - 1]
                - gray[i + 1]
                - gray[i - tw as usize]
                - gray[i + tw as usize];
            sum += lap as f64;
            sum_sq += (lap * lap) as f64;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64 - mean * mean).max(0.0) as f32;
    variance / (variance + HALF_POINT)
}

fn luma_of(r: u8, g: u8, b: u8) -> f32 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

/// Names the dominant colour inside a box, or nothing when no colour dominates.
///
/// Colour names are the language the user reaches for ("the red car"), so this
/// buckets hues coarsely on purpose: burgundy, crimson and scarlet are all
/// "red" in a request. English names, because the scene is written for the
/// language model, which handles the user's language itself.
pub fn dominant_color_name(source: &RgbaImage, bbox: [f32; 4]) -> Option<String> {
    let (width, height) = source.dimensions();

    // Sample the central region only: the edges of a detection box are mostly
    // background, and background is exactly what must not name the object.
    let [x0, y0, x1, y1] = bbox;
    let (bw, bh) = (x1 - x0, y1 - y0);
    let cx0 = (x0 + bw * 0.2).clamp(0.0, width as f32 - 1.0) as u32;
    let cy0 = (y0 + bh * 0.2).clamp(0.0, height as f32 - 1.0) as u32;
    let cx1 = (x1 - bw * 0.2).clamp(cx0 as f32 + 1.0, width as f32) as u32;
    let cy1 = (y1 - bh * 0.2).clamp(cy0 as f32 + 1.0, height as f32) as u32;

    let stride = ((((cx1 - cx0) as u64 * (cy1 - cy0) as u64) / 4_096) as f64)
        .sqrt()
        .ceil()
        .max(1.0) as u32;

    const NAMES: &[&str] = &[
        "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown", "black",
        "white", "gray",
    ];
    let mut counts = [0u32; 12];
    let mut total = 0u32;

    let mut y = cy0;
    while y < cy1 {
        let mut x = cx0;
        while x < cx1 {
            let px = source.get_pixel(x, y);
            if px[3] > 32 {
                counts[color_bucket(px[0], px[1], px[2])] += 1;
                total += 1;
            }
            x += stride;
        }
        y += stride;
    }
    if total < 8 {
        return None;
    }

    let (best, &count) = counts.iter().enumerate().max_by_key(|(_, c)| **c)?;

    // "Dominant" means an outright majority: a box that is half red and half
    // blue has no colour worth putting in the object's description.
    if count * 100 / total < 55 {
        return None;
    }
    Some(NAMES[best].to_string())
}

/// Maps one RGB pixel to a coarse colour bucket, index into the name table.
fn color_bucket(r: u8, g: u8, b: u8) -> usize {
    let (rf, gf, bf) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    // Achromatic pixels first: value decides between black, white and gray.
    let saturation = if max <= 0.0 { 0.0 } else { delta / max };
    if saturation < 0.22 || delta < 0.05 {
        return if max < 0.18 {
            9 // black
        } else if max > 0.85 {
            10 // white
        } else {
            11 // gray
        };
    }

    let hue = if max == rf {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if max == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    let hue = if hue < 0.0 { hue + 360.0 } else { hue };

    match hue {
        h if h < 15.0 => 0,                            // red
        h if h < 45.0 => {
            // Dark orange reads as brown to the eye.
            if max < 0.55 { 8 } else { 1 }
        }
        h if h < 70.0 => 2,                            // yellow
        h if h < 165.0 => 3,                           // green
        h if h < 200.0 => 4,                           // cyan
        h if h < 260.0 => 5,                           // blue
        h if h < 300.0 => 6,                           // purple
        h if h < 345.0 => 7,                           // pink
        _ => 0,                                        // red wraps around
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> RgbaImage {
        let mut img = RgbaImage::new(width, height);
        for px in img.pixels_mut() {
            *px = Rgba(rgba);
        }
        img
    }

    #[test]
    fn names_a_clearly_red_region() {
        let img = solid(100, 100, [210, 30, 25, 255]);
        assert_eq!(
            dominant_color_name(&img, [10.0, 10.0, 90.0, 90.0]),
            Some("red".to_string())
        );
    }

    #[test]
    fn names_neutral_regions_by_value() {
        let dark = solid(100, 100, [15, 15, 18, 255]);
        assert_eq!(
            dominant_color_name(&dark, [0.0, 0.0, 100.0, 100.0]),
            Some("black".to_string())
        );

        let bright = solid(100, 100, [245, 245, 240, 255]);
        assert_eq!(
            dominant_color_name(&bright, [0.0, 0.0, 100.0, 100.0]),
            Some("white".to_string())
        );
    }

    #[test]
    fn a_mixed_region_has_no_dominant_colour() {
        // Left half red, right half blue.
        let mut img = RgbaImage::new(100, 100);
        for (x, _, px) in img.enumerate_pixels_mut() {
            *px = if x < 50 {
                Rgba([220, 20, 20, 255])
            } else {
                Rgba([20, 20, 220, 255])
            };
        }
        assert_eq!(dominant_color_name(&img, [0.0, 0.0, 100.0, 100.0]), None);
    }

    #[test]
    fn stats_reflect_exposure() {
        let dark = solid(200, 200, [5, 5, 5, 255]);
        let stats = image_stats(&dark);
        assert!(stats.mean_luma < 0.05);
        assert!(stats.clipped_shadows > 0.9);
        assert!(stats.clipped_highlights < 0.01);
    }

    #[test]
    fn a_flat_image_scores_low_on_sharpness_a_textured_one_higher() {
        let flat = solid(128, 128, [128, 128, 128, 255]);

        let mut noisy = RgbaImage::new(128, 128);
        for (x, y, px) in noisy.enumerate_pixels_mut() {
            let v = if (x + y) % 2 == 0 { 30 } else { 220 };
            *px = Rgba([v, v, v, 255]);
        }

        let flat_score = image_stats(&flat).sharpness;
        let noisy_score = image_stats(&noisy).sharpness;
        assert!(flat_score < 0.05, "flat image scored {flat_score}");
        assert!(noisy_score > 0.5, "textured image scored {noisy_score}");
    }

    #[test]
    fn scene_assembly_derives_position_and_prominence() {
        let img = solid(300, 100, [200, 30, 30, 255]);
        let detections = vec![
            Detection {
                label: "car".into(),
                confidence: 0.9,
                bbox: [0.0, 0.0, 80.0, 100.0], // left third, big
            },
            Detection {
                label: "dog".into(),
                confidence: 0.8,
                bbox: [280.0, 80.0, 295.0, 95.0], // right, tiny
            },
        ];

        let scene = build_scene(&img, detections, "test-model");

        assert_eq!(scene.objects.len(), 2);
        assert_eq!(scene.objects[0].id, 1);
        assert_eq!(scene.objects[0].position, "left");
        assert_eq!(scene.objects[0].prominence, "dominant");
        assert_eq!(scene.objects[1].id, 2);
        assert_eq!(scene.objects[1].position, "right");
        assert_eq!(scene.objects[1].prominence, "minor");
        assert_eq!(scene.stats.width, 300);
    }
}
