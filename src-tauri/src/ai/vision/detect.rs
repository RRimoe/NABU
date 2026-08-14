// src-tauri/src/ai/vision/detect.rs
//! The detector capability: image in, classified boxes out.
//!
//! `run` is the whole contract. Which network fulfils it, at what resolution
//! and with which output tensors is the manifest's business; the caller only
//! ever sees `Detection` values in source-image pixels.

use image::RgbaImage;
use ort::session::Session;

use crate::ai::error::{self, NabuResult};
use crate::ai::tasks::ProgressFn;

use super::manifest::{Activation, BoxFormat, ModelManifest, OutputSpec};
use super::preprocess;

/// One detected object, in source-image pixel coordinates.
#[derive(Debug, Clone)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    /// x0, y0, x1, y1 in source pixels.
    pub bbox: [f32; 4],
}

/// Runs the detector described by the manifest on the image.
pub fn run(
    session: &mut Session,
    manifest: &ModelManifest,
    source: &RgbaImage,
    progress: ProgressFn<'_>,
) -> NabuResult<Vec<Detection>> {
    progress(20, "prepare");
    let (tensor, map) = preprocess::prepare(source, &manifest.input)?;

    progress(45, "detect");
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(error::inference_failed)?;

    let OutputSpec::Detr {
        logits: logits_name,
        boxes: boxes_name,
        activation,
        box_format,
        labels,
    } = &manifest.output
    else {
        return Err(error::inference_failed(format!(
            "manifest {} is not a detector family",
            manifest.id
        )));
    };

    let mut logits: Option<(Vec<usize>, Vec<f32>)> = None;
    let mut boxes: Option<(Vec<usize>, Vec<f32>)> = None;

    for (name, value) in outputs.iter() {
        if name != logits_name && name != boxes_name {
            continue;
        }
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(error::inference_failed)?;
        let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        if name == logits_name {
            logits = Some((shape, data.to_vec()));
        } else {
            boxes = Some((shape, data.to_vec()));
        }
    }

    let (logits_shape, logits) = logits
        .ok_or_else(|| error::inference_failed(format!("missing output tensor: {logits_name}")))?;
    let (boxes_shape, boxes) = boxes
        .ok_or_else(|| error::inference_failed(format!("missing output tensor: {boxes_name}")))?;

    let mut detections = decode_detr(
        &logits_shape,
        &logits,
        &boxes_shape,
        &boxes,
        *activation,
        *box_format,
        labels,
        manifest.confidence_threshold,
    )?;

    progress(85, "compose");

    // Highest confidence first, capped: a scene description with two hundred
    // half-guessed entries would drown the language model in noise.
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    detections.truncate(manifest.max_objects);

    for det in &mut detections {
        let (x0, y0) = map.to_source(det.bbox[0], det.bbox[1]);
        let (x1, y1) = map.to_source(det.bbox[2], det.bbox[3]);
        det.bbox = [x0, y0, x1, y1];
    }

    Ok(detections)
}

/// Reads DETR-family outputs into detections with normalised (0..1) boxes.
///
/// Kept free of any session or coordinate mapping so it can be tested against
/// hand-written tensors.
#[allow(clippy::too_many_arguments)]
fn decode_detr(
    logits_shape: &[usize],
    logits: &[f32],
    boxes_shape: &[usize],
    boxes: &[f32],
    activation: Activation,
    box_format: BoxFormat,
    labels: &[String],
    threshold: f32,
) -> NabuResult<Vec<Detection>> {
    let (queries, classes) = match logits_shape {
        [1, q, c] => (*q, *c),
        other => {
            return Err(error::inference_failed(format!(
                "unexpected logits shape: {other:?}"
            )));
        }
    };
    match boxes_shape {
        [1, q, 4] if *q == queries => {}
        other => {
            return Err(error::inference_failed(format!(
                "unexpected boxes shape: {other:?}"
            )));
        }
    }
    if logits.len() < queries * classes || boxes.len() < queries * 4 {
        return Err(error::inference_failed("detector output truncated"));
    }

    // With softmax the last class means "no object" and never becomes a
    // detection; with sigmoid every class is real.
    let real_classes = match activation {
        Activation::Sigmoid => classes,
        Activation::SoftmaxDropLast => classes.saturating_sub(1),
    };

    let mut detections = Vec::new();

    for q in 0..queries {
        let row = &logits[q * classes..(q + 1) * classes];

        let (best_class, score) = match activation {
            Activation::Sigmoid => {
                let (i, &logit) = row
                    .iter()
                    .enumerate()
                    .take(real_classes)
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .expect("at least one class");
                (i, 1.0 / (1.0 + (-logit).exp()))
            }
            Activation::SoftmaxDropLast => {
                let max = row.iter().cloned().fold(f32::MIN, f32::max);
                let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
                let (i, &logit) = row
                    .iter()
                    .enumerate()
                    .take(real_classes)
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .expect("at least one class");
                (i, (logit - max).exp() / sum)
            }
        };

        if score < threshold {
            continue;
        }

        let b = &boxes[q * 4..q * 4 + 4];
        let (x0, y0, x1, y1) = match box_format {
            BoxFormat::CxcywhNormalized => {
                (b[0] - b[2] / 2.0, b[1] - b[3] / 2.0, b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
            }
            BoxFormat::XyxyNormalized => (b[0], b[1], b[2], b[3]),
        };
        if x1 <= x0 || y1 <= y0 {
            continue;
        }

        detections.push(Detection {
            label: labels
                .get(best_class)
                .cloned()
                .unwrap_or_else(|| format!("class {best_class}")),
            confidence: score,
            bbox: [
                x0.clamp(0.0, 1.0),
                y0.clamp(0.0, 1.0),
                x1.clamp(0.0, 1.0),
                y1.clamp(0.0, 1.0),
            ],
        });
    }

    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        vec!["person".into(), "car".into(), "dog".into()]
    }

    /// Sigmoid inverse: the logit whose sigmoid is `p`.
    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    #[test]
    fn sigmoid_decoding_keeps_confident_queries_only() {
        // Two queries, three classes. Query 0 is a confident car, query 1 is
        // noise below the threshold.
        let logits = vec![
            logit(0.10), logit(0.92), logit(0.05),
            logit(0.20), logit(0.30), logit(0.10),
        ];
        let boxes = vec![
            0.5, 0.5, 0.2, 0.4, // cxcywh
            0.1, 0.1, 0.1, 0.1,
        ];

        let detections = decode_detr(
            &[1, 2, 3],
            &logits,
            &[1, 2, 4],
            &boxes,
            Activation::Sigmoid,
            BoxFormat::CxcywhNormalized,
            &labels(),
            0.45,
        )
        .expect("decoded");

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "car");
        assert!((detections[0].confidence - 0.92).abs() < 1e-3);

        let [x0, y0, x1, y1] = detections[0].bbox;
        assert!((x0 - 0.4).abs() < 1e-5 && (y0 - 0.3).abs() < 1e-5);
        assert!((x1 - 0.6).abs() < 1e-5 && (y1 - 0.7).abs() < 1e-5);
    }

    #[test]
    fn softmax_drop_last_never_reports_the_no_object_class() {
        // One query where the last ("no object") class dominates: nothing must
        // come out, even though its probability is far above the threshold.
        let logits = vec![0.0, 0.0, 10.0];
        let boxes = vec![0.5, 0.5, 0.5, 0.5];

        let detections = decode_detr(
            &[1, 1, 3],
            &logits,
            &[1, 1, 4],
            &boxes,
            Activation::SoftmaxDropLast,
            BoxFormat::CxcywhNormalized,
            &["person".into(), "car".into()],
            0.45,
        )
        .expect("decoded");

        assert!(detections.is_empty());
    }

    #[test]
    fn softmax_reports_a_dominant_real_class() {
        let logits = vec![8.0, 0.0, 0.0];
        let boxes = vec![0.25, 0.25, 0.75, 0.75];

        let detections = decode_detr(
            &[1, 1, 3],
            &logits,
            &[1, 1, 4],
            &boxes,
            Activation::SoftmaxDropLast,
            BoxFormat::XyxyNormalized,
            &["person".into(), "car".into()],
            0.45,
        )
        .expect("decoded");

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "person");
        assert_eq!(detections[0].bbox, [0.25, 0.25, 0.75, 0.75]);
    }

    #[test]
    fn degenerate_boxes_are_dropped() {
        let logits = vec![logit(0.9), logit(0.1), logit(0.1)];
        let boxes = vec![0.5, 0.5, 0.0, 0.0]; // zero width and height

        let detections = decode_detr(
            &[1, 1, 3],
            &logits,
            &[1, 1, 4],
            &boxes,
            Activation::Sigmoid,
            BoxFormat::CxcywhNormalized,
            &labels(),
            0.45,
        )
        .expect("decoded");

        assert!(detections.is_empty());
    }

    #[test]
    fn a_shape_mismatch_is_an_error_not_a_panic() {
        let result = decode_detr(
            &[1, 2, 3],
            &[0.0; 6],
            &[1, 3, 4], // box count disagrees with the query count
            &[0.0; 12],
            Activation::Sigmoid,
            BoxFormat::CxcywhNormalized,
            &labels(),
            0.45,
        );
        assert!(result.is_err());
    }
}
