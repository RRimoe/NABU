// src-tauri/src/ai/vision/manifest.rs
//! Declarative description of a vision model.
//!
//! The engine defines *capabilities* with a fixed contract — a detector
//! receives an image and returns classified boxes — and a manifest adapts one
//! concrete ONNX network to that contract: which file to load, how the input
//! tensor is prepared and which decoder *family* reads the output.
//!
//! The point of this indirection is who gets to add models. Networks inside a
//! family (D-FINE, RT-DETR and the other DETR descendants, for instance) differ
//! only in resolution, normalisation and tensor names — exactly the fields a
//! manifest carries. So a model that belongs to a known family is added by
//! writing a JSON file, not by touching the engine; only a genuinely new output
//! format costs a new decoder, and that cost is paid once per family, not once
//! per model.

use serde::Deserialize;

/// What a model can do for the engine, i.e. which contract it fulfils.
///
/// `Unknown` absorbs capabilities this build has never heard of, so a manifest
/// written for a newer Nabu is skipped instead of failing the whole catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Capability {
    /// Image in, classified boxes out.
    Detector,
    /// Image plus a box in, mask of the object inside it out.
    Segmenter,
    /// Image plus a mask in, image with the masked region rebuilt out.
    Inpainter,
    /// Image in, one relative distance per pixel out.
    Depth,
    #[serde(other)]
    Unknown,
}

/// How the source image becomes the input tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResizePolicy {
    /// Resize to exactly `width` x `height`, ignoring the aspect ratio.
    Stretch,
    /// Keep the aspect ratio, place the image top-left and pad the rest.
    Letterbox,
}

impl Default for ResizePolicy {
    fn default() -> Self {
        ResizePolicy::Stretch
    }
}

/// Preparation of the input tensor, always NCHW float32.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InputSpec {
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub resize: ResizePolicy,
    /// Multiplier applied to the raw 0..255 channel value. Almost every model
    /// wants 1/255, so that is the default.
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// Per-channel mean subtracted after scaling.
    #[serde(default)]
    pub mean: [f32; 3],
    /// Per-channel divisor applied after the mean.
    #[serde(default = "default_std")]
    pub std: [f32; 3],
}

fn default_scale() -> f32 {
    1.0 / 255.0
}

fn default_std() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// How class scores are read out of the raw logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Activation {
    /// Independent per-class scores (focal-loss models: D-FINE, RT-DETR).
    Sigmoid,
    /// Softmax over the classes, with the last one meaning "no object"
    /// (the original DETR and its direct descendants).
    SoftmaxDropLast,
}

/// Geometry of the predicted boxes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoxFormat {
    /// Centre x, centre y, width, height — all normalised to 0..1.
    CxcywhNormalized,
    /// Corners x0, y0, x1, y1 — all normalised to 0..1.
    XyxyNormalized,
}

/// The decoder family and its parameters, tagged by `family` in the JSON.
///
/// One variant per output format the engine can read. A new family is the only
/// change that requires code; every model within a family is pure data.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case")]
pub enum OutputSpec {
    /// Query-based detectors: a fixed set of predictions, no anchor grid and no
    /// non-maximum suppression. Two tensors, `[1, Q, C]` logits and
    /// `[1, Q, 4]` boxes.
    #[serde(rename_all = "camelCase")]
    Detr {
        /// Name of the logits output tensor.
        logits: String,
        /// Name of the boxes output tensor.
        boxes: String,
        activation: Activation,
        box_format: BoxFormat,
        /// Class names, indexed by the logit position.
        labels: Vec<String>,
    },
    /// Segment-Anything style segmenters split into two stages: the manifest's
    /// `file` is the vision encoder, `decoder` names the prompt-encoder /
    /// mask-decoder file next to it. This family owns its preprocessing (SAM's
    /// aspect-preserving resize and zero padding), so the manifest `input`
    /// section is documentation more than instruction.
    #[serde(rename_all = "camelCase")]
    Sam {
        /// File name of the prompt encoder + mask decoder stage.
        decoder: String,
    },
    /// Mask-directed inpainters in the LaMa lineage: an image tensor and a
    /// binary mask tensor in, the rebuilt image out, values on a 0..255 scale.
    #[serde(rename_all = "camelCase")]
    Lama {
        /// Name of the image input tensor.
        image_input: String,
        /// Name of the mask input tensor.
        mask_input: String,
        /// Name of the output tensor.
        output: String,
    },
    /// Dense-prediction transformers in the DPT lineage (MiDaS, Depth
    /// Anything): one image tensor in, one `[1, H, W]` map of *relative
    /// inverse* depth out — larger means nearer, and the scale is arbitrary.
    ///
    /// Arbitrary is why the decoder normalises: nothing downstream can compare
    /// two images' raw values, so the only honest reading is "where is the near
    /// end and the far end of *this* picture".
    #[serde(rename_all = "camelCase")]
    Dpt {
        /// Name of the output tensor. Optional in practice: a model with a
        /// single output is unambiguous, and export tools rename freely.
        #[serde(default)]
        output: String,
        /// True when the network predicts inverse depth (near = large), which
        /// is what every model in this family currently does.
        #[serde(default = "default_true")]
        inverse: bool,
    },
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifest {
    pub id: String,
    pub capability: Capability,
    /// ONNX file inside the model folder.
    pub file: String,
    /// When several installed models offer the same capability, the highest
    /// priority wins. Lets a community model outrank a built-in one on purpose.
    #[serde(default)]
    pub priority: i32,
    pub input: InputSpec,
    pub output: OutputSpec,
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_max_objects")]
    pub max_objects: usize,
    /// Force CPU execution for this model. Some networks contain operators a
    /// GPU provider claims and then fails at run time — LaMa's Fourier
    /// convolutions under DirectML are the canonical case — and where a model
    /// runs is exactly the kind of quirk a manifest exists to carry.
    #[serde(default)]
    pub cpu_only: bool,
}

fn default_threshold() -> f32 {
    0.45
}

fn default_max_objects() -> usize {
    24
}

impl ModelManifest {
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_manifest_with_defaults() {
        let manifest = ModelManifest::parse(
            r#"{
                "id": "test",
                "capability": "detector",
                "file": "model.onnx",
                "input": { "width": 640, "height": 640 },
                "output": {
                    "family": "detr",
                    "logits": "logits",
                    "boxes": "pred_boxes",
                    "activation": "sigmoid",
                    "boxFormat": "cxcywh-normalized",
                    "labels": ["person"]
                }
            }"#,
        )
        .expect("parsed");

        assert_eq!(manifest.capability, Capability::Detector);
        assert_eq!(manifest.input.resize, ResizePolicy::Stretch);
        assert!((manifest.input.scale - 1.0 / 255.0).abs() < 1e-9);
        assert_eq!(manifest.input.std, [1.0, 1.0, 1.0]);
        assert_eq!(manifest.max_objects, 24);
    }

    #[test]
    fn an_unknown_capability_is_absorbed_not_fatal() {
        let manifest = ModelManifest::parse(
            r#"{
                "id": "future",
                "capability": "hologram-projector",
                "file": "model.onnx",
                "input": { "width": 64, "height": 64 },
                "output": {
                    "family": "detr",
                    "logits": "l",
                    "boxes": "b",
                    "activation": "sigmoid",
                    "boxFormat": "xyxy-normalized",
                    "labels": []
                }
            }"#,
        )
        .expect("parsed");

        assert_eq!(manifest.capability, Capability::Unknown);
    }

    #[test]
    fn the_built_in_dfine_manifest_is_valid() {
        let manifest = ModelManifest::parse(include_str!("../../../manifests/dfine-nano.json"))
            .expect("built-in manifest must parse");

        assert_eq!(manifest.id, "dfine-nano");
        assert_eq!(manifest.capability, Capability::Detector);
        let OutputSpec::Detr { labels, .. } = &manifest.output else {
            panic!("dfine manifest must be of the detr family");
        };
        assert_eq!(labels.len(), 80, "COCO has 80 classes");
    }

    #[test]
    fn parses_the_segmenter_and_inpainter_families() {
        let sam = ModelManifest::parse(
            r#"{
                "id": "s", "capability": "segmenter", "file": "encoder.onnx",
                "input": { "width": 1024, "height": 1024 },
                "output": { "family": "sam", "decoder": "decoder.onnx" }
            }"#,
        )
        .expect("sam manifest");
        assert_eq!(sam.capability, Capability::Segmenter);

        let lama = ModelManifest::parse(
            r#"{
                "id": "l", "capability": "inpainter", "file": "model.onnx",
                "input": { "width": 512, "height": 512 },
                "output": { "family": "lama", "imageInput": "image", "maskInput": "mask", "output": "output" }
            }"#,
        )
        .expect("lama manifest");
        assert_eq!(lama.capability, Capability::Inpainter);
    }
}
