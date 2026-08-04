// src-tauri/src/ai/registry.rs
//! Catalogue of the local AI models Nabu supports.
//!
//! Each entry describes downloadable files with a URL, an expected size and a
//! SHA-256 digest. The hashes come from Hugging Face LFS metadata and are
//! verified at the end of every download: a corrupted or substituted file is
//! discarded instead of reaching the inference engine.
//!
//! `description` is a translation key, not prose: the text shown to the user
//! lives in the language catalogue so it can be translated like everything else.

use serde::Serialize;

/// Functional category, used by the interface to map a model to a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelTask {
    BackgroundRemoval,
    Upscale,
    Segmentation,
}

/// A single file making up a model (some models need more than one).
#[derive(Debug, Clone, Serialize)]
pub struct ModelFile {
    /// Name the file is saved under inside the model folder.
    pub file_name: &'static str,
    pub url: &'static str,
    /// Expected size in bytes, used for progress before the HTTP response arrives.
    pub size_bytes: u64,
    /// Expected SHA-256 in lowercase hex. `None` when the source publishes no
    /// trustworthy digest, in which case only the size is checked.
    pub sha256: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub task: ModelTask,
    /// Translation key for the description shown in the model manager.
    pub description_key: &'static str,
    pub license: &'static str,
    /// `false` for licences that forbid commercial use; the interface flags it.
    pub commercial_use: bool,
    /// Rough memory needed to run the model, in MB.
    pub vram_required_mb: u32,
    pub files: &'static [ModelFile],
    /// Working resolution the network expects (longest side), when fixed.
    pub input_size: Option<u32>,
    /// Scale factor for super-resolution models.
    pub scale_factor: Option<u32>,
}

impl ModelSpec {
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }
}

/// Static catalogue. Every URL has been verified reachable and the sizes match
/// what the source declares.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "rmbg-1.4-quantized",
        name: "RMBG-1.4 (quantizzato)",
        task: ModelTask::BackgroundRemoval,
        description_key: "model.rmbg-1.4-quantized.description",
        license: "BRIA RAIL",
        commercial_use: false,
        vram_required_mb: 512,
        files: &[ModelFile {
            file_name: "model.onnx",
            url: "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
            size_bytes: 44_403_226,
            sha256: Some("a6648479275dfd0ede0f3a8abc20aa5c437b394681b05e5af6d268250aaf40f3"),
        }],
        input_size: Some(1024),
        scale_factor: None,
    },
    ModelSpec {
        id: "rmbg-1.4",
        name: "RMBG-1.4 (piena precisione)",
        task: ModelTask::BackgroundRemoval,
        description_key: "model.rmbg-1.4.description",
        license: "BRIA RAIL",
        commercial_use: false,
        vram_required_mb: 1024,
        files: &[ModelFile {
            file_name: "model.onnx",
            url: "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
            size_bytes: 176_153_355,
            sha256: Some("8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f"),
        }],
        input_size: Some(1024),
        scale_factor: None,
    },
    ModelSpec {
        id: "isnet-general",
        name: "ISNet General Use",
        task: ModelTask::BackgroundRemoval,
        description_key: "model.isnet-general.description",
        license: "Apache-2.0 / MIT",
        commercial_use: true,
        vram_required_mb: 1024,
        files: &[ModelFile {
            file_name: "model.onnx",
            url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
            size_bytes: 178_648_008,
            sha256: Some("60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"),
        }],
        input_size: Some(1024),
        scale_factor: None,
    },
    ModelSpec {
        id: "swin2sr-lightweight-x2",
        name: "Swin2SR Lightweight x2",
        task: ModelTask::Upscale,
        description_key: "model.swin2sr-lightweight-x2.description",
        license: "Apache-2.0",
        commercial_use: true,
        vram_required_mb: 512,
        files: &[ModelFile {
            file_name: "model.onnx",
            url: "https://huggingface.co/Xenova/swin2SR-lightweight-x2-64/resolve/main/onnx/model.onnx",
            size_bytes: 8_078_888,
            sha256: Some("c2abbfe0cc8e685b5e11964970f8ebe3d24072e904fd5545a30ac31ec1e110db"),
        }],
        input_size: None,
        scale_factor: Some(2),
    },
    ModelSpec {
        id: "swin2sr-classical-x2",
        name: "Swin2SR Classical x2",
        task: ModelTask::Upscale,
        description_key: "model.swin2sr-classical-x2.description",
        license: "Apache-2.0",
        commercial_use: true,
        vram_required_mb: 1536,
        files: &[ModelFile {
            file_name: "model.onnx",
            url: "https://huggingface.co/Xenova/swin2SR-classical-sr-x2-64/resolve/main/onnx/model.onnx",
            size_bytes: 54_428_699,
            sha256: Some("6dde3fe2440543ccae7c40d175609f83c18aeaa3d8456745c4329ef97ae744bd"),
        }],
        input_size: None,
        scale_factor: Some(2),
    },
    ModelSpec {
        id: "slimsam-77",
        name: "SlimSAM-77 (Segment Anything)",
        task: ModelTask::Segmentation,
        description_key: "model.slimsam-77.description",
        license: "Apache-2.0",
        commercial_use: true,
        vram_required_mb: 1024,
        files: &[
            ModelFile {
                file_name: "vision_encoder.onnx",
                url: "https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/vision_encoder.onnx",
                size_bytes: 23_276_014,
                sha256: Some("9f8433273a6750b587779baa0cf5508111001bf7e7acfcf585d370139fd366d0"),
            },
            ModelFile {
                file_name: "prompt_encoder_mask_decoder.onnx",
                url: "https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
                size_bytes: 16_557_892,
                sha256: Some("f4514391764fbd56e08e119060d874ecd7d52994bfb1968af159e12d4943b5bb"),
            },
        ],
        input_size: Some(1024),
        scale_factor: None,
    },
];

pub fn find(model_id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == model_id)
}
