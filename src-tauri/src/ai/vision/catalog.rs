// src-tauri/src/ai/vision/catalog.rs
//! Which vision models exist and what each one can do.
//!
//! Two sources feed the catalogue, both speaking the same manifest language:
//!
//! * curated models from the download registry, whose manifests ship inside
//!   the binary (and can be overridden by a `manifest.json` next to the model,
//!   which is how a manifest fix reaches users without a release);
//! * sideloaded models: any folder inside the models directory that holds a
//!   `manifest.json` and the ONNX file it names. Dropping such a folder is the
//!   whole installation procedure, exactly like translations in `languages/`.
//!
//! A folder with a broken manifest is skipped with a log line, never fatal: one
//! community model must not take the whole engine down.

use std::path::PathBuf;

use tauri::AppHandle;

use crate::ai::download::{self, InstallState};
use crate::ai::error::{self, NabuResult};
use crate::ai::registry;

use super::manifest::{Capability, ModelManifest};

/// Manifests of the curated models, compiled in so a fresh download works
/// offline and cannot lose its description.
const BUILT_IN: &[&str] = &[
    include_str!("../../../manifests/dfine-nano.json"),
    include_str!("../../../manifests/slimsam-77.json"),
    include_str!("../../../manifests/lama-inpainting.json"),
    include_str!("../../../manifests/depth-anything-v2-small.json"),
    include_str!("../../../manifests/depth-anything-v2-small-quantized.json"),
];

/// A model ready to run: its manifest plus where it lives.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub model_id: String,
    /// Full path of the primary ONNX file.
    pub path: PathBuf,
    /// Folder holding the model, for families with more than one file.
    pub dir: PathBuf,
    pub manifest: ModelManifest,
}

/// The built-in manifest for a registry model, if it has one.
pub fn embedded(model_id: &str) -> Option<ModelManifest> {
    BUILT_IN.iter().find_map(|text| {
        ModelManifest::parse(text)
            .ok()
            .filter(|m| m.id == model_id)
    })
}

/// Reads `manifest.json` from a model folder, if present and valid.
fn disk_manifest(dir: &PathBuf) -> Option<ModelManifest> {
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).ok()?;
    match ModelManifest::parse(&text) {
        Ok(manifest) => Some(manifest),
        Err(err) => {
            eprintln!("[nabu] ignoring invalid manifest {}: {err}", path.display());
            None
        }
    }
}

/// Every installed model offering `capability`, best first.
pub fn models_for(app: &AppHandle, capability: Capability) -> Vec<ResolvedModel> {
    let mut found: Vec<ResolvedModel> = Vec::new();

    // Curated models: only those actually downloaded. A manifest on disk wins
    // over the embedded one, so a fix can be tested without recompiling.
    for spec in registry::MODELS {
        if download::status_of(app, spec).state != InstallState::Ready {
            continue;
        }
        let Ok(dir) = download::model_dir(app, spec.id) else {
            continue;
        };
        let Some(manifest) = disk_manifest(&dir).or_else(|| embedded(spec.id)) else {
            continue;
        };
        if manifest.capability != capability {
            continue;
        }
        let path = dir.join(&manifest.file);
        if path.is_file() {
            found.push(ResolvedModel {
                model_id: spec.id.to_string(),
                path,
                dir,
                manifest,
            });
        }
    }

    // Sideloaded models: folders the registry knows nothing about.
    if let Ok(root) = download::models_root(app) {
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                let id = entry.file_name().to_string_lossy().to_string();
                if registry::find(&id).is_some() {
                    continue; // already handled above
                }
                let Some(manifest) = disk_manifest(&dir) else {
                    continue;
                };
                if manifest.capability != capability {
                    continue;
                }
                let path = dir.join(&manifest.file);
                if path.is_file() {
                    found.push(ResolvedModel {
                        model_id: id,
                        path,
                        dir,
                        manifest,
                    });
                }
            }
        }
    }

    found.sort_by(|a, b| b.manifest.priority.cmp(&a.manifest.priority));
    found
}

/// The detector the fast pass should use.
pub fn best_detector(app: &AppHandle) -> NabuResult<ResolvedModel> {
    models_for(app, Capability::Detector)
        .into_iter()
        .next()
        .ok_or_else(error::no_detection_model)
}

/// The segmenter object removal should use.
pub fn best_segmenter(app: &AppHandle) -> NabuResult<ResolvedModel> {
    models_for(app, Capability::Segmenter)
        .into_iter()
        .next()
        .ok_or_else(|| error::no_model_selected("segmentation"))
}

/// The inpainter object removal should use.
pub fn best_inpainter(app: &AppHandle) -> NabuResult<ResolvedModel> {
    models_for(app, Capability::Inpainter)
        .into_iter()
        .next()
        .ok_or_else(|| error::no_model_selected("inpainting"))
}

/// The depth model, when one is installed.
///
/// Depth is genuinely optional: every effect that reads it has a geometric
/// fallback, so this returns an option rather than an error and the caller
/// decides whether the absence is worth mentioning.
pub fn best_depth(app: &AppHandle) -> Option<ResolvedModel> {
    models_for(app, Capability::Depth).into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_built_in_manifest_parses_and_names_a_registry_model() {
        for text in BUILT_IN {
            let manifest = ModelManifest::parse(text).expect("built-in manifest must parse");
            assert!(
                registry::find(&manifest.id).is_some(),
                "manifest {} does not match any registry model",
                manifest.id
            );
        }
    }

    #[test]
    fn embedded_finds_the_dfine_manifest() {
        let manifest = embedded("dfine-nano").expect("dfine manifest");
        assert_eq!(manifest.capability, Capability::Detector);
    }
}
