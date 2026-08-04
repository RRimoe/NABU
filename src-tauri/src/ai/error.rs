// src-tauri/src/ai/error.rs
//! Errors carried to the frontend as translation keys.
//!
//! The backend never produces user-facing prose: it reports a stable `key`
//! plus the values that key needs. The frontend looks the key up in the active
//! language, so the same error reads correctly for every user regardless of
//! which language the application is running in.

use std::collections::HashMap;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NabuError {
    /// Translation key, e.g. `backend.modelNotDownloaded`.
    pub key: String,
    /// Placeholder values interpolated into the translated text.
    pub params: HashMap<String, String>,
    /// Untranslated technical detail, for logs and bug reports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl NabuError {
    pub fn new(key: &str) -> Self {
        Self {
            key: key.to_string(),
            params: HashMap::new(),
            detail: None,
        }
    }

    pub fn with(mut self, name: &str, value: impl ToString) -> Self {
        self.params.insert(name.to_string(), value.to_string());
        self
    }

    /// Attaches the underlying cause. Kept separate from `params` because it is
    /// diagnostic text, not something to show translated.
    pub fn detail(mut self, detail: impl ToString) -> Self {
        self.detail = Some(detail.to_string());
        self
    }
}

impl std::fmt::Display for NabuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key)?;
        if let Some(detail) = &self.detail {
            write!(f, " ({detail})")?;
        }
        Ok(())
    }
}

impl std::error::Error for NabuError {}

pub type NabuResult<T> = Result<T, NabuError>;

// --- Constructors for the errors the backend can actually produce -----------

pub fn no_input_image() -> NabuError {
    NabuError::new("backend.noInputImage")
}

pub fn no_model_selected(task: &str) -> NabuError {
    NabuError::new("backend.noModelSelected").with("task", task)
}

pub fn unknown_model(model: &str) -> NabuError {
    NabuError::new("backend.unknownModel").with("model", model)
}

pub fn model_not_downloaded(model: &str) -> NabuError {
    NabuError::new("backend.modelNotDownloaded").with("model", model)
}

pub fn missing_file(file: &str, model: &str) -> NabuError {
    NabuError::new("backend.missingFile")
        .with("file", file)
        .with("model", model)
}

pub fn download_in_progress() -> NabuError {
    NabuError::new("backend.downloadInProgress")
}

pub fn checksum_failed(file: &str) -> NabuError {
    NabuError::new("backend.checksumFailed").with("file", file)
}

pub fn http_status(file: &str, status: impl ToString) -> NabuError {
    NabuError::new("backend.httpStatus")
        .with("file", file)
        .with("status", status)
}

pub fn connection_failed(file: &str, cause: impl ToString) -> NabuError {
    NabuError::new("backend.connectionFailed")
        .with("file", file)
        .detail(cause)
}

pub fn disk_write(cause: impl ToString) -> NabuError {
    NabuError::new("backend.diskWrite")
        .with("reason", "")
        .detail(cause)
}

pub fn session_unavailable() -> NabuError {
    NabuError::new("backend.sessionUnavailable")
}

pub fn inference_failed(cause: impl ToString) -> NabuError {
    NabuError::new("backend.inferenceFailed").detail(cause)
}

pub fn unsupported_task(task: &str) -> NabuError {
    NabuError::new("backend.unsupportedTask").with("task", task)
}

pub fn invalid_data_url(cause: impl ToString) -> NabuError {
    NabuError::new("backend.invalidDataUrl").detail(cause)
}

pub fn undecodable_image(cause: impl ToString) -> NabuError {
    NabuError::new("backend.undecodableImage").detail(cause)
}

pub fn unknown(cause: impl ToString) -> NabuError {
    NabuError::new("backend.unknown").detail(cause)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialises_key_and_params() {
        let error = model_not_downloaded("RMBG-1.4");
        let json = serde_json::to_value(&error).unwrap();

        assert_eq!(json["key"], "backend.modelNotDownloaded");
        assert_eq!(json["params"]["model"], "RMBG-1.4");
        // No `detail` was set, so the field must be absent rather than null.
        assert!(json.get("detail").is_none());
    }

    #[test]
    fn keeps_technical_detail_separate_from_params() {
        let error = inference_failed("shape mismatch");
        let json = serde_json::to_value(&error).unwrap();

        assert_eq!(json["key"], "backend.inferenceFailed");
        assert_eq!(json["detail"], "shape mismatch");
        assert!(json["params"].as_object().unwrap().is_empty());
    }
}
