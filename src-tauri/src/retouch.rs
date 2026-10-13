// src-tauri/src/retouch.rs
//! Content-aware fill and the healing brush, as the interface calls them.
//!
//! The interface sends only the part of the photo the fill can reach (the
//! sampling window around the mask), as raw pixels: the RGBA crop followed by
//! one mask byte per pixel. The healed crop comes back the same way, so a
//! retouch on a large photo moves a few megabytes, not the whole picture.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use image::{GrayImage, RgbaImage};
use nabu_retouch::{heal, HealError, HealOptions};
use serde::Serialize;
use tauri::ipc::{InvokeBody, Request, Response};
use tauri::{AppHandle, Emitter, State};

use crate::ai::error::{self, NabuError};

/// Lets a running fill be stopped from the interface.
#[derive(Default)]
pub struct RetouchState {
    cancel: AtomicBool,
}

#[derive(Serialize, Clone)]
struct RetouchProgress {
    percent: u32,
}

fn header<'a>(request: &'a Request<'_>, name: &str) -> Option<&'a str> {
    request.headers().get(name).and_then(|v| v.to_str().ok())
}

#[tauri::command]
pub async fn heal_region(
    app: AppHandle,
    state: State<'_, Arc<RetouchState>>,
    request: Request<'_>,
) -> Result<Response, NabuError> {
    let InvokeBody::Raw(body) = request.body() else {
        return Err(error::retouch_failed("expected a binary message"));
    };
    let size = |name| header(&request, name).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    let (width, height) = (size("x-nabu-width"), size("x-nabu-height"));
    let pixels = width as usize * height as usize;
    if pixels == 0 || body.len() != pixels * 5 {
        return Err(error::retouch_failed("size does not match the message"));
    }
    let options = HealOptions {
        color: header(&request, "x-nabu-color").unwrap_or("high").to_string(),
        mirror: header(&request, "x-nabu-mirror") != Some("false"),
    };
    let body = body.clone();
    let state = state.inner().clone();
    state.cancel.store(false, Ordering::SeqCst);

    tauri::async_runtime::spawn_blocking(move || {
        let photo = RgbaImage::from_raw(width, height, body[..pixels * 4].to_vec())
            .ok_or_else(|| error::retouch_failed("bad pixels"))?;
        let mask = GrayImage::from_raw(width, height, body[pixels * 4..].to_vec())
            .ok_or_else(|| error::retouch_failed("bad mask"))?;
        drop(body);

        let last = std::sync::atomic::AtomicU32::new(u32::MAX);
        let healed = heal(&photo, &mask, &options, &|fraction| {
            let percent = (fraction * 100.0).round().clamp(0.0, 100.0) as u32;
            // The fill reports often; the bar only needs whole percents.
            if last.swap(percent, Ordering::Relaxed) != percent {
                let _ = app.emit("retouch-progress", RetouchProgress { percent });
            }
            !state.cancel.load(Ordering::SeqCst)
        })
        .map_err(|e| match e {
            HealError::Cancelled => error::cancelled(),
            HealError::EmptyMask => error::retouch_empty(),
            HealError::NothingToSample => error::retouch_nothing_to_sample(),
            HealError::SizeMismatch => error::retouch_failed("mask size"),
        })?;
        Ok(Response::new(healed.into_raw()))
    })
    .await
    .map_err(error::unknown)?
}

#[tauri::command]
pub fn cancel_heal(state: State<'_, Arc<RetouchState>>) {
    state.cancel.store(true, Ordering::SeqCst);
}
