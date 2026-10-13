//! The editor's entry point: a photo and a mask in, the photo with the masked
//! area rebuilt from its surroundings out.

use std::sync::atomic::{AtomicBool, Ordering};

use image::{GrayImage, RgbaImage};

use crate::content_aware::{color_level, fill_owned, sampling_window, FillOptions};
use crate::interrupt::Interrupt;
use crate::Rect;

/// How far the fill reaches past the painted mask, in pixels. A mask drawn
/// exactly on a blemish leaves its soft edge outside; the fill has to cover it
/// or a faint ring stays behind.
const GROW: usize = 2;

#[derive(Clone, Debug)]
pub struct HealOptions {
    /// Colour adaptation: `none`, `default`, `high` or `veryHigh`. Higher lets
    /// copied texture take the light of its new place, which a gradient such
    /// as a sky or a cheek needs.
    pub color: String,
    /// Consider mirrored copies of the surroundings too.
    pub mirror: bool,
}

impl Default for HealOptions {
    fn default() -> Self {
        HealOptions { color: "high".into(), mirror: true }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum HealError {
    /// The mask selects nothing.
    EmptyMask,
    /// The mask covers almost everything: there is nothing left to copy from.
    NothingToSample,
    SizeMismatch,
    Cancelled,
}

/// Grows a mask by `r` pixels (square structuring element, separable).
fn dilate(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut rows = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
            rows[y * w + x] = (a..=b).any(|i| mask[y * w + i]);
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (a..=b).any(|j| rows[j * w + x]);
        }
    }
    out
}

/// Connected regions of the mask (8-connected): each region's pixel indices,
/// and for every pixel the number of its region plus one (0 outside the mask).
fn regions(mask: &[bool], w: usize, h: usize) -> (Vec<Vec<usize>>, Vec<u32>) {
    let mut label = vec![0u32; w * h];
    let mut out = Vec::new();
    for start in 0..w * h {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        let id = out.len() as u32 + 1;
        let mut stack = vec![start];
        let mut region = Vec::new();
        label[start] = id;
        while let Some(i) = stack.pop() {
            region.push(i);
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let n = ny as usize * w + nx as usize;
                    if mask[n] && label[n] == 0 {
                        label[n] = id;
                        stack.push(n);
                    }
                }
            }
        }
        out.push(region);
    }
    (out, label)
}

/// Rebuilds every pixel where `mask` is set (value above 127) from the rest of
/// the photo. Separate spots are filled one after the other, each from its own
/// neighbourhood: a blemish on a cheek should borrow skin, not the sky that
/// another spot in the same stroke happened to be near.
///
/// `progress` receives 0..1 and returns false to cancel.
pub fn heal(
    photo: &RgbaImage,
    mask: &GrayImage,
    options: &HealOptions,
    progress: &(dyn Fn(f32) -> bool + Sync),
) -> Result<RgbaImage, HealError> {
    let (w, h) = (photo.width() as usize, photo.height() as usize);
    if mask.width() as usize != w || mask.height() as usize != h {
        return Err(HealError::SizeMismatch);
    }
    let painted: Vec<bool> = mask.pixels().map(|p| p[0] > 127).collect();
    if !painted.iter().any(|&m| m) {
        return Err(HealError::EmptyMask);
    }
    let hole = dilate(&painted, w, h, GROW);
    let (spots, label) = regions(&hole, w, h);
    let total: usize = spots.iter().map(Vec::len).sum();

    let fill = FillOptions {
        color_adaptation: color_level(&options.color),
        mirror: options.mirror,
        ..FillOptions::default()
    };

    let mut out = photo.clone();
    let mut done = 0usize;
    for (n, spot) in spots.iter().enumerate() {
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for &i in spot {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
        let window = sampling_window(
            Rect::new(x0 as i32, y0 as i32, x1 as i32, y1 as i32),
            Rect::new(0, 0, w as i32, h as i32),
        );
        let (wx, wy) = (window.x0 as usize, window.y0 as usize);
        let (ww, wh) = (window.width() as usize, window.height() as usize);

        let mut pixels = Vec::with_capacity(ww * wh * 3);
        let mut local_hole = vec![false; ww * wh];
        let mut source = vec![false; ww * wh];
        for y in 0..wh {
            for x in 0..ww {
                let p = out.get_pixel((wx + x) as u32, (wy + y) as u32);
                pixels.extend([p[0], p[1], p[2]].map(|c| f32::from(c) / 255.0));
                let k = y * ww + x;
                let g = (wy + y) * w + wx + x;
                local_hole[k] = label[g] == n as u32 + 1;
                // Every other masked pixel is not a source either: copying
                // from another blemish would move it rather than remove it.
                source[k] = !hole[g] && p[3] > 0;
            }
        }
        if !source.iter().any(|&s| s) {
            return Err(HealError::NothingToSample);
        }

        let base = done as f32 / total.max(1) as f32;
        let share = spot.len() as f32 / total.max(1) as f32;
        let stop = AtomicBool::new(!progress(base));
        let cancel = || stop.load(Ordering::Relaxed);
        let report = |f: f32| {
            if !progress(base + share * f) {
                stop.store(true, Ordering::Relaxed);
            }
        };
        let ctl = Interrupt::new(&cancel, &report);
        let filled = fill_owned(ww, wh, 3, pixels, &local_hole, &source, &fill, &ctl)
            .map_err(|_| HealError::Cancelled)?;

        for y in 0..wh {
            for x in 0..ww {
                if local_hole[y * ww + x] {
                    let k = (y * ww + x) * 3;
                    let p = out.get_pixel_mut((wx + x) as u32, (wy + y) as u32);
                    for c in 0..3 {
                        p[c] = (filled[k + c] * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        done += spot.len();
    }
    progress(1.0);
    Ok(out)
}
