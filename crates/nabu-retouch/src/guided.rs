//! Removal with a model and the fill together.
//!
//! An inpainting network is good at *what* belongs behind a removed object (the
//! line of a kerb carrying on, the light falling across a wall) and bad at the
//! texture: it works at a few hundred pixels and its answer, magnified to the
//! photograph, is a soft smear that reads as a halo. The patch fill is the
//! opposite: everything it puts in the hole is the photograph's own pixels, but
//! across a large hole it has no idea of the structure.
//!
//! So the network's answer seeds the synthesis. The coarsest level of the
//! pyramid starts from it rather than from the hole's edge, so the layout the
//! network chose is what the patches are first matched against; the EM
//! iterations then replace its soft pixels with pieces of the photograph, level
//! by level, up to full resolution.

use image::{GrayImage, RgbaImage};

use crate::content_aware::{color_level, fill_owned, FillOptions};
use crate::heal::HealError;
use crate::interrupt::Interrupt;

#[derive(Clone, Debug)]
pub struct GuidedOptions {
    /// Colour adaptation, as in [`crate::HealOptions`].
    pub color: String,
    /// Consider mirrored copies of the surroundings too.
    pub mirror: bool,
}

impl Default for GuidedOptions {
    fn default() -> Self {
        GuidedOptions { color: "default".into(), mirror: true }
    }
}

/// Fills the masked pixels of `photo` (mask above 127) with texture from the
/// rest of it, starting from the layout of `guide` (the network's answer, same
/// size as the photo). Pixels outside the mask are returned unchanged.
pub fn fill_guided(
    photo: &RgbaImage,
    guide: &RgbaImage,
    mask: &GrayImage,
    options: &GuidedOptions,
    progress: &(dyn Fn(f32) -> bool + Sync),
) -> Result<RgbaImage, HealError> {
    let (w, h) = (photo.width() as usize, photo.height() as usize);
    if guide.dimensions() != photo.dimensions() || mask.dimensions() != photo.dimensions() {
        return Err(HealError::SizeMismatch);
    }
    let hole: Vec<bool> = mask.pixels().map(|p| p[0] > 127).collect();
    if !hole.iter().any(|&m| m) {
        return Err(HealError::EmptyMask);
    }

    let mut pixels = Vec::with_capacity(w * h * 3);
    let mut source = vec![false; w * h];
    for (i, (p, g)) in photo.pixels().zip(guide.pixels()).enumerate() {
        let from = if hole[i] { g } else { p };
        pixels.extend([from[0], from[1], from[2]].map(|c| f32::from(c) / 255.0));
        source[i] = !hole[i] && p[3] > 0;
    }
    if !source.iter().any(|&s| s) {
        return Err(HealError::NothingToSample);
    }

    let fill = FillOptions {
        color_adaptation: color_level(&options.color),
        mirror: options.mirror,
        init_from_input: true,
        ..FillOptions::default()
    };
    let stop = std::sync::atomic::AtomicBool::new(!progress(0.0));
    let cancel = || stop.load(std::sync::atomic::Ordering::Relaxed);
    let report = |f: f32| {
        if !progress(f) {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    };
    let filled = fill_owned(w, h, 3, pixels, &hole, &source, &fill, &Interrupt::new(&cancel, &report))
        .map_err(|_| HealError::Cancelled)?;

    let mut out = photo.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        if hole[i] {
            for c in 0..3 {
                p[c] = (filled[i * 3 + c] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    progress(1.0);
    Ok(out)
}
