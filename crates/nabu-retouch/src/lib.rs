//! Retouching without models: content-aware fill and spot healing.
//!
//! The patch-based completion (PatchMatch inside Wexler's EM scheme, with the
//! colour adaptation of Image Melding) is PhotoCraft's, used under its MIT
//! licence; see NOTICE. Nabu's own parts are [`heal`], which turns a photo and a
//! mask into a filled photo the way the editor needs it, and [`fill_guided`],
//! which makes the fill follow an inpainting network's answer.
//!
//! The inpainting models fill what is behind a removed object. This fills from
//! the photo itself, at full resolution and with no download: the right tool
//! for skin, sky, walls, grass and anything else made of its own texture.

pub mod content_aware;
pub mod inpaint;
pub mod interrupt;
mod par;
mod poisson;
mod guided;
mod heal;

pub use guided::{fill_guided, GuidedOptions};
pub use heal::{heal, HealOptions, HealError};
pub use interrupt::{Cancelled, Interrupt};

/// Axis-aligned pixel rectangle, `x1`/`y1` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub const EMPTY: Rect = Rect { x0: 0, y0: 0, x1: 0, y1: 0 };

    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Rect { x0, y0, x1, y1 }
    }

    pub fn width(&self) -> u32 {
        if self.x1 > self.x0 { self.x1.abs_diff(self.x0) } else { 0 }
    }

    pub fn height(&self) -> u32 {
        if self.y1 > self.y0 { self.y1.abs_diff(self.y0) } else { 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }
}
