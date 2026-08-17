// src-tauri/src/ai/vision/synth.rs
//! Putting the detail back into an inpainted hole.
//!
//! An inpainting network answers "what plausibly belongs here" and answers it
//! smoothly, because a smooth answer is the safest one it can give: averaging
//! over everything that might have been there is exactly what minimises its
//! training loss. That is right for structure — an edge, a horizon, a wall —
//! and wrong for texture, because the average of a thousand pebbles is grey mud.
//!
//! But on a beach the real pebbles are three centimetres away. So the network's
//! answer is kept as the *guide* — it knows where the light falls and how the
//! ground recedes — and the texture is **copied** from the photograph itself:
//! for each spot in the hole, the most similar real place nearby, matched
//! against the guide. Every pixel that comes out is a pixel that was really
//! there, which is why it is sharp.
//!
//! This is PatchMatch's correspondence search, used for detail transfer rather
//! than for synthesis from nothing. Deterministic despite the "random" search:
//! the generator is seeded, so pressing remove twice gives the same photograph.

use image::{GrayImage, RgbaImage};

/// Half-width of the patch compared when looking for a match, as a fraction of
/// the shorter side of the grid the search runs on.
///
/// A fraction rather than a count, because the grid is not always the same
/// size. Four pixels is right at 512 and far too coarse once the same four
/// pixels stand for seventy of the photograph's: a patch has to cover about as
/// much of the scene wherever it runs, or it stops comparing leaves and starts
/// comparing hedges.
const PATCH_FRACTION: f32 = 0.008;
const MIN_PATCH: i32 = 3;
const MAX_PATCH: i32 = 6;

/// How far a match may come from, as a fraction of the grid's longest side.
const SEARCH_FRACTION: f32 = 0.25;

/// What a match at the far end of that reach costs, against a total mismatch.
const LOCALITY: f32 = 0.05;

/// Attempts at finding a starting match beside the hole before falling back to
/// anywhere in the picture.
const SEED_TRIES: u32 = 8;

/// Sweeps of propagation and random search. Three is where it stops improving
/// visibly on the sizes this runs at.
const ITERATIONS: u32 = 3;

/// Radius, in pixels, of the blur that separates structure from detail.
const DETAIL_RADIUS: u32 = 3;

/// Below this much high-frequency energy the surroundings are not a texture,
/// and there is nothing for a search to match on. Grain goes in instead — see
/// [`transfer_grain`].
const TEXTURE_FLOOR: f32 = 2.6;

/// How far a hole pixel looks for grain to borrow, in pixels.
const GRAIN_REACH: i32 = 24;

/// Attempts at finding a real pixel within that reach before giving up.
const GRAIN_TRIES: u32 = 12;

/// Amplitude, in levels, above which a high-frequency residual is not grain.
const GRAIN_CEILING: f32 = 12.0;

/// A tiny deterministic generator, so a removal is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        // xorshift64*, plenty for choosing candidate offsets.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn below(&mut self, limit: usize) -> usize {
        if limit == 0 { 0 } else { self.next() as usize % limit }
    }
}

/// How much detail the known part of the window carries.
///
/// Mean absolute difference between neighbouring pixels, outside the hole. A
/// beach scores high, an evening sky scores near zero, and the second is a
/// picture whose smooth reconstruction is already correct.
pub fn texture_energy(image: &RgbaImage, mask: &GrayImage) -> f32 {
    let (w, h) = image.dimensions();
    let mut sum = 0f32;
    let mut count = 0u32;

    for y in 0..h {
        for x in 0..w.saturating_sub(1) {
            if mask.get_pixel(x, y)[0] > 0 || mask.get_pixel(x + 1, y)[0] > 0 {
                continue;
            }
            let a = image.get_pixel(x, y);
            let b = image.get_pixel(x + 1, y);
            for c in 0..3 {
                sum += (a[c] as f32 - b[c] as f32).abs();
            }
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum / (count as f32 * 3.0) }
}

/// Separable box blur over RGB using an O(1) per-pixel sliding accumulator.
fn blur(image: &RgbaImage, radius: u32) -> RgbaImage {
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 || radius == 0 {
        return image.clone();
    }
    let r = radius as i32;
    let span = (r * 2 + 1) as f32;
    let inv_span = 1.0 / span;

    let src = image.as_raw();
    let mut horizontal = vec![0u8; (w * h * 4) as usize];

    // Horizontal pass: sliding accumulator per row
    for y in 0..h {
        let row_offset = (y * w * 4) as usize;
        let mut acc = [0f32; 3];

        // Seed the accumulator with clamped boundary values
        for dx in -r..=r {
            let sx = dx.clamp(0, w as i32 - 1) as usize;
            let idx = row_offset + sx * 4;
            acc[0] += src[idx] as f32;
            acc[1] += src[idx + 1] as f32;
            acc[2] += src[idx + 2] as f32;
        }

        for x in 0..w {
            let out_idx = row_offset + x as usize * 4;
            horizontal[out_idx] = (acc[0] * inv_span).round().clamp(0.0, 255.0) as u8;
            horizontal[out_idx + 1] = (acc[1] * inv_span).round().clamp(0.0, 255.0) as u8;
            horizontal[out_idx + 2] = (acc[2] * inv_span).round().clamp(0.0, 255.0) as u8;
            horizontal[out_idx + 3] = src[out_idx + 3];

            // Slide window: remove leftmost pixel, add rightmost pixel
            let left_x = (x as i32 - r).clamp(0, w as i32 - 1) as usize;
            let right_x = (x as i32 + r + 1).clamp(0, w as i32 - 1) as usize;
            let left_idx = row_offset + left_x * 4;
            let right_idx = row_offset + right_x * 4;

            acc[0] += src[right_idx] as f32 - src[left_idx] as f32;
            acc[1] += src[right_idx + 1] as f32 - src[left_idx + 1] as f32;
            acc[2] += src[right_idx + 2] as f32 - src[left_idx + 2] as f32;
        }
    }

    let mut out = RgbaImage::new(w, h);
    let out_raw: &mut [u8] = out.as_mut();

    // Vertical pass: sliding accumulator per column
    for x in 0..w {
        let x_offset = x as usize * 4;
        let stride = (w * 4) as usize;
        let mut acc = [0f32; 3];

        // Seed accumulator
        for dy in -r..=r {
            let sy = dy.clamp(0, h as i32 - 1) as usize;
            let idx = sy * stride + x_offset;
            acc[0] += horizontal[idx] as f32;
            acc[1] += horizontal[idx + 1] as f32;
            acc[2] += horizontal[idx + 2] as f32;
        }

        for y in 0..h {
            let out_idx = y as usize * stride + x_offset;
            out_raw[out_idx] = (acc[0] * inv_span).round().clamp(0.0, 255.0) as u8;
            out_raw[out_idx + 1] = (acc[1] * inv_span).round().clamp(0.0, 255.0) as u8;
            out_raw[out_idx + 2] = (acc[2] * inv_span).round().clamp(0.0, 255.0) as u8;
            out_raw[out_idx + 3] = horizontal[out_idx + 3];

            let top_y = (y as i32 - r).clamp(0, h as i32 - 1) as usize;
            let bottom_y = (y as i32 + r + 1).clamp(0, h as i32 - 1) as usize;
            let top_idx = top_y * stride + x_offset;
            let bottom_idx = bottom_y * stride + x_offset;

            acc[0] += horizontal[bottom_idx] as f32 - horizontal[top_idx] as f32;
            acc[1] += horizontal[bottom_idx + 1] as f32 - horizontal[top_idx + 1] as f32;
            acc[2] += horizontal[bottom_idx + 2] as f32 - horizontal[top_idx + 2] as f32;
        }
    }

    out
}

/// Half-width of the comparison patch on a grid of this size.
fn patch_radius(w: u32, h: u32) -> i32 {
    ((w.min(h) as f32 * PATCH_FRACTION).round() as i32).clamp(MIN_PATCH, MAX_PATCH)
}

/// The cost of a patch that agrees with the guide on nothing at all.
fn worst_cost(patch: i32) -> f32 {
    let side = (patch * 2 + 1) as f32;
    side * side * 3.0 * 255.0 * 255.0
}

/// What it costs to take a patch from `(sx, sy)` rather than from beside the
/// hole at `(tx, ty)`.
///
/// A perfect match on the far side of the photograph is usually the wrong
/// answer. Sunlit sky matches the pale side of a hedge well enough to win on
/// colour alone, and a hedge rebuilt out of sky is exactly what that looks
/// like. The penalty grows with the square of the distance and is measured
/// against a total mismatch, so it argues with the patch cost in the same
/// units instead of being a threshold bolted on afterwards.
#[inline(always)]
fn locality_penalty(tx: i32, ty: i32, sx: i32, sy: i32, reach_sq: f32, worst: f32) -> f32 {
    let dx = (sx - tx) as f32;
    let dy = (sy - ty) as f32;
    LOCALITY * worst * (dx * dx + dy * dy) / reach_sq
}

/// Sum of squared differences between a patch of the guide and one of the photo.
///
/// `best` lets the loop give up as soon as a candidate is already worse than
/// what is held, which is most of them.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn patch_cost(
    guide_raw: &[u8],
    image_raw: &[u8],
    mask_raw: &[u8],
    w: i32,
    h: i32,
    patch: i32,
    tx: i32,
    ty: i32,
    sx: i32,
    sy: i32,
    best: f32,
) -> f32 {
    let mut total = 0f32;
    let stride_rgba = (w * 4) as usize;
    let stride_mask = w as usize;
    const HOLE_PENALTY: f32 = 3.0 * 255.0 * 255.0;

    for dy in -patch..=patch {
        let gy = (ty + dy).clamp(0, h - 1) as usize;
        let iy = (sy + dy).clamp(0, h - 1) as usize;
        let g_row = gy * stride_rgba;
        let i_row = iy * stride_rgba;
        let mask_row = iy * stride_mask;

        for dx in -patch..=patch {
            let gx = (tx + dx).clamp(0, w - 1) as usize;
            let ix = (sx + dx).clamp(0, w - 1) as usize;

            // Source pixels inside the hole carry no information: charging a
            // large penalty keeps the search out of the region being rebuilt.
            if mask_raw[mask_row + ix] > 0 {
                total += HOLE_PENALTY;
                if total >= best {
                    return total;
                }
                continue;
            }

            let g_idx = g_row + gx * 4;
            let i_idx = i_row + ix * 4;

            let dr = guide_raw[g_idx] as f32 - image_raw[i_idx] as f32;
            let dg = guide_raw[g_idx + 1] as f32 - image_raw[i_idx + 1] as f32;
            let db = guide_raw[g_idx + 2] as f32 - image_raw[i_idx + 2] as f32;

            total += dr * dr + dg * dg + db * db;
            if total >= best {
                return total;
            }
        }
    }

    total
}

/// Coordinates of every pixel outside the hole, to seed the search from.
fn known_pixels(mask: &GrayImage) -> Vec<(u32, u32)> {
    let (w, h) = mask.dimensions();
    let mut known = Vec::new();
    let raw = mask.as_raw();
    for y in 0..h {
        let row = y as usize * w as usize;
        for x in 0..w {
            if raw[row + x as usize] == 0 {
                known.push((x, y));
            }
        }
    }
    known
}

/// For each pixel of the hole, where in the photograph it looks most like.
///
/// Kept separately from the picture it was computed on, because the search and
/// the copy do not have to happen at the same resolution — see [`apply_detail`].
pub struct Correspondence {
    width: u32,
    height: u32,
    /// Row-major over the search grid; meaningless outside the hole, and empty
    /// when there was nothing to search.
    offset: Vec<(u32, u32)>,
}

impl Correspondence {
    /// Where a pixel of a `width` × `height` picture should take its detail
    /// from, given that the search ran on a possibly smaller grid.
    ///
    /// The match is carried up by scaling the offset and keeping whatever
    /// place the pixel held inside its own cell, so a cell of the grid maps
    /// onto the *corresponding cell* of the full-size photograph rather than
    /// onto a single pixel of it. The detail that comes out is therefore at
    /// the photograph's resolution and not the grid's, which is the whole
    /// reason the search is allowed to run small.
    ///
    /// Staying inside the cell is also what keeps the answer honest: the grid
    /// mask is a maximum-reduction of the full-size one, so a cell it calls
    /// known contains no hole pixels at all, and nothing copied out of it can
    /// be part of the object being removed.
    fn lookup(&self, x: u32, y: u32, width: u32, height: u32) -> (u32, u32) {
        if (self.width, self.height) == (width, height) {
            return self.offset[(y * self.width + x) as usize];
        }

        let cell_w = width as f32 / self.width as f32;
        let cell_h = height as f32 / self.height as f32;
        let fx = ((x as f32 / cell_w) as u32).min(self.width - 1);
        let fy = ((y as f32 / cell_h) as u32).min(self.height - 1);
        let (ox, oy) = self.offset[(fy * self.width + fx) as usize];

        let inside_x = x as f32 - fx as f32 * cell_w;
        let inside_y = y as f32 - fy as f32 * cell_h;
        let sx = (ox as f32 * cell_w + inside_x) as u32;
        let sy = (oy as f32 * cell_h + inside_y) as u32;
        (sx.min(width - 1), sy.min(height - 1))
    }
}

/// Finds, for every pixel of the hole, the most similar real place in the
/// picture — matched against the guide, which stands in for what the hole
/// should have looked like.
///
/// This is the expensive half of the detail transfer and the half that does
/// not need full resolution: finding *where* a piece of hedge lives is a
/// question about structure, and structure is what survives being reduced.
pub fn correspondence(image: &RgbaImage, guide: &RgbaImage, mask: &GrayImage) -> Correspondence {
    let (w, h) = image.dimensions();
    let nothing = Correspondence { width: w, height: h, offset: Vec::new() };

    let known = known_pixels(mask);
    if known.is_empty() {
        return nothing;
    }

    let mask_raw = mask.as_raw();
    let hole: Vec<(u32, u32)> = {
        let mut v = Vec::new();
        for y in 0..h {
            let row = y as usize * w as usize;
            for x in 0..w {
                if mask_raw[row + x as usize] > 0 {
                    v.push((x, y));
                }
            }
        }
        v
    };
    if hole.is_empty() {
        return nothing;
    }

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let wi = w as usize;
    let (w_i32, h_i32) = (w as i32, h as i32);

    let guide_raw = guide.as_raw();
    let image_raw = image.as_raw();

    let patch = patch_radius(w, h);
    let worst = worst_cost(patch);

    // How far the search ranges. It used to be the whole picture, which sounds
    // generous and is not: with the window as wide as the frame, the best match
    // for a shaded gap in a hedge is quite often a patch of sky.
    let reach = ((w.max(h) as f32 * SEARCH_FRACTION) as i32).max(8);
    let reach_sq = (reach * reach) as f32;

    // Nearest-neighbour field: for each hole pixel, where in the photo it looks
    // most like. Seeded beside the hole, then improved by propagation.
    let mut offset = vec![(0u32, 0u32); (w * h) as usize];
    let mut cost = vec![f32::MAX; (w * h) as usize];

    for &(x, y) in &hole {
        // Start from somewhere nearby if at all possible. PatchMatch improves
        // the guess it is given rather than searching from nothing, and its
        // random search then explores around wherever that guess landed — so a
        // seed on the far side of the picture is not a neutral starting point,
        // it is a place the pixel tends to stay.
        let mut start = None;
        for _ in 0..SEED_TRIES {
            let sx = x as i32 + rng.below((reach * 2) as usize) as i32 - reach;
            let sy = y as i32 + rng.below((reach * 2) as usize) as i32 - reach;
            if sx >= 0
                && sy >= 0
                && sx < w_i32
                && sy < h_i32
                && mask_raw[sy as usize * wi + sx as usize] == 0
            {
                start = Some((sx as u32, sy as u32));
                break;
            }
        }
        let (sx, sy) = start.unwrap_or_else(|| known[rng.below(known.len())]);

        let index = y as usize * wi + x as usize;
        offset[index] = (sx, sy);
        cost[index] = locality_penalty(x as i32, y as i32, sx as i32, sy as i32, reach_sq, worst)
            + patch_cost(
                guide_raw, image_raw, mask_raw, w_i32, h_i32, patch, x as i32, y as i32, sx as i32,
                sy as i32, f32::MAX,
            );
    }

    for iteration in 0..ITERATIONS {
        // Alternating sweep direction: propagation carries a good match along
        // the scan, so reversing each pass lets it travel both ways.
        let forward = iteration % 2 == 0;
        let order: Vec<&(u32, u32)> = if forward {
            hole.iter().collect()
        } else {
            hole.iter().rev().collect()
        };

        for &&(x, y) in &order {
            let index = y as usize * wi + x as usize;
            let step: i32 = if forward { -1 } else { 1 };

            // Propagation: a neighbour's match, shifted by one, is usually a
            // good match here too. This is what makes PatchMatch fast.
            for (nx, ny) in [(x as i32 + step, y as i32), (x as i32, y as i32 + step)] {
                if nx < 0 || ny < 0 || nx >= w_i32 || ny >= h_i32 {
                    continue;
                }
                let neighbour = ny as usize * wi + nx as usize;
                if cost[neighbour] == f32::MAX {
                    continue;
                }

                let (ox, oy) = offset[neighbour];
                let cx = ox as i32 - step * ((nx == x as i32 + step) as i32);
                let cy = oy as i32 - step * ((ny == y as i32 + step) as i32);
                if cx < 0 || cy < 0 || cx >= w_i32 || cy >= h_i32 {
                    continue;
                }
                if mask_raw[cy as usize * wi + cx as usize] > 0 {
                    continue;
                }

                let penalty =
                    locality_penalty(x as i32, y as i32, cx, cy, reach_sq, worst);
                if penalty >= cost[index] {
                    continue;
                }
                let candidate = penalty
                    + patch_cost(
                        guide_raw, image_raw, mask_raw, w_i32, h_i32, patch, x as i32, y as i32,
                        cx, cy, cost[index] - penalty,
                    );
                if candidate < cost[index] {
                    cost[index] = candidate;
                    offset[index] = (cx as u32, cy as u32);
                }
            }

            // Random search: halving windows around the current best, which is
            // what lets it escape a local minimum without scanning the image.
            let (mut bx, mut by) = offset[index];
            let mut radius = reach;
            while radius > 1 {
                let rx = bx as i32 + (rng.below((radius * 2) as usize) as i32 - radius);
                let ry = by as i32 + (rng.below((radius * 2) as usize) as i32 - radius);
                radius /= 2;

                if rx < 0 || ry < 0 || rx >= w_i32 || ry >= h_i32 {
                    continue;
                }
                if mask_raw[ry as usize * wi + rx as usize] > 0 {
                    continue;
                }

                let penalty =
                    locality_penalty(x as i32, y as i32, rx, ry, reach_sq, worst);
                if penalty >= cost[index] {
                    continue;
                }
                let candidate = penalty
                    + patch_cost(
                        guide_raw, image_raw, mask_raw, w_i32, h_i32, patch, x as i32, y as i32,
                        rx, ry, cost[index] - penalty,
                    );
                if candidate < cost[index] {
                    cost[index] = candidate;
                    offset[index] = (rx as u32, ry as u32);
                    bx = rx as u32;
                    by = ry as u32;
                }
            }
        }
    }

    Correspondence { width: w, height: h, offset }
}

/// Puts the photograph's own detail into the hole, following a correspondence.
///
/// `image`, `guide` and `mask` are all at the resolution of the answer; `field`
/// may have been computed on a smaller grid. Detail is what a pixel has beyond
/// its own neighbourhood, so taking the source's detail and the guide's
/// neighbourhood keeps the texture real and the lighting the inpainter's —
/// which is the point of the whole exercise.
pub fn apply_detail(
    image: &RgbaImage,
    guide: &RgbaImage,
    mask: &GrayImage,
    field: &Correspondence,
) -> RgbaImage {
    if field.offset.is_empty() {
        return guide.clone();
    }

    let (w, h) = image.dimensions();
    let image_low = blur(image, DETAIL_RADIUS);
    let guide_low = blur(guide, DETAIL_RADIUS);
    let mask_raw = mask.as_raw();

    let mut out = guide.clone();
    for y in 0..h {
        let row = y as usize * w as usize;
        for x in 0..w {
            if mask_raw[row + x as usize] == 0 {
                continue;
            }

            let (sx, sy) = field.lookup(x, y, w, h);
            let source = image.get_pixel(sx, sy);
            let source_low = image_low.get_pixel(sx, sy);
            let target_low = guide_low.get_pixel(x, y);

            let px = out.get_pixel_mut(x, y);
            for c in 0..3 {
                let detail = source[c] as f32 - source_low[c] as f32;
                px[c] = (target_low[c] as f32 + detail).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

/// Search and copy at one and the same resolution.
pub fn transfer_detail(image: &RgbaImage, guide: &RgbaImage, mask: &GrayImage) -> RgbaImage {
    let field = correspondence(image, guide, mask);
    apply_detail(image, guide, mask, &field)
}

/// Whether there is a texture here to copy, or only grain.
pub fn worth_synthesising(image: &RgbaImage, mask: &GrayImage) -> bool {
    texture_energy(image, mask) >= TEXTURE_FLOOR
}

/// Puts the photograph's own grain into a hole that has no texture to copy.
///
/// The detail transfer needs somewhere to copy detail *from*, and on a smooth
/// surface — asphalt, a wall, a clear sky — there is none to find. What a
/// photograph always has is grain, and a hole filled with a smooth answer
/// magnified several times has none of it. That is what makes a repair read as
/// a soft patch in an otherwise sharp picture: not that the colours are wrong,
/// but that the fine noise everything else carries stops at its edge.
///
/// So the residual is borrowed from a few pixels away with no search at all,
/// because grain has no structure to match — only statistics, and those are
/// the same a few pixels away. It is clamped, because a residual larger than
/// grain is an edge, and printing the ghost of a kerb across the middle of a
/// road would be a worse fault than the softness this is fixing.
pub fn transfer_grain(image: &RgbaImage, guide: &RgbaImage, mask: &GrayImage) -> RgbaImage {
    let (w, h) = image.dimensions();
    let mut out = guide.clone();
    if w == 0 || h == 0 || guide.dimensions() != (w, h) || mask.dimensions() != (w, h) {
        return out;
    }

    let low = blur(image, DETAIL_RADIUS);
    let mask_raw = mask.as_raw();
    let stride = w as usize;
    let mut rng = Rng(0x5DEE_CE66_D1B1_4A5D);

    for y in 0..h {
        let row = y as usize * stride;
        for x in 0..w {
            if mask_raw[row + x as usize] == 0 {
                continue;
            }

            let mut borrowed = None;
            for _ in 0..GRAIN_TRIES {
                let sx = x as i32 + rng.below((GRAIN_REACH * 2) as usize) as i32 - GRAIN_REACH;
                let sy = y as i32 + rng.below((GRAIN_REACH * 2) as usize) as i32 - GRAIN_REACH;
                if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    continue;
                }
                if mask_raw[sy as usize * stride + sx as usize] > 0 {
                    continue;
                }
                borrowed = Some((sx as u32, sy as u32));
                break;
            }
            // Deep inside a large hole there may be nothing near enough to
            // borrow from. Leaving those pixels as they are is right: the
            // alternative is grain fetched from so far off that it belongs to
            // another surface.
            let Some((sx, sy)) = borrowed else { continue };

            let source = image.get_pixel(sx, sy);
            let source_low = low.get_pixel(sx, sy);
            let pixel = out.get_pixel_mut(x, y);
            for c in 0..3 {
                let grain = (source[c] as f32 - source_low[c] as f32)
                    .clamp(-GRAIN_CEILING, GRAIN_CEILING);
                pixel[c] = (pixel[c] as f32 + grain).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    /// A checkerboard: high-frequency texture with an obvious right answer.
    fn checkers(w: u32, h: u32) -> RgbaImage {
        let mut image = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let on = (x / 3 + y / 3) % 2 == 0;
                let v = if on { 220 } else { 40 };
                image.put_pixel(x, y, Rgba([v, v, v, 255]));
            }
        }
        image
    }

    fn hole(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> GrayImage {
        let mut mask = GrayImage::new(w, h);
        for y in y0..y1 {
            for x in x0..x1 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        mask
    }

    #[test]
    fn texture_energy_separates_a_pattern_from_a_flat_wall() {
        let flat = RgbaImage::from_pixel(32, 32, Rgba([128, 128, 128, 255]));
        let empty = GrayImage::new(32, 32);

        assert!(texture_energy(&flat, &empty) < 0.5, "a flat image has no texture");
        assert!(
            texture_energy(&checkers(32, 32), &empty) > TEXTURE_FLOOR,
            "a checkerboard is unmistakably textured"
        );
    }

    #[test]
    fn the_router_only_synthesises_where_there_is_texture_to_copy() {
        let empty = GrayImage::new(32, 32);
        assert!(worth_synthesising(&checkers(32, 32), &empty));
        assert!(!worth_synthesising(
            &RgbaImage::from_pixel(32, 32, Rgba([90, 110, 130, 255])),
            &empty
        ));
    }

    #[test]
    fn transfer_puts_contrast_back_where_the_guide_had_none() {
        let image = checkers(48, 48);
        let mask = hole(48, 48, 20, 20, 28, 28);

        // The guide is what an inpainter produces: the local average, flat.
        let mut guide = image.clone();
        for y in 20..28 {
            for x in 20..28 {
                guide.put_pixel(x, y, Rgba([130, 130, 130, 255]));
            }
        }

        let filled = transfer_detail(&image, &guide, &mask);

        let spread = |img: &RgbaImage| {
            let mut lo = 255i32;
            let mut hi = 0i32;
            for y in 21..27 {
                for x in 21..27 {
                    let v = img.get_pixel(x, y)[0] as i32;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            hi - lo
        };

        assert_eq!(spread(&guide), 0, "the guide is flat by construction");
        assert!(
            spread(&filled) > 60,
            "the hole must come back with real contrast, got {}",
            spread(&filled)
        );
    }

    #[test]
    fn everything_outside_the_hole_is_left_exactly_as_it_was() {
        let image = checkers(40, 40);
        let mask = hole(40, 40, 15, 15, 25, 25);
        let guide = image.clone();

        let filled = transfer_detail(&image, &guide, &mask);

        for y in 0..40 {
            for x in 0..40 {
                if mask.get_pixel(x, y)[0] > 0 {
                    continue;
                }
                assert_eq!(
                    filled.get_pixel(x, y),
                    guide.get_pixel(x, y),
                    "pixel ({x}, {y}) outside the hole was altered"
                );
            }
        }
    }

    #[test]
    fn the_result_is_the_same_every_time() {
        let image = checkers(40, 40);
        let mask = hole(40, 40, 16, 16, 24, 24);
        let guide = RgbaImage::from_pixel(40, 40, Rgba([130, 130, 130, 255]));

        let first = transfer_detail(&image, &guide, &mask);
        let second = transfer_detail(&image, &guide, &mask);
        assert_eq!(first.as_raw(), second.as_raw());
    }

    /// Difference between the darkest and brightest pixel of a square.
    fn spread(img: &RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32) -> i32 {
        let mut lo = 255i32;
        let mut hi = 0i32;
        for y in y0..y1 {
            for x in x0..x1 {
                let v = img.get_pixel(x, y)[0] as i32;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        hi - lo
    }

    #[test]
    fn a_carried_match_stays_inside_the_cell_it_was_found_in() {
        // This is the safety property the whole arrangement rests on. The
        // search grid's mask is a maximum-reduction of the full-size one, so a
        // cell it calls known contains no hole pixels whatsoever — and a match
        // carried up must therefore land inside that same cell, or it could
        // pick up part of the object being removed.
        //
        // Grid of 4 × 4 over a picture of 64 × 64: cells of sixteen pixels.
        let field = Correspondence { width: 4, height: 4, offset: vec![(2, 3); 16] };

        for y in 0..64u32 {
            for x in 0..64u32 {
                let (sx, sy) = field.lookup(x, y, 64, 64);
                assert!((32..48).contains(&sx), "({x}, {y}) -> x {sx} left cell 2");
                assert!((48..64).contains(&sy), "({x}, {y}) -> y {sy} left cell 3");
            }
        }
    }

    #[test]
    fn the_detail_arrives_at_the_pictures_resolution_not_the_grids() {
        use image::imageops::FilterType;

        // A checkerboard far too fine for the search grid to represent: at a
        // quarter size it averages away to flat grey, which is precisely what
        // happened to the hedge.
        let image = checkers(128, 128);
        let mask = hole(128, 128, 48, 48, 80, 80);
        let mut guide = image.clone();
        for y in 48..80 {
            for x in 48..80 {
                guide.put_pixel(x, y, Rgba([130, 130, 130, 255]));
            }
        }

        let small_image = image::imageops::resize(&image, 32, 32, FilterType::Triangle);
        let small_guide = image::imageops::resize(&guide, 32, 32, FilterType::Triangle);
        let small_mask = image::imageops::resize(&mask, 32, 32, FilterType::Nearest);

        // Search on the grid, copy at full size.
        let field = correspondence(&small_image, &small_guide, &small_mask);
        let filled = apply_detail(&image, &guide, &mask, &field);

        // What the pipeline used to do: transfer on the grid, then magnify.
        let magnified = image::imageops::resize(
            &transfer_detail(&small_image, &small_guide, &small_mask),
            128,
            128,
            FilterType::Triangle,
        );

        let carried = spread(&filled, 52, 52, 76, 76);
        let smeared = spread(&magnified, 52, 52, 76, 76);
        assert!(carried > 120, "the hole must come back sharp, got {carried}");
        assert!(
            smeared < carried,
            "magnifying the grid's answer must be the softer one: {smeared} vs {carried}"
        );
    }

    #[test]
    fn matches_are_taken_from_beside_the_hole_rather_than_across_the_picture() {
        // The same stripes everywhere, so every candidate ties on appearance
        // and only the distance is left to decide. Without a locality penalty
        // the field scatters across the frame; with one it stays home.
        let mut image = RgbaImage::new(512, 48);
        for (x, _y, px) in image.enumerate_pixels_mut() {
            let v = if (x / 4) % 2 == 0 { 210 } else { 50 };
            *px = Rgba([v, v, v, 255]);
        }
        let mask = hole(512, 48, 240, 12, 272, 36);
        let guide = image.clone();

        let field = correspondence(&image, &guide, &mask);

        let reach = (512.0 * SEARCH_FRACTION) as u32;
        for y in 12..36u32 {
            for x in 240..272u32 {
                let (sx, _) = field.offset[(y * 512 + x) as usize];
                assert!(
                    sx.abs_diff(x) <= reach,
                    "({x}, {y}) matched at {sx}, beyond the search reach of {reach}"
                );
            }
        }
    }

    #[test]
    fn the_patch_grows_with_the_grid_it_is_measured_on() {
        assert_eq!(patch_radius(256, 256), MIN_PATCH, "floored on a small grid");
        assert_eq!(patch_radius(768, 768), 6);
        assert_eq!(patch_radius(4000, 4000), MAX_PATCH, "capped on a large one");
    }

    /// Flat grey carrying fine noise, the way a film scan does.
    fn grainy(w: u32, h: u32, level: u8, amplitude: i32) -> RgbaImage {
        let mut image = RgbaImage::new(w, h);
        let mut seed = 0x1234_5678u32;
        for px in image.pixels_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 24) as i32 % (amplitude * 2 + 1) - amplitude;
            let v = (level as i32 + noise).clamp(0, 255) as u8;
            *px = Rgba([v, v, v, 255]);
        }
        image
    }

    #[test]
    fn grain_goes_into_a_hole_that_has_none() {
        let image = grainy(96, 96, 130, 7);
        let mask = hole(96, 96, 32, 32, 64, 64);
        // What the inpainter hands back on a smooth surface: the right colour
        // and nothing else.
        let guide = RgbaImage::from_pixel(96, 96, Rgba([130, 130, 130, 255]));

        let filled = transfer_grain(&image, &guide, &mask);

        let varies = |img: &RgbaImage| {
            let mut lo = 255i32;
            let mut hi = 0i32;
            for y in 40..56 {
                for x in 40..56 {
                    let v = img.get_pixel(x, y)[0] as i32;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            hi - lo
        };

        assert_eq!(varies(&guide), 0, "the guide is flat by construction");
        assert!(varies(&filled) >= 6, "the hole must come back grainy, got {}", varies(&filled));
    }

    #[test]
    fn grain_leaves_everything_outside_the_hole_alone() {
        let image = grainy(64, 64, 120, 6);
        let mask = hole(64, 64, 20, 20, 40, 40);
        let guide = image.clone();

        let filled = transfer_grain(&image, &guide, &mask);

        for y in 0..64 {
            for x in 0..64 {
                if mask.get_pixel(x, y)[0] > 0 {
                    continue;
                }
                assert_eq!(filled.get_pixel(x, y), guide.get_pixel(x, y), "({x}, {y}) altered");
            }
        }
    }

    #[test]
    fn an_edge_beside_the_hole_is_not_printed_into_it() {
        // Half the picture black, half white: the residual across that edge is
        // enormous, and copying it would stamp a ghost of the edge inside the
        // hole. The clamp is what stops it.
        let mut image = RgbaImage::new(96, 96);
        for (x, _y, px) in image.enumerate_pixels_mut() {
            let v = if x < 48 { 10 } else { 245 };
            *px = Rgba([v, v, v, 255]);
        }
        let mask = hole(96, 96, 36, 36, 60, 60);
        let guide = RgbaImage::from_pixel(96, 96, Rgba([128, 128, 128, 255]));

        let filled = transfer_grain(&image, &guide, &mask);

        for y in 36..60u32 {
            for x in 36..60u32 {
                let v = filled.get_pixel(x, y)[0] as i32;
                assert!(
                    (v - 128).abs() <= GRAIN_CEILING as i32 + 1,
                    "({x}, {y}) moved to {v}: an edge was copied in, not grain"
                );
            }
        }
    }

    #[test]
    fn a_mask_with_nothing_known_around_it_returns_the_guide_untouched() {
        let image = checkers(16, 16);
        let mask = GrayImage::from_pixel(16, 16, Luma([255]));
        let guide = RgbaImage::from_pixel(16, 16, Rgba([10, 20, 30, 255]));

        let filled = transfer_detail(&image, &guide, &mask);
        assert_eq!(filled.as_raw(), guide.as_raw());
    }
}
