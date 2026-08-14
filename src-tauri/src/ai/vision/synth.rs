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

/// Half-width of the patch compared when looking for a match.
const PATCH: i32 = 4;

/// Sweeps of propagation and random search. Three is where it stops improving
/// visibly on the sizes this runs at.
const ITERATIONS: u32 = 3;

/// Radius, in pixels, of the blur that separates structure from detail.
const DETAIL_RADIUS: u32 = 3;

/// Below this much high-frequency energy the surroundings are not a texture,
/// and copying detail from them would add noise rather than detail.
const TEXTURE_FLOOR: f32 = 2.6;

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

/// Sum of squared differences between a patch of the guide and one of the photo.
///
/// `best` lets the loop give up as soon as a candidate is already worse than
/// what is held, which is most of them.
#[inline(always)]
fn patch_cost(
    guide_raw: &[u8],
    image_raw: &[u8],
    mask_raw: &[u8],
    w: i32,
    h: i32,
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

    for dy in -PATCH..=PATCH {
        let gy = (ty + dy).clamp(0, h - 1) as usize;
        let iy = (sy + dy).clamp(0, h - 1) as usize;
        let g_row = gy * stride_rgba;
        let i_row = iy * stride_rgba;
        let mask_row = iy * stride_mask;

        for dx in -PATCH..=PATCH {
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

/// Replaces the smooth fill inside the hole with real texture from the photo.
///
/// `guide` is the inpainter's output at the same size; `mask` is 255 inside the
/// hole. Returns a new image where the hole keeps the guide's *lighting and
/// structure* and takes its *detail* from the nearest matching real place.
pub fn transfer_detail(
    image: &RgbaImage,
    guide: &RgbaImage,
    mask: &GrayImage,
) -> RgbaImage {
    let (w, h) = image.dimensions();
    let known = known_pixels(mask);
    if known.is_empty() {
        return guide.clone();
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
        return guide.clone();
    }

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let wi = w as usize;
    let (w_i32, h_i32) = (w as i32, h as i32);

    let guide_raw = guide.as_raw();
    let image_raw = image.as_raw();

    // Nearest-neighbour field: for each hole pixel, where in the photo it looks
    // most like. Seeded at random, then improved by propagation.
    let mut offset = vec![(0u32, 0u32); (w * h) as usize];
    let mut cost = vec![f32::MAX; (w * h) as usize];

    for &(x, y) in &hole {
        let (sx, sy) = known[rng.below(known.len())];
        let index = y as usize * wi + x as usize;
        offset[index] = (sx, sy);
        cost[index] = patch_cost(
            guide_raw, image_raw, mask_raw, w_i32, h_i32, x as i32, y as i32, sx as i32, sy as i32, f32::MAX,
        );
    }

    let max_search = w.max(h) as i32;

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

                let candidate =
                    patch_cost(guide_raw, image_raw, mask_raw, w_i32, h_i32, x as i32, y as i32, cx, cy, cost[index]);
                if candidate < cost[index] {
                    cost[index] = candidate;
                    offset[index] = (cx as u32, cy as u32);
                }
            }

            // Random search: halving windows around the current best, which is
            // what lets it escape a local minimum without scanning the image.
            let (mut bx, mut by) = offset[index];
            let mut radius = max_search;
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

                let candidate =
                    patch_cost(guide_raw, image_raw, mask_raw, w_i32, h_i32, x as i32, y as i32, rx, ry, cost[index]);
                if candidate < cost[index] {
                    cost[index] = candidate;
                    offset[index] = (rx as u32, ry as u32);
                    bx = rx as u32;
                    by = ry as u32;
                }
            }
        }
    }

    // Detail is what a pixel has beyond its own neighbourhood. Taking the
    // source's detail and the guide's neighbourhood keeps the texture real and
    // the lighting the inpainter's, which is the point of the whole exercise.
    let image_low = blur(image, DETAIL_RADIUS);
    let guide_low = blur(guide, DETAIL_RADIUS);

    let mut out = guide.clone();
    for &(x, y) in &hole {
        let (sx, sy) = offset[y as usize * wi + x as usize];
        let source = image.get_pixel(sx, sy);
        let source_low = image_low.get_pixel(sx, sy);
        let target_low = guide_low.get_pixel(x, y);

        let px = out.get_pixel_mut(x, y);
        for c in 0..3 {
            let detail = source[c] as f32 - source_low[c] as f32;
            px[c] = (target_low[c] as f32 + detail).round().clamp(0.0, 255.0) as u8;
        }
    }

    out
}

/// Whether copying detail would help here, or only add noise.
pub fn worth_synthesising(image: &RgbaImage, mask: &GrayImage) -> bool {
    texture_energy(image, mask) >= TEXTURE_FLOOR
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

    #[test]
    fn a_mask_with_nothing_known_around_it_returns_the_guide_untouched() {
        let image = checkers(16, 16);
        let mask = GrayImage::from_pixel(16, 16, Luma([255]));
        let guide = RgbaImage::from_pixel(16, 16, Rgba([10, 20, 30, 255]));

        let filled = transfer_detail(&image, &guide, &mask);
        assert_eq!(filled.as_raw(), guide.as_raw());
    }
}
