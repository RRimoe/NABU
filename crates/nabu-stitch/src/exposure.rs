//! Matching the brightness and colour of the photos to each other.
//!
//! Cameras re-meter between shots and white balance drifts, so the same sky
//! comes out a shade different in each photo. One gain per photo and channel,
//! in linear light, is chosen so that overlapping areas agree, while a prior
//! keeps the gains near one (otherwise "everything black" would be a perfect
//! solution). This is Brown & Lowe's gain compensation, per channel.

use crate::imgops::srgb_to_linear_table;
use crate::warp::{Rect, Warped};
use nalgebra::{DMatrix, DVector};

/// Standard deviation of the overlap error and of the gain prior.
const SIGMA_N: f64 = 10.0 / 255.0;
///
/// The prior is deliberately loose. Brown & Lowe's 0.1 was set for intensities
/// on a 0–255 scale; in linear light on 0–1 the overlap term is an order of
/// magnitude weaker, and a 0.1 prior then won the argument: two photos a stop
/// apart came out only a fifth of a stop closer. A loose prior lets the data
/// decide, and the darkening it allows (all gains drifting below one) is taken
/// out afterwards by bringing the average gain back to one.
const SIGMA_G: f64 = 1.0;

/// Gains per photo and channel (linear light), for photos warped at a low
/// resolution onto the same surface.
pub fn gains(warped: &[Warped]) -> Vec<[f32; 3]> {
    solve_gains(warped, SIGMA_G)
}

fn solve_gains(warped: &[Warped], sigma_g: f64) -> Vec<[f32; 3]> {
    let n = warped.len();
    let lut = srgb_to_linear_table();
    let to_linear = |v: f32| lut[(v * 255.0).round().clamp(0.0, 255.0) as usize] as f64;

    // Overlap statistics: count and mean intensity of each photo over the
    // part it shares with each other photo.
    let mut count = vec![vec![0.0f64; n]; n];
    let mut mean = vec![vec![[0.0f64; 3]; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&warped[i], &warped[j]);
            let Some(r) = a.rect.intersect(&b.rect) else { continue };
            let (mut c, mut sa, mut sb) = (0.0, [0.0f64; 3], [0.0f64; 3]);
            for y in 0..r.height as i64 {
                let ya = (r.y + y - a.rect.y) as usize;
                let yb = (r.y + y - b.rect.y) as usize;
                for x in 0..r.width as i64 {
                    let xa = (r.x + x - a.rect.x) as usize;
                    let xb = (r.x + x - b.rect.x) as usize;
                    let ka = ya * a.rect.width + xa;
                    let kb = yb * b.rect.width + xb;
                    if a.mask[ka] == 0 || b.mask[kb] == 0 {
                        continue;
                    }
                    let (pa, pb) = (a.rgb[ka], b.rgb[kb]);
                    // Clipped highlights carry no information about exposure.
                    if pa.iter().chain(pb.iter()).any(|v| *v > 0.98) {
                        continue;
                    }
                    c += 1.0;
                    for ch in 0..3 {
                        sa[ch] += to_linear(pa[ch]);
                        sb[ch] += to_linear(pb[ch]);
                    }
                }
            }
            if c < 1.0 {
                continue;
            }
            count[i][j] = c;
            count[j][i] = c;
            for ch in 0..3 {
                mean[i][j][ch] = sa[ch] / c;
                mean[j][i][ch] = sb[ch] / c;
            }
        }
    }

    let mut out = vec![[1.0f32; 3]; n];
    for ch in 0..3 {
        let mut a = DMatrix::<f64>::zeros(n, n);
        let mut b = DVector::<f64>::zeros(n);
        for i in 0..n {
            // A photo with no overlaps keeps a gain of one through the prior.
            a[(i, i)] += 1.0 / (sigma_g * sigma_g);
            b[i] += 1.0 / (sigma_g * sigma_g);
            for j in 0..n {
                if i == j || count[i][j] == 0.0 {
                    continue;
                }
                let nij = count[i][j];
                let (iij, iji) = (mean[i][j][ch], mean[j][i][ch]);
                a[(i, i)] += nij * (iij * iij / (SIGMA_N * SIGMA_N) + 1.0 / (sigma_g * sigma_g));
                a[(i, j)] -= nij * iij * iji / (SIGMA_N * SIGMA_N);
                b[i] += nij / (sigma_g * sigma_g);
            }
        }
        if let Some(g) = a.lu().solve(&b) {
            // Back to an average of one, weighted by area, so the panorama
            // keeps the overall exposure of the photos it was made from.
            let (mut log_sum, mut weight) = (0.0f64, 0.0f64);
            for i in 0..n {
                let area = warped[i].mask.iter().filter(|&&m| m > 0).count() as f64;
                if g[i] > 0.0 && area > 0.0 {
                    log_sum += g[i].ln() * area;
                    weight += area;
                }
            }
            let level = if weight > 0.0 { (log_sum / weight).exp() } else { 1.0 };
            for i in 0..n {
                out[i][ch] = (g[i] / level).clamp(0.2, 5.0) as f32;
            }
        }
    }
    out
}

/// Smallest side of the blocks local gains are solved on, in pixels of the
/// small copies exposure is worked out on.
const BLOCK: usize = 32;

/// Most blocks solved for at once: the solve is dense, so this bounds its
/// cost on a large set of photos (blocks grow instead).
const MAX_BLOCKS: f64 = 1200.0;

/// A smooth gain field over one warped photo: a grid of gains per channel,
/// interpolated in between.
#[derive(Clone, Debug)]
pub struct GainField {
    columns: usize,
    rows: usize,
    gains: Vec<[f32; 3]>,
}

impl GainField {
    /// The gain at a position given as fractions (0..1) of the photo's
    /// warped rectangle.
    pub fn at(&self, fx: f32, fy: f32) -> [f32; 3] {
        let gx = (fx * self.columns as f32 - 0.5).clamp(0.0, (self.columns - 1) as f32);
        let gy = (fy * self.rows as f32 - 0.5).clamp(0.0, (self.rows - 1) as f32);
        let (x0, y0) = (gx.floor() as usize, gy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.columns - 1), (y0 + 1).min(self.rows - 1));
        let (wx, wy) = (gx - x0 as f32, gy - y0 as f32);
        let g = |x: usize, y: usize| self.gains[y * self.columns + x];
        let mut out = [0.0; 3];
        for c in 0..3 {
            let top = g(x0, y0)[c] * (1.0 - wx) + g(x1, y0)[c] * wx;
            let bottom = g(x0, y1)[c] * (1.0 - wx) + g(x1, y1)[c] * wx;
            out[c] = top * (1.0 - wy) + bottom * wy;
        }
        out
    }
}

/// Gains that vary across each photo, for photos already corrected by
/// [`gains`].
///
/// One gain per photo cannot fix what changes inside a photo: vignetting
/// darkens every corner, and phones tone-map each shot locally (a sky pulled
/// down here, a shadow lifted there), so the same place comes out a different
/// brightness in each photo it appears in. Each photo is cut into blocks and
/// the blocks are matched to each other exactly as whole photos are; the
/// block gains are then smoothed into a field (Uyttendaele et al., as in
/// OpenCV's block gain compensator).
pub fn block_gains(warped: &[Warped]) -> Vec<GainField> {
    let area: f64 = warped.iter().map(|w| (w.rect.width * w.rect.height) as f64).sum();
    let side = ((area / MAX_BLOCKS).sqrt().ceil() as usize).max(BLOCK);
    let mut blocks = Vec::new();
    let mut layout = Vec::new();
    for w in warped {
        let columns = w.rect.width.div_ceil(side).max(1);
        let rows = w.rect.height.div_ceil(side).max(1);
        layout.push((columns, rows, blocks.len()));
        for by in 0..rows {
            for bx in 0..columns {
                let x0 = bx * w.rect.width / columns;
                let x1 = (bx + 1) * w.rect.width / columns;
                let y0 = by * w.rect.height / rows;
                let y1 = (by + 1) * w.rect.height / rows;
                let (bw, bh) = (x1 - x0, y1 - y0);
                let mut rgb = Vec::with_capacity(bw * bh);
                let mut mask = Vec::with_capacity(bw * bh);
                for y in y0..y1 {
                    let row = y * w.rect.width;
                    rgb.extend_from_slice(&w.rgb[row + x0..row + x1]);
                    mask.extend_from_slice(&w.mask[row + x0..row + x1]);
                }
                blocks.push(Warped {
                    rect: Rect { x: w.rect.x + x0 as i64, y: w.rect.y + y0 as i64, width: bw, height: bh },
                    rgb,
                    mask,
                });
            }
        }
    }
    let solved = solve_gains(&blocks, SIGMA_G);

    layout
        .into_iter()
        .map(|(columns, rows, first)| {
            let mut grid: Vec<[f32; 3]> = solved[first..first + columns * rows].to_vec();
            // Two passes of a 3×3 box: the field has to be smooth, or each
            // block edge becomes a visible step.
            for _ in 0..2 {
                let previous = grid.clone();
                for y in 0..rows {
                    for x in 0..columns {
                        let mut sum = [0.0f32; 3];
                        let mut count = 0.0;
                        for ny in y.saturating_sub(1)..(y + 2).min(rows) {
                            for nx in x.saturating_sub(1)..(x + 2).min(columns) {
                                for c in 0..3 {
                                    sum[c] += previous[ny * columns + nx][c];
                                }
                                count += 1.0;
                            }
                        }
                        grid[y * columns + x] = sum.map(|s| s / count);
                    }
                }
            }
            GainField { columns, rows, gains: grid }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat grey photo at `x`, with `shade(x)` multiplying its brightness.
    fn photo(x: i64, width: usize, height: usize, shade: impl Fn(usize) -> f32) -> Warped {
        let mut rgb = Vec::with_capacity(width * height);
        for _ in 0..height {
            for px in 0..width {
                let v = 0.5 * shade(px);
                rgb.push([v, v, v]);
            }
        }
        Warped { rect: Rect { x, y: 0, width, height }, rgb, mask: vec![1; width * height] }
    }

    #[test]
    fn a_shading_inside_one_photo_is_corrected_where_it_overlaps() {
        // The right photo darkens towards its left edge, as a vignette would,
        // across exactly the part it shares with the left photo.
        let left = photo(0, 256, 64, |_| 1.0);
        let shade = |x: usize| 0.7 + 0.3 * (x as f32 / 128.0).min(1.0);
        let right = photo(128, 256, 64, shade);
        let fields = block_gains(&[left, photo(128, 256, 64, shade)]);

        // Mean difference between the two photos over their overlap, in
        // linear light, with and without the fields.
        let lin = |v: f32| v.powf(2.2);
        let mismatch = |corrected: bool| {
            let mut total = 0.0f32;
            for x in 0..128 {
                let (gl, gr) = if corrected {
                    (
                        fields[0].at((128.0 + x as f32 + 0.5) / 256.0, 0.5)[0],
                        fields[1].at((x as f32 + 0.5) / 256.0, 0.5)[0],
                    )
                } else {
                    (1.0, 1.0)
                };
                total += (lin(right.rgb[32 * 256 + x][0]) * gr - lin(0.5) * gl).abs();
            }
            total / 128.0
        };
        let (before, after) = (mismatch(false), mismatch(true));
        assert!(after < before * 0.5, "{after} vs {before}");
    }

    #[test]
    fn two_photos_a_stop_apart_are_brought_together() {
        // 0.365 in sRGB is half the light of 0.5.
        let g = gains(&[photo(0, 256, 64, |_| 1.0), photo(128, 256, 64, |_| 0.73)]);
        let ratio = g[1][0] / g[0][0];
        assert!((ratio - 2.0).abs() < 0.15, "gains {g:?}");
        // And the exposure overall is kept: one brighter, one darker.
        assert!(g[0][0] < 1.0 && g[1][0] > 1.0, "gains {g:?}");
    }
}
