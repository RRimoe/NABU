//! Multi-band blending (Burt & Adelson 1983).
//!
//! Each photo is split into frequency bands. Low frequencies (the sky's
//! gradient, a lingering exposure difference) are blended over a wide area so
//! no step is visible; high frequencies (leaves, text, edges) are blended over
//! a pixel or two so nothing is ghosted or softened. A plain cross-fade has to
//! pick one of the two and gets the other wrong.

use crate::warp::Rect;
use rayon::prelude::*;

const WEIGHT_EPS: f32 = 1e-5;

#[derive(Clone)]
struct Level {
    width: usize,
    height: usize,
    rgb: Vec<[f32; 3]>,
    weight: Vec<f32>,
}

pub struct Blender {
    canvas: Rect,
    /// Canvas size rounded up so every pyramid level divides exactly.
    width: usize,
    height: usize,
    bands: usize,
    levels: Vec<Level>,
}

#[inline]
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    if i < 0 {
        i = -i;
    }
    if i >= n {
        i = 2 * n - 2 - i;
    }
    i.clamp(0, n - 1) as usize
}

trait Pixel: Copy + Send + Sync + Default {
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, k: f32) -> Self;
}

impl Pixel for f32 {
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, k: f32) -> Self {
        self * k
    }
}

impl Pixel for [f32; 3] {
    fn add(self, o: Self) -> Self {
        [self[0] + o[0], self[1] + o[1], self[2] + o[2]]
    }
    fn sub(self, o: Self) -> Self {
        [self[0] - o[0], self[1] - o[1], self[2] - o[2]]
    }
    fn mul(self, k: f32) -> Self {
        [self[0] * k, self[1] * k, self[2] * k]
    }
}

/// Gaussian [1 4 6 4 1]/16 filter then every other pixel.
fn pyr_down<P: Pixel>(src: &[P], w: usize, h: usize) -> (Vec<P>, usize, usize) {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
    // Horizontal pass, decimated.
    let mut tmp = vec![P::default(); dw * h];
    tmp.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..(y + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            let cx = (2 * x) as isize;
            let mut acc = P::default();
            for (t, k) in K.iter().enumerate() {
                acc = acc.add(s[reflect(cx + t as isize - 2, w)].mul(*k));
            }
            *o = acc;
        }
    });
    let mut out = vec![P::default(); dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let cy = (2 * y) as isize;
        for (t, k) in K.iter().enumerate() {
            let s = &tmp[reflect(cy + t as isize - 2, h) * dw..][..dw];
            for (o, v) in row.iter_mut().zip(s) {
                *o = o.add(v.mul(*k));
            }
        }
    });
    (out, dw, dh)
}

/// Upsample to (w, h) by the transpose of `pyr_down`.
fn pyr_up<P: Pixel>(src: &[P], sw: usize, sh: usize, w: usize, h: usize) -> Vec<P> {
    let up_line = |get: &dyn Fn(usize) -> P, n_src: usize, n: usize, out: &mut [P]| {
        for (x, o) in out.iter_mut().enumerate().take(n) {
            let i = x / 2;
            let at = |k: isize| get(reflect(k, n_src));
            *o = if x % 2 == 0 {
                at(i as isize - 1).mul(0.125).add(at(i as isize).mul(0.75)).add(at(i as isize + 1).mul(0.125))
            } else {
                at(i as isize).mul(0.5).add(at(i as isize + 1).mul(0.5))
            };
        }
    };
    let mut tmp = vec![P::default(); w * sh];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * sw..(y + 1) * sw];
        up_line(&|i| s[i], sw, w, row);
    });
    let mut out = vec![P::default(); w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let i = y / 2;
        if y % 2 == 0 {
            let (a, b, c) = (reflect(i as isize - 1, sh), i.min(sh - 1), reflect(i as isize + 1, sh));
            for x in 0..w {
                row[x] = tmp[a * w + x].mul(0.125).add(tmp[b * w + x].mul(0.75)).add(tmp[c * w + x].mul(0.125));
            }
        } else {
            let (a, b) = (i.min(sh - 1), reflect(i as isize + 1, sh));
            for x in 0..w {
                row[x] = tmp[a * w + x].mul(0.5).add(tmp[b * w + x].mul(0.5));
            }
        }
    });
    out
}

/// Fills pixels outside `valid` by pull-push interpolation from their
/// surroundings, so the band-pass filters see a plausible continuation at a
/// photo's edge instead of a cliff down to black.
fn fill_holes(rgb: &mut [[f32; 3]], valid: &[bool], w: usize, h: usize) {
    if valid.iter().all(|v| *v) || !valid.iter().any(|v| *v) {
        return;
    }
    // Pull: premultiplied averages down a pyramid.
    let mut levels: Vec<(Vec<[f32; 4]>, usize, usize)> = Vec::new();
    let base: Vec<[f32; 4]> = rgb
        .iter()
        .zip(valid)
        .map(|(p, v)| if *v { [p[0], p[1], p[2], 1.0] } else { [0.0; 4] })
        .collect();
    levels.push((base, w, h));
    while {
        let (_, lw, lh) = levels.last().unwrap();
        *lw > 1 || *lh > 1
    } {
        let (src, lw, lh) = levels.last().unwrap();
        let (lw, lh) = (*lw, *lh);
        let (dw, dh) = (lw.div_ceil(2), lh.div_ceil(2));
        let mut dst = vec![[0.0f32; 4]; dw * dh];
        dst.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let mut acc = [0.0f32; 4];
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let (sx, sy) = ((2 * x + dx).min(lw - 1), (2 * y + dy).min(lh - 1));
                    let s = src[sy * lw + sx];
                    for c in 0..4 {
                        acc[c] += s[c];
                    }
                }
                *out = acc;
            }
        });
        levels.push((dst, dw, dh));
    }
    // Push: fill each level's empty pixels from the level above.
    for k in (0..levels.len() - 1).rev() {
        let (upper, uw) = {
            let (u, uw, _) = &levels[k + 1];
            (u.clone(), *uw)
        };
        let (cur, lw, _) = &mut levels[k];
        let lw = *lw;
        cur.par_chunks_mut(lw).enumerate().for_each(|(y, row)| {
            for (x, p) in row.iter_mut().enumerate() {
                if p[3] >= 1e-6 {
                    continue;
                }
                let u = upper[(y / 2) * uw + x / 2];
                if u[3] > 1e-6 {
                    *p = [u[0] / u[3], u[1] / u[3], u[2] / u[3], 1.0];
                }
            }
        });
    }
    let (filled, _, _) = &levels[0];
    for (k, p) in rgb.iter_mut().enumerate() {
        if !valid[k] {
            let f = filled[k];
            if f[3] > 1e-6 {
                *p = [f[0] / f[3], f[1] / f[3], f[2] / f[3]];
            }
        }
    }
}

impl Blender {
    pub fn new(canvas: Rect, bands: usize) -> Self {
        let unit = 1usize << bands;
        let width = canvas.width.div_ceil(unit) * unit;
        let height = canvas.height.div_ceil(unit) * unit;
        let levels = (0..=bands)
            .map(|k| {
                let (lw, lh) = (width >> k, height >> k);
                Level { width: lw, height: lh, rgb: vec![[0.0; 3]; lw * lh], weight: vec![0.0; lw * lh] }
            })
            .collect();
        Self { canvas, width, height, bands, levels }
    }

    /// Adds one photo. `coverage` says where it has pixels at all, `mask`
    /// where it should be seen (its side of the seams).
    pub fn feed(&mut self, rgb: &[[f32; 3]], coverage: &[u8], mask: &[u8], rect: Rect) {
        let unit = 1i64 << self.bands;
        let gap = 3 * unit;
        // Window in canvas coordinates, aligned to the coarsest level.
        let tl_x = ((rect.x - self.canvas.x - gap).max(0) / unit) * unit;
        let tl_y = ((rect.y - self.canvas.y - gap).max(0) / unit) * unit;
        let br_x = ((rect.right() - self.canvas.x + gap + unit - 1) / unit * unit).min(self.width as i64);
        let br_y = ((rect.bottom() - self.canvas.y + gap + unit - 1) / unit * unit).min(self.height as i64);
        if br_x <= tl_x || br_y <= tl_y {
            return;
        }
        let (ww, wh) = ((br_x - tl_x) as usize, (br_y - tl_y) as usize);

        let mut img = vec![[0.0f32; 3]; ww * wh];
        let mut valid = vec![false; ww * wh];
        let mut weight = vec![0.0f32; ww * wh];
        let ox = rect.x - self.canvas.x - tl_x;
        let oy = rect.y - self.canvas.y - tl_y;
        img.par_chunks_mut(ww)
            .zip(valid.par_chunks_mut(ww))
            .zip(weight.par_chunks_mut(ww))
            .enumerate()
            .for_each(|(wy, ((irow, vrow), wrow))| {
                let y = wy as i64 - oy;
                if y < 0 || y >= rect.height as i64 {
                    return;
                }
                for (wx, ((ip, vp), wp)) in irow.iter_mut().zip(vrow.iter_mut()).zip(wrow.iter_mut()).enumerate() {
                    let x = wx as i64 - ox;
                    if x < 0 || x >= rect.width as i64 {
                        continue;
                    }
                    let k = y as usize * rect.width + x as usize;
                    if coverage[k] != 0 {
                        *ip = rgb[k];
                        *vp = true;
                    }
                    if mask[k] != 0 {
                        *wp = 1.0;
                    }
                }
            });
        if !weight.iter().any(|w| *w > 0.0) {
            return;
        }
        fill_holes(&mut img, &valid, ww, wh);

        // Laplacian pyramid of the photo, Gaussian pyramid of its weight.
        let mut gauss = vec![(img, ww, wh)];
        let mut weights = vec![(weight, ww, wh)];
        for _ in 0..self.bands {
            let (g, gw, gh) = gauss.last().unwrap();
            gauss.push(pyr_down(g, *gw, *gh));
            let (m, mw, mh) = weights.last().unwrap();
            weights.push(pyr_down(m, *mw, *mh));
        }
        for k in 0..=self.bands {
            let (lap, lw, lh) = if k < self.bands {
                let (g, gw, gh) = &gauss[k];
                let (n, nw, nh) = &gauss[k + 1];
                let up = pyr_up(n, *nw, *nh, *gw, *gh);
                let lap: Vec<[f32; 3]> = g.iter().zip(&up).map(|(a, b)| a.sub(*b)).collect();
                (lap, *gw, *gh)
            } else {
                let (g, gw, gh) = &gauss[k];
                (g.clone(), *gw, *gh)
            };
            let (m, _, _) = &weights[k];
            let level = &mut self.levels[k];
            let (lx, ly) = ((tl_x >> k) as usize, (tl_y >> k) as usize);
            let lwidth = level.width;
            level
                .rgb
                .par_chunks_mut(lwidth)
                .zip(level.weight.par_chunks_mut(lwidth))
                .skip(ly)
                .take(lh)
                .enumerate()
                .for_each(|(y, (row, wrow))| {
                    for x in 0..lw.min(lwidth - lx) {
                        let wv = m[y * lw + x];
                        if wv <= 0.0 {
                            continue;
                        }
                        let o = lx + x;
                        row[o] = row[o].add(lap[y * lw + x].mul(wv));
                        wrow[o] += wv;
                    }
                });
        }
    }

    /// Collapses the pyramid. Returns RGB (0..1, sRGB) and the blend weight
    /// at full resolution, both over the original canvas rectangle.
    pub fn finish(mut self) -> (Vec<[f32; 3]>, Vec<f32>) {
        for level in self.levels.iter_mut() {
            level.rgb.par_iter_mut().zip(level.weight.par_iter()).for_each(|(p, w)| {
                *p = p.mul(1.0 / (w + WEIGHT_EPS));
            });
        }
        let mut img = self.levels[self.bands].rgb.clone();
        for k in (0..self.bands).rev() {
            let (sw, sh) = (self.levels[k + 1].width, self.levels[k + 1].height);
            let (w, h) = (self.levels[k].width, self.levels[k].height);
            let up = pyr_up(&img, sw, sh, w, h);
            img = up.iter().zip(&self.levels[k].rgb).map(|(a, b)| a.add(*b)).collect();
        }
        let weight0 = std::mem::take(&mut self.levels[0].weight);
        let (cw, ch) = (self.canvas.width, self.canvas.height);
        let mut rgb = Vec::with_capacity(cw * ch);
        let mut weight = Vec::with_capacity(cw * ch);
        for y in 0..ch {
            rgb.extend_from_slice(&img[y * self.width..y * self.width + cw]);
            weight.extend_from_slice(&weight0[y * self.width..y * self.width + cw]);
        }
        (rgb, weight)
    }
}
