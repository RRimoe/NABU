//! Fine alignment on the full-resolution photos.
//!
//! Features are found on reduced copies, so their positions are only as
//! precise as those copies: on a 24-megapixel photo registered at one
//! megapixel, a quarter-pixel error becomes more than a pixel. Here every
//! match is measured again on the original pixels. A patch around the point
//! in one photo is mapped into the other through the current camera model
//! (so rotation, scale and lens distortion are already accounted for) and
//! slid into place by Lucas–Kanade on normalised intensities, which ignores
//! the exposure difference between the shots. The bundle adjustment then runs
//! again on these measurements.

use crate::align::{self, Freedom, Pair, Solution};
use crate::camera::Lens;
use image::RgbaImage;
use nalgebra::{Matrix2, Matrix3, Vector2};
use rayon::prelude::*;

const POINTS_PER_PAIR: usize = 80;
const MIN_NCC: f64 = 0.95;
const MAX_SHIFT: f64 = 6.0;

/// A small piece of a photo as smoothed luma.
///
/// Bilinear interpolation blurs by an amount that depends on the fractional
/// position (none at a pixel centre, most halfway between), which pulls
/// sub-pixel estimates towards whole pixels. Smoothing first with a Gaussian
/// of one pixel makes that difference negligible.
struct Window {
    x0: i64,
    y0: i64,
    width: usize,
    height: usize,
    data: Vec<f64>,
}

const SMOOTH: [f64; 7] = [0.00443, 0.05400, 0.24203, 0.39905, 0.24203, 0.05400, 0.00443];

impl Window {
    /// Covers pixel positions [cx - reach, cx + reach] × [cy - reach, cy + reach].
    fn new(image: &RgbaImage, cx: f64, cy: f64, reach: f64) -> Option<Window> {
        let (w, h) = (image.width() as i64, image.height() as i64);
        let pad = 4;
        let x0 = (cx - reach).floor() as i64 - pad;
        let y0 = (cy - reach).floor() as i64 - pad;
        let x1 = (cx + reach).ceil() as i64 + pad;
        let y1 = (cy + reach).ceil() as i64 + pad;
        if x0 < 0 || y0 < 0 || x1 >= w || y1 >= h {
            return None;
        }
        let (ww, wh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let raw = image.as_raw();
        let mut lum = vec![0.0; ww * wh];
        for y in 0..wh {
            for x in 0..ww {
                let k = (((y0 + y as i64) * w + x0 + x as i64) * 4) as usize;
                lum[y * ww + x] = 0.299 * raw[k] as f64 + 0.587 * raw[k + 1] as f64 + 0.114 * raw[k + 2] as f64;
            }
        }
        let mut tmp = vec![0.0; ww * wh];
        for y in 0..wh {
            for x in 0..ww {
                let mut acc = 0.0;
                for (t, k) in SMOOTH.iter().enumerate() {
                    let sx = (x as i64 + t as i64 - 3).clamp(0, ww as i64 - 1) as usize;
                    acc += k * lum[y * ww + sx];
                }
                tmp[y * ww + x] = acc;
            }
        }
        let mut data = vec![0.0; ww * wh];
        for y in 0..wh {
            for x in 0..ww {
                let mut acc = 0.0;
                for (t, k) in SMOOTH.iter().enumerate() {
                    let sy = (y as i64 + t as i64 - 3).clamp(0, wh as i64 - 1) as usize;
                    acc += k * tmp[sy * ww + x];
                }
                data[y * ww + x] = acc;
            }
        }
        Some(Window { x0, y0, width: ww, height: wh, data })
    }

    /// Bilinear sample at an image position.
    #[inline]
    fn at(&self, x: f64, y: f64) -> Option<f64> {
        let (x, y) = (x - self.x0 as f64, y - self.y0 as f64);
        if x < 0.0 || y < 0.0 || x > (self.width - 1) as f64 || y > (self.height - 1) as f64 {
            return None;
        }
        let (xi, yi) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((xi + 1).min(self.width - 1), (yi + 1).min(self.height - 1));
        let (fx, fy) = (x - xi as f64, y - yi as f64);
        let g = |xx: usize, yy: usize| self.data[yy * self.width + xx];
        let top = g(xi, yi) * (1.0 - fx) + g(x1, yi) * fx;
        let bottom = g(xi, y1) * (1.0 - fx) + g(x1, y1) * fx;
        Some(top * (1.0 - fy) + bottom * fy)
    }
}

fn normalise(v: &mut [f64]) -> Option<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if var < 1.0 {
        // A flat patch (sky) cannot be located; leave it to the others.
        return None;
    }
    let sd = var.sqrt();
    v.iter_mut().for_each(|x| *x = (*x - mean) / sd);
    Some(sd)
}

/// Locates in photo j the content around `a` (centred, full resolution) of
/// photo i. Returns the centred position in j.
#[allow(clippy::too_many_arguments)]
pub fn track(
    img_i: &RgbaImage,
    img_j: &RgbaImage,
    lens: &Lens,
    rji: &Matrix3<f64>,
    a: [f64; 2],
    radius: i64,
) -> Option<[f64; 2]> {
    let (ci, cj) = (
        [(img_i.width() as f64 - 1.0) / 2.0, (img_i.height() as f64 - 1.0) / 2.0],
        [(img_j.width() as f64 - 1.0) / 2.0, (img_j.height() as f64 - 1.0) / 2.0],
    );
    let side = (2 * radius + 1) as usize;
    let mut template = Vec::with_capacity(side * side);
    let mut targets = Vec::with_capacity(side * side);
    let centre = lens.project(&(rji * lens.ray(a)))?;
    let reach = radius as f64 * 1.5 + MAX_SHIFT + 2.0;
    let win_i = Window::new(img_i, a[0] + ci[0], a[1] + ci[1], radius as f64 + 1.0)?;
    let win_j = Window::new(img_j, centre[0] + cj[0], centre[1] + cj[1], reach)?;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let p = [a[0] + dx as f64, a[1] + dy as f64];
            template.push(win_i.at(p[0] + ci[0], p[1] + ci[1])?);
            let q = lens.project(&(rji * lens.ray(p)))?;
            targets.push([q[0] + cj[0], q[1] + cj[1]]);
        }
    }
    normalise(&mut template)?;

    let mut t = Vector2::new(0.0, 0.0);
    let mut values = vec![0.0; targets.len()];
    let mut grads = vec![[0.0; 2]; targets.len()];
    let mut ncc = 0.0;
    for _ in 0..15 {
        for (k, q) in targets.iter().enumerate() {
            let (x, y) = (q[0] + t.x, q[1] + t.y);
            values[k] = win_j.at(x, y)?;
            grads[k] = [
                (win_j.at(x + 0.5, y)? - win_j.at(x - 0.5, y)?),
                (win_j.at(x, y + 0.5)? - win_j.at(x, y - 0.5)?),
            ];
        }
        let sd = normalise(&mut values)?;
        let mut hess = Matrix2::zeros();
        let mut rhs = Vector2::zeros();
        let mut sq = 0.0;
        for k in 0..values.len() {
            let g = Vector2::new(grads[k][0] / sd, grads[k][1] / sd);
            let e = values[k] - template[k];
            sq += e * e;
            hess += g * g.transpose();
            rhs -= g * e;
        }
        ncc = 1.0 - sq / (2.0 * values.len() as f64);
        // An edge (one strong direction) slides along itself: unusable.
        let eig = hess.symmetric_eigen();
        if eig.eigenvalues.min() < 1e-3 * eig.eigenvalues.max() {
            return None;
        }
        let step = hess.try_inverse()? * rhs;
        t += step;
        if t.norm() > MAX_SHIFT {
            return None;
        }
        if step.norm() < 0.005 {
            break;
        }
    }
    if ncc < MIN_NCC {
        return None;
    }
    Some([centre[0] + t.x, centre[1] + t.y])
}

/// Re-measures the matches at full resolution and re-solves. `lens` and
/// `rotations` are the current solution in full-resolution pixels; `pairs`
/// hold points at `work_scale`.
#[allow(clippy::too_many_arguments)]
pub fn refine(
    images: &[RgbaImage],
    lens: &Lens,
    rotations: &[Matrix3<f64>],
    pairs: &[Pair],
    members: &[usize],
    root: usize,
    work_scale: f64,
    huber: f64,
) -> Option<Solution> {
    // The patch covers about as much of the picture as a feature did on the
    // reduced copy it was found on.
    let radius = (5.0 / work_scale).round().clamp(10.0, 24.0) as i64;
    let mut fine = 0usize;
    let measured: Vec<Pair> = pairs
        .par_iter()
        .map(|p| {
            let rji = rotations[p.j].transpose() * rotations[p.i];
            let a_full: Vec<[f64; 2]> = p.a.iter().map(|a| [a[0] / work_scale, a[1] / work_scale]).collect();
            let (w, h) = (images[p.i].width() as usize, images[p.i].height() as usize);
            let chosen = align::spread(&a_full, w, h, POINTS_PER_PAIR);
            let mut a = Vec::new();
            let mut b = Vec::new();
            for k in chosen {
                if let Some(q) = track(&images[p.i], &images[p.j], lens, &rji, a_full[k], radius) {
                    a.push(a_full[k]);
                    b.push(q);
                }
            }
            if a.len() >= 8 {
                (Pair { a, b, ..p.clone() }, true)
            } else {
                // Sky, water, a featureless wall: the patches found nothing to
                // hold on to, so this pair keeps its original matches rather
                // than leaving its photos unconnected.
                let b_full = p.b.iter().map(|b| [b[0] / work_scale, b[1] / work_scale]).collect();
                (Pair { a: a_full, b: b_full, weight: work_scale * work_scale, ..p.clone() }, false)
            }
        })
        .collect::<Vec<_>>()
        .into_iter()
        .filter_map(|(pair, ok)| {
            fine += ok as usize;
            Some(pair)
        })
        .collect();

    crate::debug(&format!(
        "fine alignment: {fine} of {} pairs, {} points re-measured",
        measured.len(),
        measured.iter().map(|p| p.a.len()).sum::<usize>()
    ));
    if fine == 0 {
        return None;
    }

    let solve = |freedom| align::bundle_adjust(*lens, rotations, &measured, members, root, freedom, huber.min(2.0));
    let simple = solve(Freedom { focal: true, k1: false, k2: false });
    // Two photos cannot tell distortion from focal length; three or more can.
    if members.len() < 3 {
        return Some(simple);
    }
    let points: usize = measured.iter().map(|p| p.a.len()).sum();
    let with_lens = solve(Freedom { focal: true, k1: true, k2: points > 1500 });
    crate::debug(&format!(
        "pair rms simple {:?} / with lens {:?}",
        simple.pair_rms.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(),
        with_lens.pair_rms.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>()
    ));
    for (p, r) in measured.iter().zip(&simple.pair_rms) {
        crate::debug(&format!("  pair {}-{} weight {:.2} points {} rms {:.2}", p.i, p.j, p.weight, p.a.len(), r));
    }
    Some(if with_lens.rms < simple.rms * 0.97 { with_lens } else { simple })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(w: u32, h: u32, sx: f64, sy: f64) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| {
            let (x, y) = (x as f64 - sx, y as f64 - sy);
            let v = 128.0
                + 40.0 * (x * 0.21).sin() * (y * 0.17).cos()
                + 30.0 * ((x + y) * 0.13).sin()
                + 20.0 * ((x * 0.07) - (y * 0.11)).cos();
            let v = v.clamp(0.0, 255.0) as u8;
            image::Rgba([v, v, v, 255])
        })
    }

    #[test]
    fn tracks_a_subpixel_shift() {
        let a = texture(200, 200, 0.0, 0.0);
        let b = texture(200, 200, 1.3, -0.6);
        let lens = Lens { focal: 1000.0, k1: 0.0, k2: 0.0 };
        let q = track(&a, &b, &lens, &Matrix3::identity(), [10.0, -5.0], 12).expect("tracked");
        assert!((q[0] - 11.3).abs() < 0.05 && (q[1] + 5.6).abs() < 0.05, "{q:?}");
    }
}
