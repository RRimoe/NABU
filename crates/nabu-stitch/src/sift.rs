//! Scale-invariant feature transform (Lowe 2004), the detector behind every
//! precise panorama stitcher since AutoStitch.
//!
//! Follows the reference algorithm closely: a Gaussian scale space with three
//! layers per octave, extrema of the difference of Gaussians refined to
//! sub-pixel and sub-scale position, edge responses rejected, dominant
//! orientations from a smoothed 36-bin histogram, and the 4×4×8 gradient
//! descriptor with trilinear interpolation. Descriptors are stored as
//! RootSIFT (Arandjelović & Zisserman 2012), which matches measurably better
//! than plain SIFT at no cost.

use crate::imgops::{gaussian_blur, Plane};
use rayon::prelude::*;
use std::f32::consts::PI;

pub const DESCRIPTOR_LEN: usize = 128;

const LAYERS: usize = 3;
const SIGMA: f32 = 1.6;
/// Blur a camera image is assumed to already have.
const INITIAL_SIGMA: f32 = 0.5;
const EDGE_THRESHOLD: f32 = 10.0;
const BORDER: usize = 5;
const MAX_INTERP_STEPS: usize = 5;
const ORI_BINS: usize = 36;
const ORI_SIGMA_FACTOR: f32 = 1.5;
const ORI_RADIUS_FACTOR: f32 = 3.0 * ORI_SIGMA_FACTOR;
const ORI_PEAK_RATIO: f32 = 0.8;
const DESCR_WIDTH: usize = 4;
const DESCR_BINS: usize = 8;
const DESCR_SCALE_FACTOR: f32 = 3.0;
const DESCR_MAG_THR: f32 = 0.2;

#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    /// Position in the input image's pixels, pixel centres at integers.
    pub x: f32,
    pub y: f32,
    /// Scale of the feature in input pixels.
    pub sigma: f32,
    /// Dominant gradient direction, radians, image axes (y down).
    pub angle: f32,
    /// Contrast of the extremum: bigger is more reliable.
    pub response: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Features {
    pub keypoints: Vec<Keypoint>,
    /// `keypoints.len() * DESCRIPTOR_LEN` values, each descriptor unit length.
    pub descriptors: Vec<f32>,
}

impl Features {
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    pub fn descriptor(&self, i: usize) -> &[f32] {
        &self.descriptors[i * DESCRIPTOR_LEN..(i + 1) * DESCRIPTOR_LEN]
    }
}

struct Candidate {
    octave: usize,
    layer: usize,
    /// Refined position in octave pixels.
    x: f32,
    y: f32,
    /// Scale relative to the octave.
    scale: f32,
    response: f32,
}

/// Detects features and computes their descriptors.
///
/// `contrast` is Lowe's contrast threshold (0.04 in the paper); lower finds
/// more features in flat images such as skies and water. At most `limit`
/// keypoints are kept, strongest first.
pub fn detect(image: &Plane, contrast: f32, limit: usize) -> Features {
    let min_side = image.width.min(image.height) as f32;
    let octaves = ((min_side.log2() - 3.0).floor() as usize).clamp(1, 8);

    // Bring the image from its assumed blur to the scale space's base blur.
    let base_sigma = (SIGMA * SIGMA - INITIAL_SIGMA * INITIAL_SIGMA).max(0.01).sqrt();
    let base = gaussian_blur(image, base_sigma);

    // Incremental blurs between consecutive layers of an octave.
    let k = 2f32.powf(1.0 / LAYERS as f32);
    let mut steps = vec![0.0f32; LAYERS + 3];
    for (i, step) in steps.iter_mut().enumerate().skip(1) {
        let prev = SIGMA * k.powi(i as i32 - 1);
        let total = prev * k;
        *step = (total * total - prev * prev).sqrt();
    }

    let mut gauss: Vec<Vec<Plane>> = Vec::with_capacity(octaves);
    for o in 0..octaves {
        let first = if o == 0 { base.clone() } else { gauss[o - 1][LAYERS].decimate() };
        if first.width < 2 * BORDER + 4 || first.height < 2 * BORDER + 4 {
            break;
        }
        let mut layers = vec![first];
        for step in steps.iter().skip(1) {
            let next = gaussian_blur(layers.last().unwrap(), *step);
            layers.push(next);
        }
        gauss.push(layers);
    }

    let dog: Vec<Vec<Plane>> = gauss
        .iter()
        .map(|layers| {
            layers
                .windows(2)
                .map(|pair| {
                    let mut d = Plane::new(pair[0].width, pair[0].height);
                    for ((o, a), b) in d.data.iter_mut().zip(&pair[0].data).zip(&pair[1].data) {
                        *o = b - a;
                    }
                    d
                })
                .collect()
        })
        .collect();

    let threshold = 0.5 * contrast / LAYERS as f32;

    // Extrema, one task per (octave, layer, row).
    let mut tasks = Vec::new();
    for (o, layers) in dog.iter().enumerate() {
        for l in 1..=LAYERS {
            for y in BORDER..layers[l].height - BORDER {
                tasks.push((o, l, y));
            }
        }
    }
    let mut candidates: Vec<Candidate> = tasks
        .par_iter()
        .flat_map_iter(|&(o, l, y)| {
            let layers = &dog[o];
            let (prev, cur, next) = (&layers[l - 1], &layers[l], &layers[l + 1]);
            let w = cur.width;
            let mut found = Vec::new();
            for x in BORDER..w - BORDER {
                let v = cur.at(x, y);
                if v.abs() <= threshold {
                    continue;
                }
                if !is_extremum(prev, cur, next, x, y, v) {
                    continue;
                }
                if let Some(c) = refine(layers, o, l, x, y, contrast) {
                    found.push(c);
                }
            }
            found
        })
        .collect();

    // Strongest first, so the limit drops the weakest.
    candidates.sort_by(|a, b| b.response.total_cmp(&a.response));

    // Orientations, then descriptors. Each candidate can yield several
    // keypoints, one per strong orientation.
    let oriented: Vec<(Keypoint, usize, usize, f32, f32, f32)> = candidates
        .par_iter()
        .flat_map_iter(|c| {
            let g = &gauss[c.octave][c.layer];
            let scale_octave = SIGMA * c.scale;
            let factor = (1u32 << c.octave) as f32;
            orientations(g, c.x, c.y, scale_octave)
                .into_iter()
                .map(|angle| {
                    (
                        Keypoint {
                            x: c.x * factor,
                            y: c.y * factor,
                            sigma: scale_octave * factor,
                            angle,
                            response: c.response,
                        },
                        c.octave,
                        c.layer,
                        c.x,
                        c.y,
                        scale_octave,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let kept: Vec<_> = oriented.into_iter().take(limit).collect();
    let descriptors: Vec<[f32; DESCRIPTOR_LEN]> = kept
        .par_iter()
        .map(|(kp, o, l, x, y, s)| descriptor(&gauss[*o][*l], *x, *y, kp.angle, *s))
        .collect();

    Features {
        keypoints: kept.iter().map(|k| k.0).collect(),
        descriptors: descriptors.into_iter().flatten().collect(),
    }
}

fn is_extremum(prev: &Plane, cur: &Plane, next: &Plane, x: usize, y: usize, v: f32) -> bool {
    let w = cur.width;
    let check = |p: &Plane, skip_centre: bool| -> bool {
        for dy in 0..3 {
            let row = &p.data[(y + dy - 1) * w + x - 1..(y + dy - 1) * w + x + 2];
            for (dx, &n) in row.iter().enumerate() {
                if skip_centre && dx == 1 && dy == 1 {
                    continue;
                }
                if v > 0.0 {
                    if n > v {
                        return false;
                    }
                } else if n < v {
                    return false;
                }
            }
        }
        true
    };
    check(cur, true) && check(prev, false) && check(next, false)
}

/// Quadratic interpolation of an extremum in (x, y, scale), with the
/// contrast and edge tests. Returns None for unstable points.
fn refine(
    layers: &[Plane],
    octave: usize,
    layer: usize,
    x: usize,
    y: usize,
    contrast: f32,
) -> Option<Candidate> {
    let (mut xi, mut yi, mut li) = (x as isize, y as isize, layer as isize);
    let w = layers[0].width as isize;
    let h = layers[0].height as isize;
    let mut offset = [0.0f32; 3];
    let mut grad = [0.0f32; 3];
    let mut converged = false;

    for _ in 0..MAX_INTERP_STEPS {
        let (p, c, n) = (&layers[li as usize - 1], &layers[li as usize], &layers[li as usize + 1]);
        let at = |img: &Plane, dx: isize, dy: isize| img.at((xi + dx) as usize, (yi + dy) as usize);

        grad = [
            (at(c, 1, 0) - at(c, -1, 0)) * 0.5,
            (at(c, 0, 1) - at(c, 0, -1)) * 0.5,
            (at(n, 0, 0) - at(p, 0, 0)) * 0.5,
        ];
        let v2 = at(c, 0, 0) * 2.0;
        let dxx = at(c, 1, 0) + at(c, -1, 0) - v2;
        let dyy = at(c, 0, 1) + at(c, 0, -1) - v2;
        let dss = at(n, 0, 0) + at(p, 0, 0) - v2;
        let dxy = (at(c, 1, 1) - at(c, -1, 1) - at(c, 1, -1) + at(c, -1, -1)) * 0.25;
        let dxs = (at(n, 1, 0) - at(n, -1, 0) - at(p, 1, 0) + at(p, -1, 0)) * 0.25;
        let dys = (at(n, 0, 1) - at(n, 0, -1) - at(p, 0, 1) + at(p, 0, -1)) * 0.25;

        let hessian = nalgebra::Matrix3::new(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
        let solved = hessian.lu().solve(&nalgebra::Vector3::new(-grad[0], -grad[1], -grad[2]))?;
        offset = [solved[0], solved[1], solved[2]];

        if offset.iter().all(|v| v.abs() < 0.5) {
            converged = true;
            break;
        }
        if offset.iter().any(|v| v.abs() > 1e6 || !v.is_finite()) {
            return None;
        }
        xi += offset[0].round() as isize;
        yi += offset[1].round() as isize;
        li += offset[2].round() as isize;
        if li < 1
            || li > LAYERS as isize
            || xi < BORDER as isize
            || xi >= w - BORDER as isize
            || yi < BORDER as isize
            || yi >= h - BORDER as isize
        {
            return None;
        }
    }
    if !converged {
        return None;
    }

    let c = &layers[li as usize];
    let value = c.at(xi as usize, yi as usize)
        + 0.5 * (grad[0] * offset[0] + grad[1] * offset[1] + grad[2] * offset[2]);
    if value.abs() * (LAYERS as f32) < contrast {
        return None;
    }

    // Principal curvature ratio: a ridge has one large and one small
    // curvature and localises badly along its length.
    let (xu, yu) = (xi as usize, yi as usize);
    let v2 = c.at(xu, yu) * 2.0;
    let dxx = c.at(xu + 1, yu) + c.at(xu - 1, yu) - v2;
    let dyy = c.at(xu, yu + 1) + c.at(xu, yu - 1) - v2;
    let dxy = (c.at(xu + 1, yu + 1) - c.at(xu - 1, yu + 1) - c.at(xu + 1, yu - 1)
        + c.at(xu - 1, yu - 1))
        * 0.25;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    let r = EDGE_THRESHOLD;
    if det <= 0.0 || tr * tr * r >= (r + 1.0) * (r + 1.0) * det {
        return None;
    }

    Some(Candidate {
        octave,
        layer: li as usize,
        x: xi as f32 + offset[0],
        y: yi as f32 + offset[1],
        scale: 2f32.powf((li as f32 + offset[2]) / LAYERS as f32),
        response: value.abs(),
    })
}

/// Dominant gradient directions around a point.
fn orientations(img: &Plane, x: f32, y: f32, scale: f32) -> Vec<f32> {
    let radius = (ORI_RADIUS_FACTOR * scale).round() as isize;
    let sigma = ORI_SIGMA_FACTOR * scale;
    let weight_factor = -1.0 / (2.0 * sigma * sigma);
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    let (w, h) = (img.width as isize, img.height as isize);
    let mut hist = [0.0f32; ORI_BINS];

    for dy in -radius..=radius {
        let py = cy + dy;
        if py <= 0 || py >= h - 1 {
            continue;
        }
        for dx in -radius..=radius {
            let px = cx + dx;
            if px <= 0 || px >= w - 1 {
                continue;
            }
            let (pxu, pyu) = (px as usize, py as usize);
            let gx = img.at(pxu + 1, pyu) - img.at(pxu - 1, pyu);
            let gy = img.at(pxu, pyu + 1) - img.at(pxu, pyu - 1);
            let weight = (((dx * dx + dy * dy) as f32) * weight_factor).exp();
            let angle = gy.atan2(gx);
            let mut bin = ((angle + PI) / (2.0 * PI) * ORI_BINS as f32).floor() as isize;
            bin = bin.rem_euclid(ORI_BINS as isize);
            hist[bin as usize] += weight * (gx * gx + gy * gy).sqrt();
        }
    }

    // Smooth the circular histogram twice with [1 4 6 4 1] / 16.
    let n = ORI_BINS;
    let mut smooth = [0.0f32; ORI_BINS];
    for i in 0..n {
        smooth[i] = (hist[(i + n - 2) % n] + hist[(i + 2) % n]) * (1.0 / 16.0)
            + (hist[(i + n - 1) % n] + hist[(i + 1) % n]) * (4.0 / 16.0)
            + hist[i] * (6.0 / 16.0);
    }

    let max = smooth.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max * ORI_PEAK_RATIO;
    let mut angles = Vec::new();
    for i in 0..n {
        let l = smooth[(i + n - 1) % n];
        let r = smooth[(i + 1) % n];
        let c = smooth[i];
        if c > l && c > r && c >= threshold {
            let shift = 0.5 * (l - r) / (l - 2.0 * c + r);
            let bin = (i as f32 + shift + 0.5).rem_euclid(n as f32);
            angles.push(bin / n as f32 * 2.0 * PI - PI);
        }
    }
    angles
}

/// The 4×4×8 gradient histogram descriptor, as RootSIFT.
fn descriptor(img: &Plane, x: f32, y: f32, angle: f32, scale: f32) -> [f32; DESCRIPTOR_LEN] {
    let d = DESCR_WIDTH;
    let n = DESCR_BINS;
    let hist_width = DESCR_SCALE_FACTOR * scale;
    let radius = ((hist_width * std::f32::consts::SQRT_2 * (d as f32 + 1.0) * 0.5).round() as isize)
        .min(((img.width * img.width + img.height * img.height) as f32).sqrt() as isize);
    let (cos_t, sin_t) = (angle.cos() / hist_width, angle.sin() / hist_width);
    let bins_per_rad = n as f32 / (2.0 * PI);
    let exp_scale = -1.0 / (d as f32 * d as f32 * 0.5);
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    let (w, h) = (img.width as isize, img.height as isize);

    // (d + 2) × (d + 2) × (n + 2) so interpolation never needs bounds checks.
    let mut hist = vec![0.0f32; (d + 2) * (d + 2) * (n + 2)];
    let idx = |r: usize, c: usize, o: usize| (r * (d + 2) + c) * (n + 2) + o;

    for i in -radius..=radius {
        for j in -radius..=radius {
            // Offset rotated into the keypoint's frame.
            let c_rot = j as f32 * cos_t + i as f32 * sin_t;
            let r_rot = -(j as f32) * sin_t + i as f32 * cos_t;
            let rbin = r_rot + d as f32 / 2.0 - 0.5;
            let cbin = c_rot + d as f32 / 2.0 - 0.5;
            if rbin <= -1.0 || rbin >= d as f32 || cbin <= -1.0 || cbin >= d as f32 {
                continue;
            }
            let (py, px) = (cy + i, cx + j);
            if py <= 0 || py >= h - 1 || px <= 0 || px >= w - 1 {
                continue;
            }
            let (pxu, pyu) = (px as usize, py as usize);
            let gx = img.at(pxu + 1, pyu) - img.at(pxu - 1, pyu);
            let gy = img.at(pxu, pyu + 1) - img.at(pxu, pyu - 1);
            let mag = (gx * gx + gy * gy).sqrt()
                * ((c_rot * c_rot + r_rot * r_rot) * exp_scale).exp();
            let mut obin = (gy.atan2(gx) - angle) * bins_per_rad;
            obin = obin.rem_euclid(n as f32);

            let (r0, c0, o0) = (rbin.floor(), cbin.floor(), obin.floor());
            let (dr, dc, dob) = (rbin - r0, cbin - c0, obin - o0);
            let (r0, c0) = ((r0 as isize + 1) as usize, (c0 as isize + 1) as usize);
            let o0 = o0 as usize;

            let v_r1 = mag * dr;
            let v_r0 = mag - v_r1;
            for (rr, vr) in [(r0, v_r0), (r0 + 1, v_r1)] {
                let v_c1 = vr * dc;
                let v_c0 = vr - v_c1;
                for (cc, vc) in [(c0, v_c0), (c0 + 1, v_c1)] {
                    let v_o1 = vc * dob;
                    let v_o0 = vc - v_o1;
                    hist[idx(rr, cc, o0)] += v_o0;
                    hist[idx(rr, cc, o0 + 1)] += v_o1;
                }
            }
        }
    }

    let mut out = [0.0f32; DESCRIPTOR_LEN];
    for r in 0..d {
        for c in 0..d {
            // Fold the wrap-around orientation bins back in.
            let base = idx(r + 1, c + 1, 0);
            let mut bins = [0.0f32; DESCR_BINS];
            for (o, b) in bins.iter_mut().enumerate() {
                *b = hist[base + o];
            }
            bins[0] += hist[base + n];
            bins[1] += hist[base + n + 1];
            for o in 0..n {
                out[(r * d + c) * n + o] = bins[o];
            }
        }
    }

    // Normalise, clip the dominant gradients, renormalise.
    let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    let clip = norm * DESCR_MAG_THR;
    out.iter_mut().for_each(|v| *v = v.min(clip));
    // RootSIFT: L1-normalise and take the square root; the result has unit
    // L2 norm, so the Euclidean distance becomes the Hellinger kernel.
    let l1 = out.iter().sum::<f32>().max(1e-12);
    out.iter_mut().for_each(|v| *v = (*v / l1).sqrt());
    out
}
