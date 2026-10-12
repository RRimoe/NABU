//! Projecting photos onto the panorama surface.
//!
//! Every output pixel is traced back to the photo that saw it (inverse
//! mapping), so the result has no holes, and sampled with a bicubic kernel so
//! fine detail survives the resampling.

use crate::camera::Lens;
use image::RgbaImage;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// A flat picture: straight lines stay straight. Only for moderate fields
    /// of view; stretches badly past about 120°.
    Rectilinear,
    /// Wraps horizontally: the classic wide panorama.
    Cylindrical,
    /// Wraps both ways: for multi-row panoramas and very tall fields of view.
    Spherical,
}

/// The panorama surface: a projection and its scale in pixels per radian.
#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub projection: Projection,
    pub scale: f64,
}

impl Surface {
    /// World direction to surface coordinates (pixels, origin straight ahead).
    #[inline]
    pub fn forward(&self, w: &Vector3<f64>) -> Option<[f64; 2]> {
        let s = self.scale;
        match self.projection {
            Projection::Rectilinear => {
                if w.z <= 1e-6 {
                    return None;
                }
                Some([s * w.x / w.z, s * w.y / w.z])
            }
            Projection::Cylindrical => {
                let r = (w.x * w.x + w.z * w.z).sqrt();
                if r < 1e-9 {
                    return None;
                }
                Some([s * w.x.atan2(w.z), s * w.y / r])
            }
            Projection::Spherical => {
                let n = w.norm();
                if n < 1e-12 {
                    return None;
                }
                Some([s * w.x.atan2(w.z), s * (w.y / n).clamp(-1.0, 1.0).asin()])
            }
        }
    }

    /// Range of u that wraps around, if the projection wraps.
    pub fn period(&self) -> Option<f64> {
        match self.projection {
            Projection::Rectilinear => None,
            _ => Some(2.0 * PI * self.scale),
        }
    }
}

/// Precomputed per-column and per-row terms of the inverse projection.
struct Inverse {
    /// For each output column: (a, b) so that the ray's x and z come from them.
    cols: Vec<(f64, f64)>,
    /// For each output row: (c, d).
    rows: Vec<(f64, f64)>,
    projection: Projection,
}

impl Inverse {
    fn new(surface: &Surface, x0: f64, y0: f64, width: usize, height: usize) -> Self {
        let s = surface.scale;
        let cols = (0..width)
            .map(|x| {
                let u = (x0 + x as f64) / s;
                match surface.projection {
                    Projection::Rectilinear => (u, 1.0),
                    _ => (u.sin(), u.cos()),
                }
            })
            .collect();
        let rows = (0..height)
            .map(|y| {
                let v = (y0 + y as f64) / s;
                match surface.projection {
                    Projection::Spherical => (v.sin(), v.cos()),
                    _ => (v, 1.0),
                }
            })
            .collect();
        Self { cols, rows, projection: surface.projection }
    }

    #[inline]
    fn ray(&self, x: usize, y: usize) -> Vector3<f64> {
        let (a, b) = self.cols[x];
        let (c, d) = self.rows[y];
        match self.projection {
            Projection::Rectilinear => Vector3::new(a, c, 1.0),
            Projection::Cylindrical => Vector3::new(a, c, b),
            Projection::Spherical => Vector3::new(a * d, c, b * d),
        }
    }
}

/// Where one photo lands on the surface, as a pixel rectangle.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    pub fn right(&self) -> i64 {
        self.x + self.width as i64
    }

    pub fn bottom(&self) -> i64 {
        self.y + self.height as i64
    }

    pub fn intersect(&self, other: &Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let r = self.right().min(other.right());
        let b = self.bottom().min(other.bottom());
        (r > x && b > y).then(|| Rect { x, y, width: (r - x) as usize, height: (b - y) as usize })
    }

    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let r = self.right().max(other.right());
        let b = self.bottom().max(other.bottom());
        Rect { x, y, width: (r - x) as usize, height: (b - y) as usize }
    }
}

/// Footprint of a photo on the surface, from points along its border and
/// across its interior (the interior matters when a pole is in view).
/// `u` values are unwrapped around the photo's centre so a photo straddling
/// the back seam of a 360° panorama stays in one piece.
pub fn footprint(
    surface: &Surface,
    lens: &Lens,
    rotation: &Matrix3<f64>,
    width: usize,
    height: usize,
) -> Option<(f64, f64, f64, f64)> {
    let (hw, hh) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
    let centre = surface.forward(&(rotation * lens.ray([0.0, 0.0])))?;
    let period = surface.period();
    let (mut u0, mut u1, mut v0, mut v1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    const STEPS: usize = 48;
    for i in 0..=STEPS {
        for j in 0..=STEPS {
            let on_border = i == 0 || j == 0 || i == STEPS || j == STEPS;
            if !on_border && (i % 4 != 0 || j % 4 != 0) {
                continue;
            }
            let p = [-hw + 2.0 * hw * i as f64 / STEPS as f64, -hh + 2.0 * hh * j as f64 / STEPS as f64];
            let Some(mut q) = surface.forward(&(rotation * lens.ray(p))) else { continue };
            if let Some(period) = period {
                while q[0] - centre[0] > period / 2.0 {
                    q[0] -= period;
                }
                while q[0] - centre[0] < -period / 2.0 {
                    q[0] += period;
                }
            }
            u0 = u0.min(q[0]);
            u1 = u1.max(q[0]);
            v0 = v0.min(q[1]);
            v1 = v1.max(q[1]);
        }
    }
    (u0 <= u1 && v0 <= v1).then_some((u0, u1, v0, v1))
}

/// A photo projected onto its rectangle of the surface.
pub struct Warped {
    pub rect: Rect,
    /// RGB, 0..1 sRGB-encoded, row-major over `rect`.
    pub rgb: Vec<[f32; 3]>,
    /// 1 where the photo has a pixel, 0 elsewhere.
    pub mask: Vec<u8>,
}

#[inline]
fn cubic_weights(t: f32) -> [f32; 4] {
    // Keys' cubic convolution with a = -0.5 (Catmull-Rom).
    let a = -0.5f32;
    let w = |x: f32| {
        let x = x.abs();
        if x <= 1.0 {
            (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
        } else {
            0.0
        }
    };
    [w(1.0 + t), w(t), w(1.0 - t), w(2.0 - t)]
}

/// Bicubic sample of an RGBA image at a pixel position; None when outside the
/// photo or on a transparent part of it.
#[inline]
pub fn sample_bicubic(image: &RgbaImage, x: f64, y: f64) -> Option<[f32; 3]> {
    let (w, h) = (image.width() as i64, image.height() as i64);
    if x < -0.5 || y < -0.5 || x > w as f64 - 0.5 || y > h as f64 - 0.5 {
        return None;
    }
    let raw = image.as_raw();
    let xf = x.floor();
    let yf = y.floor();
    let (xi, yi) = (xf as i64, yf as i64);
    let near = (((y.round() as i64).clamp(0, h - 1) * w + (x.round() as i64).clamp(0, w - 1)) * 4 + 3) as usize;
    if raw[near] < 128 {
        return None;
    }
    let wx = cubic_weights((x - xf) as f32);
    let wy = cubic_weights((y - yf) as f32);
    let mut acc = [0.0f32; 3];
    for (j, wyj) in wy.iter().enumerate() {
        let sy = (yi - 1 + j as i64).clamp(0, h - 1);
        let row = (sy * w) as usize;
        let mut racc = [0.0f32; 3];
        for (i, wxi) in wx.iter().enumerate() {
            let sx = (xi - 1 + i as i64).clamp(0, w - 1) as usize;
            let k = (row + sx) * 4;
            racc[0] += raw[k] as f32 * wxi;
            racc[1] += raw[k + 1] as f32 * wxi;
            racc[2] += raw[k + 2] as f32 * wxi;
        }
        acc[0] += racc[0] * wyj;
        acc[1] += racc[1] * wyj;
        acc[2] += racc[2] * wyj;
    }
    Some([
        (acc[0] / 255.0).clamp(0.0, 1.0),
        (acc[1] / 255.0).clamp(0.0, 1.0),
        (acc[2] / 255.0).clamp(0.0, 1.0),
    ])
}

/// Projects a photo onto `rect` of the surface.
///
/// `lens` must be at the photo's own resolution, `surface.scale` at the
/// output's. Pixel coordinates on the surface are `rect.x + column` etc.,
/// with the surface origin straight ahead.
pub fn warp(
    image: &RgbaImage,
    lens: &Lens,
    rotation: &Matrix3<f64>,
    surface: &Surface,
    rect: Rect,
) -> Warped {
    let (iw, ih) = (image.width() as f64, image.height() as f64);
    let (cx, cy) = ((iw - 1.0) / 2.0, (ih - 1.0) / 2.0);
    let inverse = Inverse::new(surface, rect.x as f64, rect.y as f64, rect.width, rect.height);
    let to_camera = rotation.transpose();

    let mut rgb = vec![[0.0f32; 3]; rect.width * rect.height];
    let mut mask = vec![0u8; rect.width * rect.height];
    rgb.par_chunks_mut(rect.width)
        .zip(mask.par_chunks_mut(rect.width))
        .enumerate()
        .for_each(|(y, (row, mrow))| {
            for x in 0..rect.width {
                let c = to_camera * inverse.ray(x, y);
                let Some(p) = lens.project(&c) else { continue };
                if let Some(px) = sample_bicubic(image, p[0] + cx, p[1] + cy) {
                    row[x] = px;
                    mrow[x] = 1;
                }
            }
        });
    Warped { rect, rgb, mask }
}

/// Integer rectangle covering a footprint at a given scale factor.
pub fn rect_of(bounds: (f64, f64, f64, f64)) -> Rect {
    let (u0, u1, v0, v1) = bounds;
    let x = u0.floor() as i64 - 1;
    let y = v0.floor() as i64 - 1;
    let r = u1.ceil() as i64 + 2;
    let b = v1.ceil() as i64 + 2;
    Rect { x, y, width: (r - x).max(1) as usize, height: (b - y).max(1) as usize }
}
