//! Small image containers and the filters the rest of the pipeline shares.
//!
//! Everything works on `f32` planes: features need sub-pixel precision, the
//! exposure model needs linear light and the blender needs signed values, and
//! converting back and forth from 8-bit at each stage would lose all three.

use image::RgbaImage;
use rayon::prelude::*;

/// One channel of floating point pixels, row-major.
#[derive(Clone, Debug)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height] }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    /// Bilinear sample with clamped edges.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let top = self.at(x0, y0) * (1.0 - fx) + self.at(x1, y0) * fx;
        let bottom = self.at(x0, y1) * (1.0 - fx) + self.at(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// Every other pixel: the step between SIFT octaves, applied to an image
    /// that has already been blurred enough not to alias.
    pub fn decimate(&self) -> Plane {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut out = Plane::new(width, height);
        out.data.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            let src = self.row((y * 2).min(self.height - 1));
            for (x, value) in row.iter_mut().enumerate() {
                *value = src[(x * 2).min(self.width - 1)];
            }
        });
        out
    }
}

/// Normalised 1-D Gaussian kernel, truncated at four sigma.
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 4.0).ceil().max(1.0) as i32;
    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= sum);
    kernel
}

/// Separable Gaussian blur with mirrored edges.
pub fn gaussian_blur(src: &Plane, sigma: f32) -> Plane {
    if sigma <= 0.01 {
        return src.clone();
    }
    let kernel = gaussian_kernel(sigma);
    let radius = (kernel.len() / 2) as isize;
    let (w, h) = (src.width, src.height);

    let mirror = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let mut i = i;
        if n == 1 {
            return 0;
        }
        while i < 0 || i >= n {
            if i < 0 {
                i = -i - 1;
            }
            if i >= n {
                i = 2 * n - i - 1;
            }
        }
        i as usize
    };

    // Horizontal pass, one row per task.
    let mut tmp = Plane::new(w, h);
    tmp.data.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = src.row(y);
        for x in 0..w {
            let mut acc = 0.0;
            if x as isize >= radius && (x as isize) + radius < w as isize {
                let start = x - radius as usize;
                for (k, v) in kernel.iter().zip(&row[start..start + kernel.len()]) {
                    acc += k * v;
                }
            } else {
                for (t, k) in kernel.iter().enumerate() {
                    acc += k * row[mirror(x as isize + t as isize - radius, w)];
                }
            }
            out[x] = acc;
        }
    });

    // Vertical pass: each output row is a weighted sum of whole input rows,
    // which keeps the inner loop contiguous and vectorisable.
    let mut out = Plane::new(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (t, k) in kernel.iter().enumerate() {
            let src_row = tmp.row(mirror(y as isize + t as isize - radius, h));
            for (o, v) in row.iter_mut().zip(src_row) {
                *o += k * v;
            }
        }
    });
    out
}

/// sRGB transfer function, 8-bit code to linear light.
pub fn srgb_to_linear_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    for (i, v) in table.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    }
    table
}

#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Luma in 0..1, the input SIFT works on.
pub fn to_gray(image: &RgbaImage) -> Plane {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let raw = image.as_raw();
    let mut out = Plane::new(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let i = (y * w + x) * 4;
            *v = (0.299 * raw[i] as f32 + 0.587 * raw[i + 1] as f32 + 0.114 * raw[i + 2] as f32)
                / 255.0;
        }
    });
    out
}

/// Area-correct downscale to a target scale (0 < scale <= 1).
///
/// A box filter over the exact footprint of each output pixel: unlike point
/// sampling it does not alias fine texture into false features, and unlike a
/// fixed-width kernel it stays correct at any ratio.
pub fn downscale(image: &RgbaImage, scale: f64) -> RgbaImage {
    if scale >= 0.999 {
        return image.clone();
    }
    let (sw, sh) = (image.width() as usize, image.height() as usize);
    let dw = ((sw as f64 * scale).round() as usize).max(1);
    let dh = ((sh as f64 * scale).round() as usize).max(1);
    let fx = sw as f64 / dw as f64;
    let fy = sh as f64 / dh as f64;
    let raw = image.as_raw();

    // Per-axis footprints: (first index, weights).
    let spans = |dst: usize, step: f64, limit: usize| -> Vec<(usize, Vec<f32>)> {
        (0..dst)
            .map(|d| {
                let a = d as f64 * step;
                let b = ((d + 1) as f64 * step).min(limit as f64);
                let first = a.floor() as usize;
                let last = (b.ceil() as usize).min(limit).max(first + 1);
                let weights = (first..last)
                    .map(|s| {
                        let lo = (s as f64).max(a);
                        let hi = ((s + 1) as f64).min(b);
                        (hi - lo).max(0.0) as f32
                    })
                    .collect();
                (first, weights)
            })
            .collect()
    };
    let xs = spans(dw, fx, sw);
    let ys = spans(dh, fy, sh);

    let mut out = vec![0u8; dw * dh * 4];
    out.par_chunks_mut(dw * 4).enumerate().for_each(|(y, row)| {
        let (y0, wy) = &ys[y];
        for (x, px) in row.chunks_mut(4).enumerate() {
            let (x0, wx) = &xs[x];
            let mut acc = [0.0f32; 4];
            let mut total = 0.0f32;
            for (j, wyj) in wy.iter().enumerate() {
                let base = (y0 + j) * sw;
                for (i, wxi) in wx.iter().enumerate() {
                    let w = wxi * wyj;
                    let k = (base + x0 + i) * 4;
                    acc[0] += raw[k] as f32 * w;
                    acc[1] += raw[k + 1] as f32 * w;
                    acc[2] += raw[k + 2] as f32 * w;
                    acc[3] += raw[k + 3] as f32 * w;
                    total += w;
                }
            }
            for c in 0..4 {
                px[c] = (acc[c] / total.max(1e-6)).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    RgbaImage::from_raw(dw as u32, dh as u32, out).expect("buffer size matches")
}

/// sRGB and linear light, both ways, through interpolated tables: exact to
/// well under one 16-bit code, and fast enough to run on every pixel.
pub struct Transfer {
    to_linear: Vec<f32>,
    to_srgb: Vec<f32>,
}

const TRANSFER_STEPS: usize = 16384;

impl Default for Transfer {
    fn default() -> Self {
        Self::new()
    }
}

impl Transfer {
    pub fn new() -> Self {
        let decode = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        let to_linear = (0..=TRANSFER_STEPS).map(|i| decode(i as f32 / TRANSFER_STEPS as f32)).collect();
        // Linear values are sampled on a square-root grid, dense near black
        // where the encoding curve is steepest.
        let to_srgb = (0..=TRANSFER_STEPS)
            .map(|i| {
                let t = i as f32 / TRANSFER_STEPS as f32;
                linear_to_srgb(t * t)
            })
            .collect();
        Self { to_linear, to_srgb }
    }

    #[inline]
    fn lookup(table: &[f32], t: f32) -> f32 {
        let x = t.clamp(0.0, 1.0) * TRANSFER_STEPS as f32;
        let i = (x as usize).min(TRANSFER_STEPS - 1);
        let f = x - i as f32;
        table[i] * (1.0 - f) + table[i + 1] * f
    }

    #[inline]
    pub fn linear(&self, srgb: f32) -> f32 {
        Self::lookup(&self.to_linear, srgb)
    }

    #[inline]
    pub fn srgb(&self, linear: f32) -> f32 {
        Self::lookup(&self.to_srgb, linear.clamp(0.0, 1.0).sqrt())
    }
}
