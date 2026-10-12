//! The camera model: a pinhole that only rotates, with one lens shared by
//! every photo and a radial distortion term.
//!
//! A panorama is shot by turning on the spot, so each photo differs from the
//! next by a rotation alone. Modelling exactly that, rather than a free
//! homography per pair, is what lets errors cancel instead of accumulating
//! around the panorama. Pixel coordinates here are always centred on the
//! image, with y pointing down.

use nalgebra::{Matrix3, Rotation3, Vector3};

/// Intrinsics shared by all photos: they come from the same lens.
#[derive(Clone, Copy, Debug)]
pub struct Lens {
    /// Focal length in pixels.
    pub focal: f64,
    /// Radial distortion, on coordinates normalised by the focal length.
    pub k1: f64,
    pub k2: f64,
}

impl Lens {
    pub fn scaled(&self, factor: f64) -> Lens {
        Lens { focal: self.focal * factor, ..*self }
    }

    #[inline]
    pub fn distort(&self, n: [f64; 2]) -> [f64; 2] {
        let r2 = n[0] * n[0] + n[1] * n[1];
        let s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        [n[0] * s, n[1] * s]
    }

    /// Inverse of `distort`, by fixed-point iteration (converges for any
    /// distortion a real lens has).
    #[inline]
    pub fn undistort(&self, d: [f64; 2]) -> [f64; 2] {
        if self.k1 == 0.0 && self.k2 == 0.0 {
            return d;
        }
        let mut n = d;
        for _ in 0..10 {
            let r2 = n[0] * n[0] + n[1] * n[1];
            let s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
            if s.abs() < 1e-6 {
                break;
            }
            n = [d[0] / s, d[1] / s];
        }
        n
    }

    /// Direction, in the camera's frame, seen by a centred pixel.
    #[inline]
    pub fn ray(&self, p: [f64; 2]) -> Vector3<f64> {
        let n = self.undistort([p[0] / self.focal, p[1] / self.focal]);
        Vector3::new(n[0], n[1], 1.0)
    }

    /// Centred pixel at which the camera sees a direction in its own frame.
    #[inline]
    pub fn project(&self, c: &Vector3<f64>) -> Option<[f64; 2]> {
        if c.z <= 1e-9 {
            return None;
        }
        let n = [c.x / c.z, c.y / c.z];
        // Past the radius where the distortion polynomial turns back on
        // itself, directions far outside the photo would fold into it.
        let r2 = n[0] * n[0] + n[1] * n[1];
        if 1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r2 * r2 <= 0.2 {
            return None;
        }
        let d = self.distort(n);
        Some([d[0] * self.focal, d[1] * self.focal])
    }
}

/// Rotation from a rotation vector (axis × angle).
pub fn exp_so3(w: &Vector3<f64>) -> Matrix3<f64> {
    Rotation3::new(*w).into_inner()
}

/// The rotation closest to a matrix that is one up to scale and noise.
pub fn nearest_rotation(m: &Matrix3<f64>) -> Matrix3<f64> {
    let mut m = *m;
    if m.determinant() < 0.0 {
        m = -m;
    }
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        let mut u = u;
        u.column_mut(2).neg_mut();
        r = u * v_t;
    }
    r
}

/// Focal lengths implied by a homography between two rotated views
/// (Szeliski & Shum 1997), for centred coordinates and p1 ~ H p0.
/// Returns the geometric mean of the two estimates when both exist.
pub fn focal_from_homography(h: &Matrix3<f64>) -> Option<f64> {
    let h = h / h[(2, 2)];
    let g = |r: usize, c: usize| h[(r, c)];

    let f1 = {
        let d1 = g(2, 0) * g(2, 1);
        let d2 = (g(2, 1) - g(2, 0)) * (g(2, 1) + g(2, 0));
        let v1 = -(g(0, 0) * g(0, 1) + g(1, 0) * g(1, 1)) / d1;
        let v2 = (g(0, 0) * g(0, 0) + g(1, 0) * g(1, 0) - g(0, 1) * g(0, 1) - g(1, 1) * g(1, 1)) / d2;
        let (v1, v2) = if v1 < v2 { (v2, v1) } else { (v1, v2) };
        if v1 > 0.0 && v2 > 0.0 {
            Some((if d1.abs() > d2.abs() { v1 } else { v2 }).sqrt())
        } else if v1 > 0.0 {
            Some(v1.sqrt())
        } else {
            None
        }
    };
    let f0 = {
        let d1 = g(0, 0) * g(1, 0) + g(0, 1) * g(1, 1);
        let d2 = g(0, 0) * g(0, 0) + g(0, 1) * g(0, 1) - g(1, 0) * g(1, 0) - g(1, 1) * g(1, 1);
        let v1 = -g(0, 2) * g(1, 2) / d1;
        let v2 = (g(1, 2) * g(1, 2) - g(0, 2) * g(0, 2)) / d2;
        let (v1, v2) = if v1 < v2 { (v2, v1) } else { (v1, v2) };
        if v1 > 0.0 && v2 > 0.0 {
            Some((if d1.abs() > d2.abs() { v1 } else { v2 }).sqrt())
        } else if v1 > 0.0 {
            Some(v1.sqrt())
        } else {
            None
        }
    };
    match (f0, f1) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some((a * b).sqrt()),
        _ => None,
    }
}
