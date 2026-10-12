//! Robust homography between two overlapping photos.
//!
//! RANSAC on four-point samples finds the consensus, then the model is
//! re-estimated from every inlier and the inlier set re-evaluated until it
//! stops changing ("LO-RANSAC"-style local optimisation). Only the
//! verification job is done here: the final geometry comes from the bundle
//! adjustment, which models the camera properly.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

pub type Point = [f64; 2];

pub struct Fit {
    /// Maps points of the first image to the second: p2 ~ H p1.
    pub h: Matrix3<f64>,
    pub inliers: Vec<usize>,
}

/// Deterministic generator, so a stitch is reproducible run to run.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub fn apply(h: &Matrix3<f64>, p: Point) -> Option<Point> {
    let v = h * Vector3::new(p[0], p[1], 1.0);
    if v.z.abs() < 1e-12 {
        return None;
    }
    Some([v.x / v.z, v.y / v.z])
}

/// Hartley normalisation: centroid at the origin, mean distance √2.
fn normaliser(points: &[Point]) -> Matrix3<f64> {
    let n = points.len() as f64;
    let (mut cx, mut cy) = (0.0, 0.0);
    for p in points {
        cx += p[0];
        cy += p[1];
    }
    cx /= n;
    cy /= n;
    let mean = points.iter().map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()).sum::<f64>()
        / n;
    let s = if mean > 1e-12 { std::f64::consts::SQRT_2 / mean } else { 1.0 };
    Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
}

/// Exact homography from four correspondences (h33 = 1).
fn four_point(a: &[Point; 4], b: &[Point; 4]) -> Option<Matrix3<f64>> {
    let mut m = SMatrix::<f64, 8, 8>::zeros();
    let mut rhs = SVector::<f64, 8>::zeros();
    for i in 0..4 {
        let (x, y) = (a[i][0], a[i][1]);
        let (u, v) = (b[i][0], b[i][1]);
        let r = 2 * i;
        m[(r, 0)] = x;
        m[(r, 1)] = y;
        m[(r, 2)] = 1.0;
        m[(r, 6)] = -u * x;
        m[(r, 7)] = -u * y;
        rhs[r] = u;
        m[(r + 1, 3)] = x;
        m[(r + 1, 4)] = y;
        m[(r + 1, 5)] = 1.0;
        m[(r + 1, 6)] = -v * x;
        m[(r + 1, 7)] = -v * y;
        rhs[r + 1] = v;
    }
    let h = m.lu().solve(&rhs)?;
    Some(Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0))
}

/// Least-squares homography from any number of correspondences (normalised DLT).
pub fn dlt(a: &[Point], b: &[Point]) -> Option<Matrix3<f64>> {
    if a.len() < 4 {
        return None;
    }
    let ta = normaliser(a);
    let tb = normaliser(b);
    let mut ata = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U9, _>::zeros();
    for (pa, pb) in a.iter().zip(b) {
        let qa = ta * Vector3::new(pa[0], pa[1], 1.0);
        let qb = tb * Vector3::new(pb[0], pb[1], 1.0);
        let (x, y) = (qa.x, qa.y);
        let (u, v) = (qb.x, qb.y);
        let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
        let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
        for i in 0..9 {
            for j in 0..9 {
                ata[(i, j)] += r1[i] * r1[j] + r2[i] * r2[j];
            }
        }
    }
    let eig = ata.symmetric_eigen();
    let (mut min_i, mut min_v) = (0, f64::MAX);
    for (i, v) in eig.eigenvalues.iter().enumerate() {
        if *v < min_v {
            min_v = *v;
            min_i = i;
        }
    }
    let h = eig.eigenvectors.column(min_i);
    let hn = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let full = tb.try_inverse()? * hn * ta;
    if full[(2, 2)].abs() < 1e-15 {
        return None;
    }
    Some(full / full[(2, 2)])
}

fn inliers_of(h: &Matrix3<f64>, a: &[Point], b: &[Point], threshold: f64) -> Vec<usize> {
    let t2 = threshold * threshold;
    (0..a.len())
        .filter(|&i| {
            apply(h, a[i]).is_some_and(|p| (p[0] - b[i][0]).powi(2) + (p[1] - b[i][1]).powi(2) <= t2)
        })
        .collect()
}

/// A sample whose points are nearly collinear gives a meaningless model.
fn degenerate(p: &[Point; 4]) -> bool {
    let area = |a: Point, b: Point, c: Point| ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs();
    let scale = p.iter().map(|q| q[0].abs() + q[1].abs()).fold(1.0, f64::max);
    let tiny = 1e-4 * scale * scale;
    area(p[0], p[1], p[2]) < tiny
        || area(p[0], p[1], p[3]) < tiny
        || area(p[0], p[2], p[3]) < tiny
        || area(p[1], p[2], p[3]) < tiny
}

/// Robust fit. `threshold` is the reprojection tolerance in pixels.
pub fn ransac(a: &[Point], b: &[Point], threshold: f64, seed: u64) -> Option<Fit> {
    let n = a.len();
    if n < 8 {
        return None;
    }
    let mut rng = Rng::new(seed ^ n as u64);
    let mut best: Vec<usize> = Vec::new();
    let mut best_h = Matrix3::identity();
    let max_iterations = 4000usize;
    let mut needed = max_iterations;
    let mut iteration = 0;

    while iteration < needed.min(max_iterations) {
        iteration += 1;
        let mut idx = [0usize; 4];
        for k in 0..4 {
            loop {
                let c = rng.below(n);
                if !idx[..k].contains(&c) {
                    idx[k] = c;
                    break;
                }
            }
        }
        let sa = [a[idx[0]], a[idx[1]], a[idx[2]], a[idx[3]]];
        let sb = [b[idx[0]], b[idx[1]], b[idx[2]], b[idx[3]]];
        if degenerate(&sa) || degenerate(&sb) {
            continue;
        }
        let Some(h) = four_point(&sa, &sb) else { continue };
        // Orientation-reversing models are mirror images, never a camera.
        if h.fixed_view::<2, 2>(0, 0).determinant() <= 0.0 {
            continue;
        }
        let inliers = inliers_of(&h, a, b, threshold);
        if inliers.len() > best.len() {
            best = inliers;
            best_h = h;
            let ratio = best.len() as f64 / n as f64;
            let denom = (1.0 - ratio.powi(4)).max(1e-12).ln();
            needed = if denom < 0.0 { ((1.0 - 0.999f64).ln() / denom).ceil() as usize } else { 0 };
            needed = needed.max(50);
        }
    }
    if best.len() < 8 {
        return None;
    }

    // Local optimisation: refit on all inliers until the set settles.
    let mut h = best_h;
    let mut inliers = best;
    for _ in 0..5 {
        let pa: Vec<Point> = inliers.iter().map(|&i| a[i]).collect();
        let pb: Vec<Point> = inliers.iter().map(|&i| b[i]).collect();
        let Some(refit) = dlt(&pa, &pb) else { break };
        let next = inliers_of(&refit, a, b, threshold);
        if next.len() < inliers.len() {
            break;
        }
        let same = next == inliers;
        h = refit;
        inliers = next;
        if same {
            break;
        }
    }
    Some(Fit { h, inliers })
}

/// Root-mean-square transfer error of a homography over the given matches.
pub fn rms(h: &Matrix3<f64>, a: &[Point], b: &[Point]) -> f64 {
    let mut sum = 0.0;
    for (pa, pb) in a.iter().zip(b) {
        if let Some(p) = apply(h, *pa) {
            sum += (p[0] - pb[0]).powi(2) + (p[1] - pb[1]).powi(2);
        }
    }
    (sum / a.len().max(1) as f64).sqrt()
}
