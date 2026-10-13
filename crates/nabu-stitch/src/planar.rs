//! Photos taken by walking along a subject rather than turning on the spot.
//!
//! The rest of the crate assumes one point of view: the camera turns, and a
//! rotation per photo explains everything. Walk along a train, a wall or a
//! mural and that model no longer holds: the camera *moves*, nearby things
//! slide past faster than far ones, and forcing rotations onto that bends the
//! subject at every join.
//!
//! What does hold is that the subject is roughly flat, and a flat subject
//! seen from a camera that keeps facing it maps from photo to photo by an
//! affine transform. So each photo gets one, solved for all photos at once by
//! least squares over every match (linear, so there is no drift to correct
//! and no starting guess to get wrong), with the matches that disagree —
//! the ground, which is not on the subject's plane — weighted down and
//! solved again. The middle photo keeps its own frame.

use crate::align::Pair;
use nalgebra::{DMatrix, DVector, Matrix3};

/// The planar solution, in the centred working coordinates the matches use.
pub struct PlanarFit {
    /// Per photo (indexed like the input), its centred coordinates to the
    /// reference photo's. Identity for photos that are not members.
    pub transforms: Vec<Matrix3<f64>>,
    /// Root mean square of the match residuals, working pixels.
    pub rms: f64,
}

/// Reweighting rounds after the first solve.
const ROUNDS: usize = 4;

/// Solves an affine transform per member photo, `root` fixed to identity.
/// `huber` is the residual (working pixels) above which a match counts less.
pub fn fit(n: usize, members: &[usize], pairs: &[Pair], root: usize, huber: f64) -> Option<PlanarFit> {
    // Unknowns: six per member other than the root.
    let mut slot = vec![usize::MAX; n];
    let mut count = 0;
    for &m in members {
        if m != root {
            slot[m] = count;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let size = count * 6;
    let mut transforms = vec![Matrix3::identity(); n];
    let mut weights: Vec<Vec<f64>> = pairs.iter().map(|p| vec![1.0; p.a.len()]).collect();
    let mut rms = f64::INFINITY;

    for round in 0..=ROUNDS {
        let mut ata = DMatrix::<f64>::zeros(size, size);
        let mut atb = DVector::<f64>::zeros(size);
        for (k, p) in pairs.iter().enumerate() {
            if slot[p.i] == usize::MAX && p.i != root || slot[p.j] == usize::MAX && p.j != root {
                continue;
            }
            for (m, (a, b)) in p.a.iter().zip(&p.b).enumerate() {
                let w = weights[k][m];
                if w <= 0.0 {
                    continue;
                }
                // Residual A_i·a − A_j·b, one row per coordinate. A transform
                // is laid out as [a11 a12 tx a21 a22 ty].
                for axis in 0..2 {
                    let mut row: Vec<(usize, f64)> = Vec::with_capacity(6);
                    let mut rhs = 0.0;
                    for (photo, point, sign) in [(p.i, a, 1.0), (p.j, b, -1.0)] {
                        if photo == root {
                            rhs -= sign * point[axis];
                        } else {
                            let base = slot[photo] * 6 + axis * 3;
                            row.push((base, sign * point[0]));
                            row.push((base + 1, sign * point[1]));
                            row.push((base + 2, sign));
                        }
                    }
                    for &(r, vr) in &row {
                        atb[r] += w * vr * rhs;
                        for &(c, vc) in &row {
                            ata[(r, c)] += w * vr * vc;
                        }
                    }
                }
            }
        }
        // A whisper of a prior towards "no change" keeps a photo that only
        // touches the others along a thin strip from collapsing.
        for (s, v) in [(0, 1.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 1.0), (5, 0.0)] {
            for photo in 0..count {
                let i = photo * 6 + s;
                ata[(i, i)] += 1e-3;
                atb[i] += 1e-3 * v;
            }
        }
        let solution = ata.cholesky()?.solve(&atb);
        for (photo, &s) in slot.iter().enumerate() {
            if s == usize::MAX {
                continue;
            }
            let x = &solution.as_slice()[s * 6..s * 6 + 6];
            transforms[photo] = Matrix3::new(x[0], x[1], x[2], x[3], x[4], x[5], 0.0, 0.0, 1.0);
        }

        // Residuals, and weights for the next round.
        let (mut total, mut used) = (0.0, 0.0);
        for (k, p) in pairs.iter().enumerate() {
            let (ti, tj) = (&transforms[p.i], &transforms[p.j]);
            for (m, (a, b)) in p.a.iter().zip(&p.b).enumerate() {
                let pa = ti * nalgebra::Vector3::new(a[0], a[1], 1.0);
                let pb = tj * nalgebra::Vector3::new(b[0], b[1], 1.0);
                let r = ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt();
                total += r * r;
                used += 1.0;
                weights[k][m] = if r <= huber { 1.0 } else { huber / r };
                // Far off the plane: out altogether.
                if r > huber * 8.0 && round > 0 {
                    weights[k][m] = 0.0;
                }
            }
        }
        rms = if used > 0.0 { (total / used).sqrt() } else { f64::INFINITY };
    }

    Some(PlanarFit { transforms, rms })
}

/// Robust spread of the residuals of the planar fit: the median, which is
/// what decides whether the photos were shot walking. A mean would let the
/// ground (off the plane, and so always wrong) speak for the whole set.
pub fn median_residual(fit: &PlanarFit, pairs: &[Pair]) -> f64 {
    let mut all = Vec::new();
    for p in pairs {
        let (ti, tj) = (&fit.transforms[p.i], &fit.transforms[p.j]);
        for (a, b) in p.a.iter().zip(&p.b) {
            let pa = ti * nalgebra::Vector3::new(a[0], a[1], 1.0);
            let pb = tj * nalgebra::Vector3::new(b[0], b[1], 1.0);
            all.push(((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt());
        }
    }
    if all.is_empty() {
        return f64::INFINITY;
    }
    all.sort_by(f64::total_cmp);
    all[all.len() / 2]
}
