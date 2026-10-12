//! From features to cameras: which photos overlap, how each one is turned,
//! and the focal length and distortion of the lens.
//!
//! The pipeline is Brown & Lowe's ("Automatic Panoramic Image Stitching using
//! Invariant Features", 2007): pairwise homographies verified by RANSAC and a
//! probabilistic inlier test, a maximum spanning tree for the initial
//! rotations, then a bundle adjustment of every rotation and the shared lens
//! over all the matches at once, with a robust loss.

use crate::camera::{exp_so3, focal_from_homography, nearest_rotation, Lens};
use crate::homography::{self, Point};
use crate::matching::match_features;
use crate::sift::Features;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use rayon::prelude::*;
use std::collections::VecDeque;

/// Features of one photo, with positions centred on the image.
pub struct ImageFeatures {
    pub features: Features,
    pub points: Vec<Point>,
    pub width: usize,
    pub height: usize,
}

/// Verified correspondences between two photos.
#[derive(Clone, Debug)]
pub struct Pair {
    pub i: usize,
    pub j: usize,
    /// Matching centred points in photo i and photo j.
    pub a: Vec<Point>,
    pub b: Vec<Point>,
    /// Maps photo i to photo j (centred coordinates).
    pub h: Matrix3<f64>,
    /// Inliers relative to what chance would give (Brown & Lowe's test).
    pub confidence: f64,
    /// How much its matches count in the bundle adjustment: one over the
    /// variance of their positions, relative to full-resolution measurements.
    pub weight: f64,
}

/// Most correspondences any pair contributes to the bundle adjustment. More
/// adds time, not precision: the rotation of a photo is pinned down by a
/// couple of hundred well spread points.
const MAX_POINTS_PER_PAIR: usize = 250;

/// Matches every candidate pair and keeps those that pass verification.
pub fn match_pairs(
    images: &[ImageFeatures],
    threshold: f64,
    mut progress: impl FnMut(f32),
) -> Vec<Pair> {
    let n = images.len();
    let mut candidates: Vec<(usize, usize)> = Vec::new();

    if n <= 10 {
        for i in 0..n {
            for j in i + 1..n {
                candidates.push((i, j));
            }
        }
    } else {
        // A quick pass with the strongest features finds who overlaps whom,
        // so the full matching runs on a handful of neighbours per photo
        // rather than on every pair of a large set.
        const QUICK: usize = 400;
        let mut counts = vec![vec![0usize; n]; n];
        let all: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
        let found: Vec<(usize, usize, usize)> = all
            .par_iter()
            .map(|&(i, j)| {
                let m = match_features(&images[i].features, &images[j].features, 0.8, QUICK);
                (i, j, m.len())
            })
            .collect();
        for (i, j, c) in found {
            counts[i][j] = c;
            counts[j][i] = c;
        }
        let mut set = std::collections::BTreeSet::new();
        for (i, row) in counts.iter().enumerate() {
            let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            order.sort_by(|&x, &y| row[y].cmp(&row[x]));
            for &j in order.iter().take(6) {
                if row[j] >= 6 {
                    set.insert((i.min(j), i.max(j)));
                }
            }
        }
        candidates = set.into_iter().collect();
    }

    let total = candidates.len().max(1);
    let mut pairs = Vec::new();
    for (done, &(i, j)) in candidates.iter().enumerate() {
        progress(done as f32 / total as f32);
        let (fa, fb) = (&images[i], &images[j]);
        let matches = match_features(&fa.features, &fb.features, 0.8, usize::MAX);
        if matches.len() < 12 {
            continue;
        }
        let a: Vec<Point> = matches.iter().map(|m| fa.points[m.a]).collect();
        let b: Vec<Point> = matches.iter().map(|m| fb.points[m.b]).collect();
        let Some(fit) = homography::ransac(&a, &b, threshold, (i * 1_000 + j) as u64) else {
            continue;
        };
        let inliers = fit.inliers.len() as f64;
        let chance = 8.0 + 0.3 * matches.len() as f64;
        if inliers < 15.0 || inliers < chance {
            continue;
        }
        let ia: Vec<Point> = fit.inliers.iter().map(|&k| a[k]).collect();
        let ib: Vec<Point> = fit.inliers.iter().map(|&k| b[k]).collect();
        let keep = spread(&ia, fa.width, fa.height, MAX_POINTS_PER_PAIR);
        pairs.push(Pair {
            i,
            j,
            a: keep.iter().map(|&k| ia[k]).collect(),
            b: keep.iter().map(|&k| ib[k]).collect(),
            h: fit.h,
            confidence: inliers / chance,
            weight: 1.0,
        });
    }
    progress(1.0);
    pairs
}

/// Picks up to `limit` points spread evenly over the image, so the bundle
/// adjustment is not dominated by one heavily textured corner.
pub fn spread(points: &[Point], width: usize, height: usize, limit: usize) -> Vec<usize> {
    if points.len() <= limit {
        return (0..points.len()).collect();
    }
    const GRID: usize = 8;
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); GRID * GRID];
    for (k, p) in points.iter().enumerate() {
        let gx = (((p[0] / width as f64) + 0.5) * GRID as f64).clamp(0.0, GRID as f64 - 1.0) as usize;
        let gy = (((p[1] / height as f64) + 0.5) * GRID as f64).clamp(0.0, GRID as f64 - 1.0) as usize;
        buckets[gy * GRID + gx].push(k);
    }
    let mut out = Vec::with_capacity(limit);
    let mut round = 0;
    while out.len() < limit {
        let mut any = false;
        for bucket in &buckets {
            if let Some(&k) = bucket.get(round) {
                out.push(k);
                any = true;
                if out.len() == limit {
                    break;
                }
            }
        }
        if !any {
            break;
        }
        round += 1;
    }
    out
}

/// The photos that connect to each other, as the largest connected group.
pub fn largest_component(n: usize, pairs: &[Pair]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut x = x;
        while p[x] != r {
            let next = p[x];
            p[x] = r;
            x = next;
        }
        r
    }
    for pair in pairs {
        let (a, b) = (find(&mut parent, pair.i), find(&mut parent, pair.j));
        if a != b {
            parent[a] = b;
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    groups.into_values().max_by_key(|g| g.len()).unwrap_or_default()
}

/// Initial lens and rotations from the pairwise homographies.
///
/// Rotations are chained along a maximum spanning tree of the most confident
/// pairs, rooted at its centre so no chain is longer than it has to be.
/// Returns the rotations (camera to world) and the index of the root.
pub fn initial_cameras(
    n: usize,
    members: &[usize],
    pairs: &[Pair],
    fallback_focal: f64,
) -> (Lens, Vec<Matrix3<f64>>, usize) {
    let mut focals: Vec<f64> = pairs.iter().filter_map(|p| focal_from_homography(&p.h)).collect();
    focals.retain(|f| f.is_finite() && *f > fallback_focal * 0.1 && *f < fallback_focal * 20.0);
    focals.sort_by(f64::total_cmp);
    let focal = if focals.is_empty() {
        fallback_focal
    } else if focals.len() % 2 == 1 {
        focals[focals.len() / 2]
    } else {
        (focals[focals.len() / 2 - 1] + focals[focals.len() / 2]) * 0.5
    };
    let lens = Lens { focal, k1: 0.0, k2: 0.0 };

    // Kruskal on descending confidence.
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by(|&a, &b| pairs[b].confidence.total_cmp(&pairs[a].confidence));
    let mut parent: Vec<usize> = (0..n).collect();
    let find = |p: &mut Vec<usize>, mut x: usize| {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    };
    let mut tree: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for &k in &order {
        let p = &pairs[k];
        let (a, b) = (find(&mut parent, p.i), find(&mut parent, p.j));
        if a != b {
            parent[a] = b;
            tree[p.i].push((p.j, k));
            tree[p.j].push((p.i, k));
        }
    }

    // Centre of the tree: the member whose farthest member is nearest.
    let bfs = |start: usize| -> Vec<usize> {
        let mut dist = vec![usize::MAX; n];
        dist[start] = 0;
        let mut q = VecDeque::from([start]);
        while let Some(u) = q.pop_front() {
            for &(v, _) in &tree[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        dist
    };
    let root = *members
        .iter()
        .min_by_key(|&&m| {
            let d = bfs(m);
            members.iter().map(|&o| d[o]).max().unwrap_or(0)
        })
        .unwrap_or(&members[0]);

    let k = Matrix3::new(focal, 0.0, 0.0, 0.0, focal, 0.0, 0.0, 0.0, 1.0);
    let k_inv = Matrix3::new(1.0 / focal, 0.0, 0.0, 0.0, 1.0 / focal, 0.0, 0.0, 0.0, 1.0);
    let mut rotations = vec![Matrix3::identity(); n];
    let mut seen = vec![false; n];
    seen[root] = true;
    let mut q = VecDeque::from([root]);
    while let Some(u) = q.pop_front() {
        for &(v, k_idx) in &tree[u] {
            if seen[v] {
                continue;
            }
            let p = &pairs[k_idx];
            // H maps i to j: K⁻¹ H K = Rjᵀ Ri.
            let h_uv = if p.i == u { p.h } else { p.h.try_inverse().unwrap_or(p.h) };
            let m = nearest_rotation(&(k_inv * h_uv * k));
            rotations[v] = rotations[u] * m.transpose();
            seen[v] = true;
            q.push_back(v);
        }
    }
    (lens, rotations, root)
}

/// What the bundle adjustment may change besides the rotations.
#[derive(Clone, Copy)]
pub struct Freedom {
    pub focal: bool,
    pub k1: bool,
    pub k2: bool,
}

pub struct Solution {
    pub lens: Lens,
    pub rotations: Vec<Matrix3<f64>>,
    /// Root-mean-square reprojection error, pixels.
    pub rms: f64,
    /// The same, for each pair.
    pub pair_rms: Vec<f64>,
}

/// Residuals of one pair: both directions, x and y.
fn pair_residuals(lens: &Lens, ri: &Matrix3<f64>, rj: &Matrix3<f64>, pair: &Pair, out: &mut Vec<f64>) {
    let rij = ri.transpose() * rj;
    let rji = rij.transpose();
    for (a, b) in pair.a.iter().zip(&pair.b) {
        // b seen from camera i must land on a, and the reverse.
        match lens.project(&(rij * lens.ray(*b))) {
            Some(p) => {
                out.push(p[0] - a[0]);
                out.push(p[1] - a[1]);
            }
            None => {
                out.push(0.0);
                out.push(0.0);
            }
        }
        match lens.project(&(rji * lens.ray(*a))) {
            Some(p) => {
                out.push(p[0] - b[0]);
                out.push(p[1] - b[1]);
            }
            None => {
                out.push(0.0);
                out.push(0.0);
            }
        }
    }
}

/// Levenberg–Marquardt over rotations and lens, with a Huber loss.
///
/// The root photo's rotation is held fixed (the panorama's orientation is
/// free and is set later by straightening). Distortion terms carry a weak
/// prior towards zero so that a set with few matches cannot invent a lens.
pub fn bundle_adjust(
    lens: Lens,
    rotations: &[Matrix3<f64>],
    pairs: &[Pair],
    members: &[usize],
    root: usize,
    freedom: Freedom,
    huber: f64,
) -> Solution {
    let n = rotations.len();
    // Parameter layout: [focal?, k1?, k2?, ω for each member but the root].
    let mut slots = vec![usize::MAX; n];
    let mut next = 0usize;
    let focal_slot = freedom.focal.then(|| {
        next += 1;
        next - 1
    });
    let k1_slot = freedom.k1.then(|| {
        next += 1;
        next - 1
    });
    let k2_slot = freedom.k2.then(|| {
        next += 1;
        next - 1
    });
    for &m in members {
        if m != root {
            slots[m] = next;
            next += 3;
        }
    }
    let params = next;

    let mut lens = lens;
    let mut rots = rotations.to_vec();
    let focal0 = lens.focal;
    // Prior strength on distortion, in pixel units.
    let prior = 0.02 * focal0;

    let cost_of = |lens: &Lens, rots: &[Matrix3<f64>]| -> (f64, Vec<f64>) {
        let mut total = 0.0;
        let mut per_pair = Vec::with_capacity(pairs.len());
        let mut buf = Vec::new();
        for p in pairs {
            buf.clear();
            pair_residuals(lens, &rots[p.i], &rots[p.j], p, &mut buf);
            let mut sq = 0.0;
            for xy in buf.chunks(2) {
                let r2 = xy[0] * xy[0] + xy[1] * xy[1];
                let r = r2.sqrt();
                // Huber: quadratic near zero, linear in the tails.
                total += p.weight * if r <= huber { r2 } else { 2.0 * huber * r - huber * huber };
                sq += r2;
            }
            per_pair.push((sq / (buf.len() / 2).max(1) as f64).sqrt());
        }
        total += (prior * lens.k1).powi(2) + (prior * lens.k2).powi(2);
        (total, per_pair)
    };

    let apply_step = |lens: &Lens, rots: &[Matrix3<f64>], step: &DVector<f64>| -> (Lens, Vec<Matrix3<f64>>) {
        let mut l = *lens;
        if let Some(s) = focal_slot {
            l.focal = (l.focal * (1.0 + step[s])).max(1.0);
        }
        if let Some(s) = k1_slot {
            l.k1 += step[s];
        }
        if let Some(s) = k2_slot {
            l.k2 += step[s];
        }
        let mut r = rots.to_vec();
        for &m in members {
            if slots[m] != usize::MAX {
                let s = slots[m];
                let w = Vector3::new(step[s], step[s + 1], step[s + 2]);
                r[m] = rots[m] * exp_so3(&w);
            }
        }
        (l, r)
    };

    let (mut cost, _) = cost_of(&lens, &rots);
    let mut lambda = 1e-3;
    const EPS: f64 = 1e-6;

    for _ in 0..100 {
        // Normal equations, accumulated pair by pair (each pair only touches
        // the lens and its two photos, so the Jacobian is never formed whole).
        let blocks: Vec<(DMatrix<f64>, DVector<f64>)> = pairs
            .par_iter()
            .map(|p| {
                let mut local: Vec<usize> = [focal_slot, k1_slot, k2_slot].into_iter().flatten().collect();
                for cam in [p.i, p.j] {
                    if slots[cam] != usize::MAX {
                        local.extend([slots[cam], slots[cam] + 1, slots[cam] + 2]);
                    }
                }
                let mut base = Vec::new();
                pair_residuals(&lens, &rots[p.i], &rots[p.j], p, &mut base);
                // IRLS weights for the Huber loss, per 2-D residual.
                let weights: Vec<f64> = base
                    .chunks(2)
                    .map(|xy| {
                        let r = (xy[0] * xy[0] + xy[1] * xy[1]).sqrt();
                        p.weight * if r <= huber { 1.0 } else { huber / r }
                    })
                    .collect();
                let rows = base.len();
                let mut jac = DMatrix::<f64>::zeros(rows, local.len());
                let mut buf = Vec::with_capacity(rows);
                for (c, &slot) in local.iter().enumerate() {
                    let mut step = DVector::<f64>::zeros(params);
                    step[slot] = EPS;
                    let (l2, _) = apply_step(&lens, &rots, &step);
                    let within = |cam: usize| slots[cam] != usize::MAX && slot >= slots[cam] && slot < slots[cam] + 3;
                    let ri = if within(p.i) {
                        rots[p.i] * exp_so3(&unit(slot - slots[p.i], EPS))
                    } else {
                        rots[p.i]
                    };
                    let rj = if within(p.j) {
                        rots[p.j] * exp_so3(&unit(slot - slots[p.j], EPS))
                    } else {
                        rots[p.j]
                    };
                    buf.clear();
                    pair_residuals(&l2, &ri, &rj, p, &mut buf);
                    for r in 0..rows {
                        jac[(r, c)] = (buf[r] - base[r]) / EPS;
                    }
                }
                for (r, w) in base.iter_mut().zip(weights.iter().flat_map(|w| [*w, *w])) {
                    *r *= w.sqrt();
                }
                for (r, w) in (0..rows).zip(weights.iter().flat_map(|w| [*w, *w])) {
                    let sw = w.sqrt();
                    for c in 0..local.len() {
                        jac[(r, c)] *= sw;
                    }
                }
                let jtj_local = jac.transpose() * &jac;
                let jtr_local = jac.transpose() * DVector::from_vec(base);
                let mut jtj = DMatrix::<f64>::zeros(params, params);
                let mut jtr = DVector::<f64>::zeros(params);
                for (a, &sa) in local.iter().enumerate() {
                    jtr[sa] += jtr_local[a];
                    for (b, &sb) in local.iter().enumerate() {
                        jtj[(sa, sb)] += jtj_local[(a, b)];
                    }
                }
                (jtj, jtr)
            })
            .collect();

        let mut jtj = DMatrix::<f64>::zeros(params, params);
        let mut jtr = DVector::<f64>::zeros(params);
        for (a, b) in blocks {
            jtj += a;
            jtr += b;
        }
        // The distortion prior, linear in the parameters.
        if let Some(s) = k1_slot {
            jtj[(s, s)] += prior * prior;
            jtr[s] += prior * prior * lens.k1;
        }
        if let Some(s) = k2_slot {
            jtj[(s, s)] += prior * prior;
            jtr[s] += prior * prior * lens.k2;
        }

        let mut improved = false;
        for _ in 0..10 {
            let mut damped = jtj.clone();
            for d in 0..params {
                damped[(d, d)] += lambda * jtj[(d, d)].max(1e-9);
            }
            let Some(step) = damped.cholesky().map(|c| c.solve(&(-&jtr))) else {
                lambda *= 10.0;
                continue;
            };
            let (l2, r2) = apply_step(&lens, &rots, &step);
            let (c2, _) = cost_of(&l2, &r2);
            if c2 < cost {
                let gain = (cost - c2) / cost.max(1e-30);
                lens = l2;
                rots = r2;
                cost = c2;
                lambda = (lambda / 3.0).max(1e-12);
                improved = gain > 1e-10;
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }

    for r in rots.iter_mut() {
        *r = nearest_rotation(r);
    }
    let (_, pair_rms) = cost_of(&lens, &rots);
    // Weighted, so coarse fallback matches do not mask how well the
    // precise ones agree.
    let mut sum = 0.0;
    let mut count = 0.0;
    let mut buf = Vec::new();
    for p in pairs {
        buf.clear();
        pair_residuals(&lens, &rots[p.i], &rots[p.j], p, &mut buf);
        for xy in buf.chunks(2) {
            sum += p.weight * (xy[0] * xy[0] + xy[1] * xy[1]);
            count += p.weight;
        }
    }
    Solution { lens, rotations: rots, rms: (sum / count.max(1e-12)).sqrt(), pair_rms }
}

fn unit(axis: usize, eps: f64) -> Vector3<f64> {
    let mut v = Vector3::zeros();
    v[axis] = eps;
    v
}

/// Turns the whole panorama so its horizon is level (Brown & Lowe's
/// "automatic panorama straightening"): photos taken while turning around a
/// vertical axis have their x axes in one plane, and that plane's normal is
/// the true vertical.
pub fn straighten(rotations: &mut [Matrix3<f64>], members: &[usize]) {
    if members.len() < 2 {
        return;
    }
    // Horizontal or vertical sweep: the axis along which the optical axes
    // spread the most.
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for &m in members {
        let z = rotations[m].column(2);
        min_x = min_x.min(z[0]);
        max_x = max_x.max(z[0]);
        min_y = min_y.min(z[1]);
        max_y = max_y.max(z[1]);
    }
    let horizontal = (max_x - min_x) >= (max_y - min_y);

    let mut moment = Matrix3::<f64>::zeros();
    for &m in members {
        let col = rotations[m].column(if horizontal { 0 } else { 1 }).into_owned();
        moment += col * col.transpose();
    }
    let eig = moment.symmetric_eigen();
    let mut idx: Vec<usize> = (0..3).collect();
    idx.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    // Smallest eigenvector: the normal to the plane of the x (or y) axes.
    let normal: Vector3<f64> = eig.eigenvectors.column(idx[0]).into_owned();

    let mut forward = Vector3::zeros();
    for &m in members {
        forward += rotations[m].column(2);
    }

    let (rg0, rg1, rg2);
    if horizontal {
        // rg1 is the new y (vertical) axis.
        let mut up = normal;
        // Keep "down" pointing the way the photos' own y axes point.
        let mut agree = 0.0;
        for &m in members {
            agree += up.dot(&rotations[m].column(1));
        }
        if agree < 0.0 {
            up = -up;
        }
        let mut x = up.cross(&forward);
        if x.norm() < 1e-12 {
            return;
        }
        x.normalize_mut();
        let z = x.cross(&up);
        rg0 = x;
        rg1 = up;
        rg2 = z;
    } else {
        let mut side = normal;
        let mut agree = 0.0;
        for &m in members {
            agree += side.dot(&rotations[m].column(0));
        }
        if agree < 0.0 {
            side = -side;
        }
        let mut y = forward.cross(&side);
        if y.norm() < 1e-12 {
            return;
        }
        y.normalize_mut();
        let z = side.cross(&y);
        rg0 = side;
        rg1 = y;
        rg2 = z;
    }
    let world = Matrix3::from_rows(&[rg0.transpose(), rg1.transpose(), rg2.transpose()]);
    for &m in members {
        rotations[m] = nearest_rotation(&(world * rotations[m]));
    }
}
