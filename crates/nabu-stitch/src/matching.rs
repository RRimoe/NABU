//! Descriptor matching between two images.
//!
//! Exhaustive nearest-neighbour search: with a few thousand features per image
//! it is fast enough, and unlike approximate trees it never misses the true
//! neighbour, which is what the ratio test depends on.

use crate::sift::{Features, DESCRIPTOR_LEN};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Match {
    pub a: usize,
    pub b: usize,
    /// Distance ratio to the second-best candidate; smaller is more distinctive.
    pub ratio: f32,
}

/// Best and second-best neighbour of every descriptor of `a` among `b`,
/// as (index, squared distance, second squared distance).
fn nearest(a: &Features, b: &Features, take_a: usize, take_b: usize) -> Vec<(usize, f32, f32)> {
    let nb = take_b.min(b.len());
    (0..take_a.min(a.len()))
        .into_par_iter()
        .map(|i| {
            let da = a.descriptor(i);
            let (mut best, mut best_d, mut second_d) = (usize::MAX, f32::MAX, f32::MAX);
            for j in 0..nb {
                let db = &b.descriptors[j * DESCRIPTOR_LEN..(j + 1) * DESCRIPTOR_LEN];
                // Unit vectors: |a - b|² = 2 - 2 a·b.
                let mut dot = 0.0f32;
                for k in 0..DESCRIPTOR_LEN {
                    dot += da[k] * db[k];
                }
                let d = 2.0 - 2.0 * dot;
                if d < best_d {
                    second_d = best_d;
                    best_d = d;
                    best = j;
                } else if d < second_d {
                    second_d = d;
                }
            }
            (best, best_d, second_d)
        })
        .collect()
}

/// Matches passing Lowe's ratio test in both directions and agreeing with
/// each other (mutual nearest neighbours).
///
/// Only the first `take` features of each side are used, which lets the
/// caller run a cheap pass on the strongest features to find which images
/// overlap before paying for the full one.
pub fn match_features(a: &Features, b: &Features, ratio: f32, take: usize) -> Vec<Match> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let forward = nearest(a, b, take, take);
    let backward = nearest(b, a, take, take);
    let r2 = ratio * ratio;

    forward
        .iter()
        .enumerate()
        .filter_map(|(i, &(j, d1, d2))| {
            if j == usize::MAX || d1 > r2 * d2 {
                return None;
            }
            let (back, _, _) = backward[j];
            if back != i {
                return None;
            }
            Some(Match { a: i, b: j, ratio: (d1 / d2.max(1e-12)).sqrt() })
        })
        .collect()
}
