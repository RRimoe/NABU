//! Matching the brightness and colour of the photos to each other.
//!
//! Cameras re-meter between shots and white balance drifts, so the same sky
//! comes out a shade different in each photo. One gain per photo and channel,
//! in linear light, is chosen so that overlapping areas agree, while a prior
//! keeps the gains near one (otherwise "everything black" would be a perfect
//! solution). This is Brown & Lowe's gain compensation, per channel.

use crate::imgops::srgb_to_linear_table;
use crate::warp::Warped;
use nalgebra::{DMatrix, DVector};

/// Standard deviation of the overlap error and of the gain prior.
const SIGMA_N: f64 = 10.0 / 255.0;
const SIGMA_G: f64 = 0.1;

/// Gains per photo and channel (linear light), for photos warped at a low
/// resolution onto the same surface.
pub fn gains(warped: &[Warped]) -> Vec<[f32; 3]> {
    let n = warped.len();
    let lut = srgb_to_linear_table();
    let to_linear = |v: f32| lut[(v * 255.0).round().clamp(0.0, 255.0) as usize] as f64;

    // Overlap statistics: count and mean intensity of each photo over the
    // part it shares with each other photo.
    let mut count = vec![vec![0.0f64; n]; n];
    let mut mean = vec![vec![[0.0f64; 3]; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&warped[i], &warped[j]);
            let Some(r) = a.rect.intersect(&b.rect) else { continue };
            let (mut c, mut sa, mut sb) = (0.0, [0.0f64; 3], [0.0f64; 3]);
            for y in 0..r.height as i64 {
                let ya = (r.y + y - a.rect.y) as usize;
                let yb = (r.y + y - b.rect.y) as usize;
                for x in 0..r.width as i64 {
                    let xa = (r.x + x - a.rect.x) as usize;
                    let xb = (r.x + x - b.rect.x) as usize;
                    let ka = ya * a.rect.width + xa;
                    let kb = yb * b.rect.width + xb;
                    if a.mask[ka] == 0 || b.mask[kb] == 0 {
                        continue;
                    }
                    let (pa, pb) = (a.rgb[ka], b.rgb[kb]);
                    // Clipped highlights carry no information about exposure.
                    if pa.iter().chain(pb.iter()).any(|v| *v > 0.98) {
                        continue;
                    }
                    c += 1.0;
                    for ch in 0..3 {
                        sa[ch] += to_linear(pa[ch]);
                        sb[ch] += to_linear(pb[ch]);
                    }
                }
            }
            if c < 1.0 {
                continue;
            }
            count[i][j] = c;
            count[j][i] = c;
            for ch in 0..3 {
                mean[i][j][ch] = sa[ch] / c;
                mean[j][i][ch] = sb[ch] / c;
            }
        }
    }

    let mut out = vec![[1.0f32; 3]; n];
    for ch in 0..3 {
        let mut a = DMatrix::<f64>::zeros(n, n);
        let mut b = DVector::<f64>::zeros(n);
        for i in 0..n {
            // A photo with no overlaps keeps a gain of one through the prior.
            a[(i, i)] += 1.0 / (SIGMA_G * SIGMA_G);
            b[i] += 1.0 / (SIGMA_G * SIGMA_G);
            for j in 0..n {
                if i == j || count[i][j] == 0.0 {
                    continue;
                }
                let nij = count[i][j];
                let (iij, iji) = (mean[i][j][ch], mean[j][i][ch]);
                a[(i, i)] += nij * (iij * iij / (SIGMA_N * SIGMA_N) + 1.0 / (SIGMA_G * SIGMA_G));
                a[(i, j)] -= nij * iij * iji / (SIGMA_N * SIGMA_N);
                b[i] += nij / (SIGMA_G * SIGMA_G);
            }
        }
        if let Some(g) = a.lu().solve(&b) {
            for i in 0..n {
                out[i][ch] = g[i].clamp(0.2, 5.0) as f32;
            }
        }
    }
    out
}
