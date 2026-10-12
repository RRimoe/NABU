//! Precision against ground truth.
//!
//! Photos are rendered from a procedural world by a camera that turns on the
//! spot, through a lens with known focal length and barrel distortion, with a
//! different exposure for each shot, vignetting and sensor noise. Because the
//! true geometry is known, the alignment error can be measured in pixels,
//! which real photos never allow.

use image::{Rgba, RgbaImage};
use nabu_stitch::camera::Lens;
use nabu_stitch::{align_images, compose, Options};
use nalgebra::{Matrix3, Rotation3, Vector3};

/// Smooth pseudo-random value noise in 3-D.
fn hash(x: i64, y: i64, z: i64) -> f64 {
    let mut h = (x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263) ^ z.wrapping_mul(2_147_483_647)) as u64;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFFFF) as f64 / 65535.0
}

fn noise(p: Vector3<f64>) -> f64 {
    let (x0, y0, z0) = (p.x.floor(), p.y.floor(), p.z.floor());
    let (fx, fy, fz) = (p.x - x0, p.y - y0, p.z - z0);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy, sz) = (s(fx), s(fy), s(fz));
    let (x0, y0, z0) = (x0 as i64, y0 as i64, z0 as i64);
    let mut acc = 0.0;
    for (dz, wz) in [(0, 1.0 - sz), (1, sz)] {
        for (dy, wy) in [(0, 1.0 - sy), (1, sy)] {
            for (dx, wx) in [(0, 1.0 - sx), (1, sx)] {
                acc += wx * wy * wz * hash(x0 + dx, y0 + dy, z0 + dz);
            }
        }
    }
    acc
}

/// Radiance of the world in a direction: detail at every scale like a real
/// scene (a 1/f-like spectrum down to below a pixel), crisp edges where a
/// noise field crosses a threshold, and a slowly varying tint.
fn world(d: Vector3<f64>) -> [f64; 3] {
    let d = d.normalize();
    let mut amp = 0.5;
    let mut freq = 5.0;
    let mut luma = 0.0;
    for _ in 0..11 {
        luma += amp * noise(d * freq);
        amp *= 0.78;
        freq *= 2.0;
    }
    let mut edges = 0.0;
    for (k, f) in [37.0, 140.0, 520.0, 1900.0].iter().enumerate() {
        if noise(d * *f + Vector3::repeat(5.5 + k as f64)) > 0.6 {
            edges += 0.12;
        }
    }
    let base = 0.15 + 0.35 * luma + edges;
    let tint = [noise(d * 3.0 + Vector3::repeat(17.3)), noise(d * 3.0 + Vector3::repeat(41.7)), 0.5];
    [
        (base * (0.75 + 0.5 * tint[0])).clamp(0.0, 1.0),
        (base * (0.75 + 0.5 * tint[1])).clamp(0.0, 1.0),
        (base * 0.9).clamp(0.0, 1.0),
    ]
}

struct Shot {
    rotation: Matrix3<f64>,
    gain: f64,
}

fn render(lens: &Lens, shot: &Shot, width: u32, height: u32, seed: u64) -> RgbaImage {
    let (cx, cy) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rand = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % 10_000) as f64 / 10_000.0 - 0.5
    };
    let half_diag = (cx * cx + cy * cy).sqrt();
    RgbaImage::from_fn(width, height, |x, y| {
        // 2×2 supersampling, as a sensor integrates over its pixel.
        let mut acc = [0.0; 3];
        for (ox, oy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
            let p = [x as f64 - cx + ox, y as f64 - cy + oy];
            let c = world(shot.rotation * lens.ray(p));
            for k in 0..3 {
                acc[k] += c[k] * 0.25;
            }
        }
        let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt() / half_diag;
        let vignette = 1.0 - 0.25 * r * r;
        let mut px = |v: f64| ((v * shot.gain * vignette + rand() * 0.01).clamp(0.0, 1.0) * 255.0).round() as u8;
        Rgba([px(acc[0]), px(acc[1]), px(acc[2]), 255])
    })
}

/// Pixel error of the estimated relative geometry over a grid of points.
fn transfer_error(truth: &Lens, est: &Lens, rt: &[Matrix3<f64>], re: &[Matrix3<f64>], w: f64, h: f64) -> (f64, f64) {
    let (mut sum, mut max, mut count) = (0.0, 0.0f64, 0usize);
    for i in 0..rt.len() {
        for j in 0..rt.len() {
            if i == j {
                continue;
            }
            for gy in 0..12 {
                for gx in 0..16 {
                    let p = [(gx as f64 / 15.0 - 0.5) * (w - 1.0), (gy as f64 / 11.0 - 0.5) * (h - 1.0)];
                    let Some(qt) = truth.project(&(rt[j].transpose() * rt[i] * truth.ray(p))) else { continue };
                    if qt[0].abs() > w / 2.0 || qt[1].abs() > h / 2.0 {
                        continue;
                    }
                    let Some(qe) = est.project(&(re[j].transpose() * re[i] * est.ray(p))) else { continue };
                    let e = ((qt[0] - qe[0]).powi(2) + (qt[1] - qe[1]).powi(2)).sqrt();
                    sum += e * e;
                    max = max.max(e);
                    count += 1;
                }
            }
        }
    }
    ((sum / count.max(1) as f64).sqrt(), max)
}

fn scene(yaws: &[f64], pitches: &[f64]) -> Vec<Shot> {
    let mut shots = Vec::new();
    let mut k = 0;
    for &pitch in pitches {
        for &yaw in yaws {
            let r = Rotation3::from_euler_angles(0.0, yaw.to_radians(), 0.0)
                * Rotation3::from_euler_angles(pitch.to_radians(), 0.0, (k as f64 * 0.7).sin().to_radians());
            shots.push(Shot { rotation: r.into_inner(), gain: 0.85 + 0.1 * ((k * 7 % 5) as f64 / 4.0) * 2.0 });
            k += 1;
        }
    }
    shots
}

fn run(name: &str, shots: &[Shot], truth: Lens, w: u32, h: u32, options: &Options, limit: f64) {
    let images: Vec<RgbaImage> = shots.iter().enumerate().map(|(k, s)| render(&truth, s, w, h, k as u64 + 1)).collect();
    let alignment = align_images(&images, options, &mut |_, _| true).expect("alignment");
    assert_eq!(alignment.members.len(), shots.len(), "{name}: every photo must be placed");
    let rt: Vec<Matrix3<f64>> = shots.iter().map(|s| s.rotation).collect();
    let (rms, max) = transfer_error(&truth, &alignment.lens, &rt, &alignment.rotations, w as f64, h as f64);
    let focal_error = (alignment.lens.focal / truth.focal - 1.0).abs() * 100.0;
    println!(
        "{name}: transfer rms {rms:.3}px max {max:.3}px, focal {:.2} (true {:.2}, {focal_error:.3}%), k1 {:.4} (true {:.4}), reported rms {:.3}px",
        alignment.lens.focal, truth.focal, alignment.lens.k1, truth.k1, alignment.rms
    );
    assert!(rms < limit, "{name}: alignment error {rms:.3}px over {limit}px");

    if std::env::var_os("NABU_STITCH_SAVE").is_some() {
        let pano = compose(&images, &alignment, options, &mut |_, _| true).expect("compose");
        pano.image.save(format!("/tmp/nabu-stitch-{name}.png")).unwrap();
    }
}

#[test]
fn single_row_with_distortion() {
    let truth = Lens { focal: 900.0, k1: -0.06, k2: 0.0 };
    let shots = scene(&[-50.0, -25.0, 0.0, 25.0, 50.0], &[0.0]);
    run("row", &shots, truth, 960, 640, &Options::default(), 0.05);
}

#[test]
fn two_rows() {
    let truth = Lens { focal: 700.0, k1: 0.0, k2: 0.0 };
    let shots = scene(&[-30.0, 0.0, 30.0], &[-18.0, 18.0]);
    run("rows", &shots, truth, 800, 600, &Options::default(), 0.05);
}

#[test]
fn high_resolution_benefits_from_fine_alignment() {
    // Registration at 0.3 MP of 3 MP photos: the features alone are ten
    // times coarser than the output.
    let truth = Lens { focal: 2200.0, k1: -0.03, k2: 0.0 };
    let shots = scene(&[-20.0, 0.0, 20.0], &[0.0]);
    let coarse = Options { registration_megapixels: 0.3, fine_alignment: false, ..Options::default() };
    let fine = Options { registration_megapixels: 0.3, ..Options::default() };
    run("coarse", &shots, truth, 2100, 1400, &coarse, 0.5);
    run("fine", &shots, truth, 2100, 1400, &fine, 0.05);
}

#[test]
fn tracking_is_subpixel_on_rendered_photos() {
    let truth = Lens { focal: 2200.0, k1: -0.03, k2: 0.0 };
    let shots = scene(&[0.0, 20.0], &[0.0]);
    let (w, h) = (2100, 1400);
    let images: Vec<RgbaImage> = shots.iter().enumerate().map(|(k, s)| render(&truth, s, w, h, k as u64 + 1)).collect();
    let rji = shots[1].rotation.transpose() * shots[0].rotation;
    let mut errors = Vec::new();
    for gy in 0..8 {
        for gx in 0..6 {
            let a = [200.0 + gx as f64 * 60.0, -500.0 + gy as f64 * 140.0];
            let Some(truth_b) = truth.project(&(rji * truth.ray(a))) else { continue };
            if let Some(b) = nabu_stitch::refine::track(&images[0], &images[1], &truth, &rji, a, 16) {
                errors.push(((b[0] - truth_b[0]).powi(2) + (b[1] - truth_b[1]).powi(2)).sqrt());
            }
        }
    }
    errors.sort_by(f64::total_cmp);
    println!("tracked {} points, median {:.3}px, max {:.3}px", errors.len(), errors[errors.len() / 2], errors.last().unwrap());
    assert!(errors[errors.len() / 2] < 0.05);
}

#[test]
#[ignore]
fn save_sample_render() {
    let truth = Lens { focal: 2200.0, k1: -0.03, k2: 0.0 };
    let s0 = Shot { rotation: Matrix3::identity(), gain: 1.0 };
    render(&truth, &s0, 2100, 1400, 1).save("/tmp/nabu-render.png").unwrap();
}
