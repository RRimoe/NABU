//! Photos taken walking along a flat subject.
//!
//! A textured wall is photographed by a camera that slides along it (and
//! turns a little, as a hand does), with a different exposure each time. The
//! stitcher has to recognise that the camera moved rather than turned, and
//! place each photo where it truly lies on the wall.

use image::{Rgba, RgbaImage};
use nabu_stitch::{align_images, compose, Options, Projection};
use nalgebra::Vector3;

fn hash(x: i64, y: i64) -> f64 {
    let mut h = (x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263)) as u64;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xFFFF) as f64 / 65535.0
}

fn noise(x: f64, y: f64) -> f64 {
    let (x0, y0) = (x.floor(), y.floor());
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(x - x0), s(y - y0));
    let (xi, yi) = (x0 as i64, y0 as i64);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    lerp(
        lerp(hash(xi, yi), hash(xi + 1, yi), sx),
        lerp(hash(xi, yi + 1), hash(xi + 1, yi + 1), sx),
        sy,
    )
}

/// The wall's colour at a point on it (metres).
fn wall(x: f64, y: f64) -> [f64; 3] {
    let mut v = 0.0;
    let mut amp = 0.5;
    let mut freq = 4.0;
    for _ in 0..6 {
        v += amp * noise(x * freq, y * freq);
        amp *= 0.5;
        freq *= 2.1;
    }
    let edge = if noise(x * 3.0 + 7.0, y * 3.0) > 0.55 { 0.35 } else { 0.0 };
    [v + edge, v * 0.8 + 0.1, v * 0.6 + edge * 0.5]
}

const FOCAL: f64 = 700.0;
const DISTANCE: f64 = 4.0;

/// The wall photographed from `along` metres along it, turned by `yaw`.
fn shot(along: f64, yaw: f64, gain: f64, width: u32, height: u32) -> RgbaImage {
    let (cx, cy) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
    let (s, c) = yaw.to_radians().sin_cos();
    RgbaImage::from_fn(width, height, |x, y| {
        let d = Vector3::new((x as f64 - cx) / FOCAL, (y as f64 - cy) / FOCAL, 1.0);
        let d = Vector3::new(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
        let t = DISTANCE / d.z;
        let p = wall(along + d.x * t, d.y * t);
        let px = |v: f64| ((v * gain).clamp(0.0, 1.0) * 255.0).round() as u8;
        Rgba([px(p[0]), px(p[1]), px(p[2]), 255])
    })
}

#[test]
fn a_walk_along_a_wall_is_laid_flat() {
    let (w, h) = (800, 600);
    // Two metres between shots: a little under half a frame.
    let stops = [(0.0, 1.0, 1.0), (2.0, -1.5, 1.15), (4.0, 0.5, 0.9), (6.0, -0.5, 1.05)];
    let images: Vec<RgbaImage> = stops.iter().map(|&(a, yaw, g)| shot(a, yaw, g, w, h)).collect();
    let options = Options { in_order: true, ..Options::default() };
    let alignment = align_images(&images, &options, &mut |_, _| true).expect("alignment");
    assert_eq!(alignment.members.len(), 4);
    assert!(alignment.walking, "a walk must be recognised");

    // Neighbours sit two metres apart on the wall, 2 × 700 / 4 = 350 pixels,
    // plus what the turn of the hand adds.
    let planes = alignment.planar.as_ref().expect("planar placement");
    for k in 1..4 {
        let step = planes[k][(0, 2)] - planes[k - 1][(0, 2)];
        let turn = FOCAL * (stops[k].1 - stops[k - 1].1).to_radians().tan();
        let expected = 2.0 * FOCAL / DISTANCE + turn;
        assert!((step - expected).abs() < 6.0, "photo {k} is {step:.1}px from the previous one, not {expected:.1}px");
        let rise = planes[k][(1, 2)] - planes[k - 1][(1, 2)];
        assert!(rise.abs() < 10.0, "photo {k} rises {rise:.1}px");
    }

    let pano = compose(&images, &alignment, &options, &mut |_, _| true).expect("compose");
    assert_eq!(pano.projection, Projection::Planar);
    // Four photos, each 800 wide, 350 apart: about 1850 wide, not bent into a
    // taller band.
    let (pw, ph) = (pano.image.width() as f64, pano.image.height() as f64);
    assert!((pw - 1850.0).abs() < 150.0, "width {pw}");
    assert!(ph > 520.0 && ph < 640.0, "height {ph}");
}
