//! Healing on synthetic photos: the blemish must go and the texture around it
//! must carry on through the hole.

use image::{GrayImage, Luma, Rgba, RgbaImage};
use nabu_retouch::{heal, HealError, HealOptions};

/// Diagonal stripes over a soft vertical gradient: texture plus changing light.
fn stripes(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let band = ((x + y) / 6) % 2 == 0;
        let light = 60.0 + 120.0 * y as f32 / h as f32;
        let v = if band { light } else { light * 0.55 };
        Rgba([v as u8, (v * 0.9) as u8, (v * 0.7) as u8, 255])
    })
}

fn disc(w: u32, h: u32, cx: f32, cy: f32, r: f32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
        Luma([if d <= r { 255 } else { 0 }])
    })
}

fn mean_error(a: &RgbaImage, b: &RgbaImage, mask: &GrayImage) -> f32 {
    let (mut sum, mut n) = (0.0f32, 0usize);
    for (x, y, m) in mask.enumerate_pixels() {
        if m[0] > 127 {
            let (p, q) = (a.get_pixel(x, y), b.get_pixel(x, y));
            sum += (0..3).map(|c| (f32::from(p[c]) - f32::from(q[c])).abs()).sum::<f32>() / 3.0;
            n += 1;
        }
    }
    sum / n as f32
}

#[test]
fn removes_a_spot_and_continues_the_texture() {
    let (w, h) = (160, 120);
    let clean = stripes(w, h);
    let mask = disc(w, h, 80.0, 60.0, 9.0);
    let mut spotted = clean.clone();
    for (x, y, m) in mask.enumerate_pixels() {
        if m[0] > 127 {
            spotted.put_pixel(x, y, Rgba([250, 20, 200, 255]));
        }
    }
    let before = mean_error(&spotted, &clean, &mask);
    let healed = heal(&spotted, &mask, &HealOptions::default(), &|_| true).unwrap();
    let after = mean_error(&healed, &clean, &mask);
    assert!(after < 22.0 && after < before / 5.0, "error {before} -> {after}");
    // Nothing outside the grown mask moves.
    let far = disc(w, h, 80.0, 60.0, 14.0);
    for (x, y, m) in far.enumerate_pixels() {
        if m[0] == 0 {
            assert_eq!(healed.get_pixel(x, y), spotted.get_pixel(x, y));
        }
    }
}

#[test]
fn heals_separate_spots_each_from_its_own_surroundings() {
    let (w, h) = (200, 100);
    // Left half warm, right half cool: each spot must borrow its own side.
    let photo = RgbaImage::from_fn(w, h, |x, y| {
        let n = ((x * 7 + y * 13) % 11) as u8;
        if x < w / 2 { Rgba([200 + n, 120 + n, 60, 255]) } else { Rgba([50, 90 + n, 190 + n, 255]) }
    });
    let mut mask = disc(w, h, 40.0, 50.0, 6.0);
    for (x, y, m) in disc(w, h, 160.0, 50.0, 6.0).enumerate_pixels() {
        if m[0] > 0 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    let healed = heal(&photo, &mask, &HealOptions::default(), &|_| true).unwrap();
    let (l, r) = (healed.get_pixel(40, 50), healed.get_pixel(160, 50));
    assert!(l[0] > 150 && l[2] < 110, "left spot {l:?}");
    assert!(r[2] > 150 && r[0] < 110, "right spot {r:?}");
}

#[test]
fn reports_progress_and_cancels() {
    let photo = stripes(96, 96);
    let mask = disc(96, 96, 48.0, 48.0, 10.0);
    let err = heal(&photo, &mask, &HealOptions::default(), &|_| false).unwrap_err();
    assert_eq!(err, HealError::Cancelled);

    let last = std::sync::Mutex::new(0.0f32);
    heal(&photo, &mask, &HealOptions::default(), &|p| {
        *last.lock().unwrap() = p;
        true
    })
    .unwrap();
    assert_eq!(*last.lock().unwrap(), 1.0);
}

#[test]
fn rejects_bad_masks() {
    let photo = stripes(32, 32);
    let empty = GrayImage::new(32, 32);
    assert_eq!(heal(&photo, &empty, &HealOptions::default(), &|_| true).unwrap_err(), HealError::EmptyMask);
    let wrong = GrayImage::new(16, 16);
    assert_eq!(heal(&photo, &wrong, &HealOptions::default(), &|_| true).unwrap_err(), HealError::SizeMismatch);
}

#[test]
fn guided_fill_keeps_the_layout_it_is_given() {
    // Warm left half, cool right half; the hole straddles the boundary. The
    // guide says where the boundary goes; the fill has to keep it there and
    // fill each side with that side's own texture.
    let (w, h) = (160, 120);
    let truth = RgbaImage::from_fn(w, h, |x, y| {
        let n = ((x * 7 + y * 13) % 11) as u8 * 3;
        if x < 80 { Rgba([200 + n / 2, 120 + n, 60, 255]) } else { Rgba([50, 90 + n, 190 + n / 2, 255]) }
    });
    let mask = disc(w, h, 80.0, 60.0, 25.0);
    let mut photo = truth.clone();
    let mut guide = truth.clone();
    for (x, y, m) in mask.enumerate_pixels() {
        if m[0] > 127 {
            photo.put_pixel(x, y, Rgba([0, 255, 0, 255]));
            // The network's answer: right layout, no texture.
            guide.put_pixel(x, y, if x < 80 { Rgba([205, 135, 60, 255]) } else { Rgba([50, 105, 200, 255]) });
        }
    }
    let filled = nabu_retouch::fill_guided(&photo, &guide, &mask, &Default::default(), &|_| true).unwrap();
    let (l, r) = (filled.get_pixel(70, 60), filled.get_pixel(90, 60));
    assert!(l[0] > 150 && l[2] < 110, "left of the boundary {l:?}");
    assert!(r[2] > 150 && r[0] < 110, "right of the boundary {r:?}");
    assert!(mean_error(&filled, &truth, &mask) < 30.0);
}
