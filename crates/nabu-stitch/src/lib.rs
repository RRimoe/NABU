//! Panorama stitching.
//!
//! Turns a set of overlapping photos taken from one spot into one picture:
//!
//! 1. **Features** — SIFT on a working copy of each photo ([`sift`]).
//! 2. **Matching** — mutual nearest neighbours with Lowe's ratio test, every
//!    pair verified by RANSAC and Brown & Lowe's inlier test ([`align`]).
//! 3. **Alignment** — a rotation per photo and one shared lens (focal length
//!    and radial distortion) from a robust bundle adjustment over all matches,
//!    then refined against the full-resolution pixels ([`refine`]), and the
//!    horizon levelled.
//! 4. **Projection** — rectilinear, cylindrical or spherical, chosen from the
//!    field of view unless asked for ([`warp`]).
//! 5. **Exposure** — per-photo, per-channel gains in linear light ([`exposure`]).
//! 6. **Seams** — minimum graph cuts through the overlaps ([`seam`]).
//! 7. **Blending** — multi-band, so seams vanish at every scale ([`blend`]).
//! 8. **Crop** — optionally to the largest rectangle of picture ([`crop`]).

pub mod align;
pub mod blend;
pub mod camera;
pub mod crop;
pub mod exposure;
pub mod homography;
pub mod imgops;
pub mod matching;
pub mod refine;
pub mod seam;
pub mod sift;
pub mod warp;

use align::{Freedom, ImageFeatures, Pair};
use camera::Lens;
use image::RgbaImage;
use imgops::{downscale, to_gray, Transfer};
use nalgebra::Matrix3;
use rayon::prelude::*;
pub use warp::Projection;
use warp::{footprint, rect_of, Rect, Surface, Warped};

#[derive(Clone, Debug)]
pub struct Options {
    /// None chooses from the field of view.
    pub projection: Option<Projection>,
    /// Largest output, in megapixels; bigger panoramas are scaled down.
    pub max_megapixels: f64,
    /// Crop to the largest rectangle that has picture everywhere.
    pub crop: bool,
    /// Size of the copies features are found on, in megapixels.
    pub registration_megapixels: f64,
    /// Re-measure the matches on the full-resolution photos.
    pub fine_alignment: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            projection: None,
            max_megapixels: 60.0,
            crop: true,
            registration_megapixels: 1.0,
            fine_alignment: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Features,
    Matching,
    Aligning,
    Refining,
    Exposure,
    Seams,
    Blending,
}

#[derive(Debug)]
pub enum StitchError {
    /// Fewer than two photos were given.
    TooFewImages,
    /// No two photos could be matched to each other.
    NoOverlap,
    /// The caller asked to stop.
    Cancelled,
    /// The panorama could not be laid out (degenerate geometry).
    Layout(String),
}

impl std::fmt::Display for StitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StitchError::TooFewImages => write!(f, "at least two photos are needed"),
            StitchError::NoOverlap => write!(f, "the photos do not overlap"),
            StitchError::Cancelled => write!(f, "cancelled"),
            StitchError::Layout(m) => write!(f, "layout failed: {m}"),
        }
    }
}

impl std::error::Error for StitchError {}

pub struct Panorama {
    pub image: RgbaImage,
    /// Indices of the input photos that are in the panorama, and those left out
    /// because they did not connect to the rest.
    pub used: Vec<usize>,
    pub dropped: Vec<usize>,
    pub projection: Projection,
    /// Alignment error, root mean square, in full-resolution pixels.
    pub rms: f64,
    /// Horizontal and vertical field of view, degrees.
    pub field_of_view: (f64, f64),
    /// Focal length of the lens in full-resolution pixels, and its distortion.
    pub lens: Lens,
}

/// Where alignment ended up, in full-resolution pixels.
pub struct Alignment {
    pub lens: Lens,
    pub rotations: Vec<Matrix3<f64>>,
    pub members: Vec<usize>,
    pub rms: f64,
}

/// Reports progress; returning false cancels.
pub type Progress<'a> = &'a mut dyn FnMut(Stage, f32) -> bool;

fn step(progress: &mut dyn FnMut(Stage, f32) -> bool, stage: Stage, value: f32) -> Result<(), StitchError> {
    if progress(stage, value) {
        Ok(())
    } else {
        Err(StitchError::Cancelled)
    }
}

/// Stitches photos into a panorama.
pub fn stitch(images: &[RgbaImage], options: &Options, progress: Progress) -> Result<Panorama, StitchError> {
    let alignment = align_images(images, options, progress)?;
    compose(images, &alignment, options, progress)
}

/// Steps 1–3: features, matching and the camera solution.
pub fn align_images(images: &[RgbaImage], options: &Options, progress: Progress) -> Result<Alignment, StitchError> {
    let n = images.len();
    if n < 2 {
        return Err(StitchError::TooFewImages);
    }

    // One scale for every photo, so pixel units agree between them.
    let largest = images.iter().map(|i| i.width() as f64 * i.height() as f64).fold(0.0, f64::max);
    let work_scale = (options.registration_megapixels * 1e6 / largest).sqrt().min(1.0);

    step(progress, Stage::Features, 0.0)?;
    let mut features = Vec::with_capacity(n);
    for (k, image) in images.iter().enumerate() {
        let small = downscale(image, work_scale);
        let gray = to_gray(&small);
        let mut f = sift::detect(&gray, 0.04, 4000);
        // Skies, water and snow: try harder before giving up on a photo.
        if f.len() < 400 {
            f = sift::detect(&gray, 0.012, 4000);
        }
        let (w, h) = (small.width() as usize, small.height() as usize);
        let points = f
            .keypoints
            .iter()
            .map(|k| [k.x as f64 - (w as f64 - 1.0) / 2.0, k.y as f64 - (h as f64 - 1.0) / 2.0])
            .collect();
        features.push(ImageFeatures { features: f, points, width: w, height: h });
        step(progress, Stage::Features, (k + 1) as f32 / n as f32)?;
    }

    let diag = features
        .iter()
        .map(|f| ((f.width * f.width + f.height * f.height) as f64).sqrt())
        .fold(0.0, f64::max);
    let threshold = (diag * 0.002).max(2.0);
    let mut cancelled = false;
    let mut pairs = align::match_pairs(&features, threshold, |v| {
        if !progress(Stage::Matching, v) {
            cancelled = true;
        }
    });
    if cancelled {
        return Err(StitchError::Cancelled);
    }

    step(progress, Stage::Aligning, 0.0)?;
    let fallback_focal = features.iter().map(|f| f.width.max(f.height) as f64).fold(0.0, f64::max);
    let huber = threshold * 0.5;
    let mut solution = None;

    // Solve, drop the pairs the global model cannot explain (repeated
    // patterns that fooled RANSAC), solve again.
    for round in 0..3 {
        let members = align::largest_component(n, &pairs);
        if members.len() < 2 {
            return Err(StitchError::NoOverlap);
        }
        let used: Vec<Pair> = pairs
            .iter()
            .filter(|p| members.contains(&p.i) && members.contains(&p.j))
            .cloned()
            .collect();
        let (lens, rotations, root) = align::initial_cameras(n, &members, &used, fallback_focal);
        let solve = |freedom: Freedom, lens: Lens, rotations: &[Matrix3<f64>]| {
            align::bundle_adjust(lens, rotations, &used, &members, root, freedom, huber)
        };
        let simple = solve(Freedom { focal: true, k1: false, k2: false }, lens, &rotations);
        let total_points: usize = used.iter().map(|p| p.a.len()).sum();
        // Keep the distortion only when it explains the data clearly better,
        // and never from two photos, which cannot tell it from focal length.
        let best = if members.len() >= 3 {
            let with_lens = solve(
                Freedom { focal: true, k1: true, k2: total_points > 3000 },
                simple.lens,
                &simple.rotations,
            );
            if with_lens.rms < simple.rms * 0.97 { with_lens } else { simple }
        } else {
            simple
        };

        let mut sorted = best.pair_rms.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted.get(sorted.len() / 2).cloned().unwrap_or(0.0);
        let limit = (median * 4.0).max(threshold);
        let bad: Vec<usize> = (0..used.len()).filter(|&k| best.pair_rms[k] > limit).collect();
        step(progress, Stage::Aligning, (round + 1) as f32 / 3.0)?;

        if bad.is_empty() || round == 2 {
            solution = Some((best, members, used, root));
            break;
        }
        let bad_pairs: Vec<(usize, usize)> = bad.iter().map(|&k| (used[k].i, used[k].j)).collect();
        pairs.retain(|p| !bad_pairs.contains(&(p.i, p.j)));
    }
    let (best, members, used, root) = solution.expect("loop always sets a solution");

    // Back to full-resolution pixels.
    let mut lens = best.lens.scaled(1.0 / work_scale);
    let mut rotations = best.rotations;
    let mut rms = best.rms / work_scale;
    debug(&format!(
        "aligned {} photos, {} pairs, work scale {:.3}, rms {:.3}px work / {:.3}px full, focal {:.1}, k1 {:.4} k2 {:.4}",
        members.len(), used.len(), work_scale, best.rms, rms, lens.focal, lens.k1, lens.k2
    ));

    if options.fine_alignment {
        step(progress, Stage::Refining, 0.0)?;
        let refined = refine::refine(images, &lens, &rotations, &used, &members, root, work_scale, huber / work_scale);
        debug(&format!("refined: {:?}", refined.as_ref().map(|r| (r.rms, r.lens))));
        if let Some(refined) = refined {
            lens = refined.lens;
            rotations = refined.rotations;
            rms = refined.rms;
        }
        step(progress, Stage::Refining, 1.0)?;
    }

    align::straighten(&mut rotations, &members);
    Ok(Alignment { lens, rotations, members, rms })
}

/// Surface rectangle of one photo, if it projects at all.
fn place(surface: &Surface, lens: &Lens, rotation: &Matrix3<f64>, width: usize, height: usize) -> Option<Rect> {
    footprint(surface, lens, rotation, width, height).map(rect_of)
}

fn choose_projection(alignment: &Alignment, images: &[RgbaImage]) -> (Projection, (f64, f64)) {
    let lens = &alignment.lens;
    let fov = |projection: Projection| -> Option<(f64, f64)> {
        let surface = Surface { projection, scale: 1.0 };
        let mut all: Option<(f64, f64, f64, f64)> = None;
        for &m in &alignment.members {
            let (w, h) = (images[m].width() as usize, images[m].height() as usize);
            let b = footprint(&surface, lens, &alignment.rotations[m], w, h)?;
            all = Some(match all {
                None => b,
                Some(a) => (a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3)),
            });
        }
        all.map(|(u0, u1, v0, v1)| ((u1 - u0).to_degrees(), (v1 - v0).to_degrees()))
    };
    let spherical = fov(Projection::Spherical).unwrap_or((360.0, 180.0));
    let (h, v) = spherical;
    // Every photo must face forward for a flat projection to exist at all.
    let flat_possible = alignment
        .members
        .iter()
        .all(|&m| alignment.rotations[m].column(2)[2] > 0.35);
    let projection = if h <= 110.0 && v <= 110.0 && flat_possible {
        Projection::Rectilinear
    } else if v <= 100.0 {
        Projection::Cylindrical
    } else {
        Projection::Spherical
    };
    (projection, spherical)
}

/// Steps 4–8: projection, exposure, seams, blending and crop.
pub fn compose(
    images: &[RgbaImage],
    alignment: &Alignment,
    options: &Options,
    progress: Progress,
) -> Result<Panorama, StitchError> {
    let members = &alignment.members;
    let lens = alignment.lens;
    let (auto, field_of_view) = choose_projection(alignment, images);
    let projection = options.projection.unwrap_or(auto);

    // Output scale: one output pixel per input pixel at the centre of the
    // projection, unless that exceeds the size limit.
    let full = Surface { projection, scale: lens.focal };
    let mut rects = Vec::new();
    for &m in members {
        let (w, h) = (images[m].width() as usize, images[m].height() as usize);
        let r = place(&full, &lens, &alignment.rotations[m], w, h)
            .ok_or_else(|| StitchError::Layout(format!("photo {m} does not project")))?;
        rects.push(r);
    }
    let canvas = rects.iter().skip(1).fold(rects[0], |a, b| a.union(b));
    let area = canvas.width as f64 * canvas.height as f64;
    let compose_scale = (options.max_megapixels * 1e6 / area).sqrt().min(1.0);

    // Seams and exposure are worked out on small copies.
    let mean_area = members
        .iter()
        .map(|&m| images[m].width() as f64 * images[m].height() as f64)
        .sum::<f64>()
        / members.len() as f64;
    let seam_scale = (0.3e6 / mean_area).sqrt().min(compose_scale);

    step(progress, Stage::Exposure, 0.0)?;
    let seam_surface = Surface { projection, scale: lens.focal * seam_scale };
    let seam_lens = lens.scaled(seam_scale);
    let mut small: Vec<Warped> = members
        .iter()
        .map(|&m| {
            let img = downscale(&images[m], seam_scale);
            let rect = place(&seam_surface, &seam_lens, &alignment.rotations[m], img.width() as usize, img.height() as usize)
                .unwrap_or(Rect { x: 0, y: 0, width: 1, height: 1 });
            warp::warp(&img, &seam_lens, &alignment.rotations[m], &seam_surface, rect)
        })
        .collect();
    let gains = exposure::gains(&small);
    let transfer = Transfer::new();
    let apply_gain = |rgb: &mut [[f32; 3]], gain: [f32; 3]| {
        rgb.par_iter_mut().for_each(|p| {
            for c in 0..3 {
                p[c] = transfer.srgb(transfer.linear(p[c]) * gain[c]);
            }
        });
    };
    for (w, g) in small.iter_mut().zip(&gains) {
        apply_gain(&mut w.rgb, *g);
    }
    step(progress, Stage::Exposure, 1.0)?;

    step(progress, Stage::Seams, 0.0)?;
    seam::find_seams(&mut small);
    step(progress, Stage::Seams, 1.0)?;

    // Full-size composition.
    let out_surface = Surface { projection, scale: lens.focal * compose_scale };
    let out_lens = lens.scaled(compose_scale);
    let mut out_rects = Vec::new();
    for &m in members {
        let (w, h) = (
            ((images[m].width() as f64 * compose_scale).round() as usize).max(1),
            ((images[m].height() as f64 * compose_scale).round() as usize).max(1),
        );
        out_rects.push(place(&out_surface, &out_lens, &alignment.rotations[m], w, h).unwrap_or(Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        }));
    }
    let out_canvas = out_rects.iter().skip(1).fold(out_rects[0], |a, b| a.union(b));
    let blend_width = ((out_canvas.width * out_canvas.height) as f64).sqrt() * 0.05;
    let bands = ((blend_width.max(2.0).log2().ceil() as i64) - 1).clamp(1, 7) as usize;
    let mut blender = blend::Blender::new(out_canvas, bands);
    let ratio = seam_scale / compose_scale;

    for (k, &m) in members.iter().enumerate() {
        step(progress, Stage::Blending, k as f32 / members.len() as f32)?;
        let t0 = std::time::Instant::now();
        let img = downscale(&images[m], compose_scale);
        let mut warped = warp::warp(&img, &out_lens, &alignment.rotations[m], &out_surface, out_rects[k]);
        let t1 = t0.elapsed().as_secs_f64();
        apply_gain(&mut warped.rgb, gains[k]);
        let t2 = t0.elapsed().as_secs_f64();

        // The seam mask, grown by a pixel and brought up to this resolution.
        let sm = &small[k];
        let dilated = dilate(&sm.mask, sm.rect.width, sm.rect.height);
        let r = warped.rect;
        let mut mask = vec![0u8; r.width * r.height];
        mask.par_chunks_mut(r.width).enumerate().for_each(|(y, row)| {
            let sy = (((r.y + y as i64) as f64 + 0.5) * ratio - 0.5).round() as i64 - sm.rect.y;
            for (x, v) in row.iter_mut().enumerate() {
                let sx = (((r.x + x as i64) as f64 + 0.5) * ratio - 0.5).round() as i64 - sm.rect.x;
                if sx >= 0 && sy >= 0 && (sx as usize) < sm.rect.width && (sy as usize) < sm.rect.height {
                    *v = dilated[sy as usize * sm.rect.width + sx as usize];
                }
            }
        });
        for (v, c) in mask.iter_mut().zip(&warped.mask) {
            *v &= *c;
        }
        let t3 = t0.elapsed().as_secs_f64();
        blender.feed(&warped.rgb, &warped.mask, &mask, warped.rect);
        debug(&format!("photo {m}: warp {t1:.2}s gain {:.2}s mask {:.2}s feed {:.2}s", t2 - t1, t3 - t2, t0.elapsed().as_secs_f64() - t3));
    }
    step(progress, Stage::Blending, 1.0)?;

    let t0 = std::time::Instant::now();
    let (rgb, weight) = blender.finish();
    debug(&format!("collapse {:.2}s, {bands} bands", t0.elapsed().as_secs_f64()));
    let (w, h) = (out_canvas.width, out_canvas.height);
    let covered: Vec<bool> = weight.iter().map(|v| *v > 1e-3).collect();
    let (cx, cy, cw, ch) = if options.crop {
        crop::largest_rectangle(&covered, w, h).unwrap_or((0, 0, w, h))
    } else {
        (0, 0, w, h)
    };

    let mut out = RgbaImage::new(cw as u32, ch as u32);
    out.as_mut()
        .par_chunks_mut(cw * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..cw {
                let k = (cy + y) * w + cx + x;
                let p = rgb[k];
                let o = x * 4;
                if covered[k] {
                    row[o] = (p[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                    row[o + 1] = (p[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                    row[o + 2] = (p[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                    row[o + 3] = 255;
                }
            }
        });

    let n = images.len();
    Ok(Panorama {
        image: out,
        used: members.clone(),
        dropped: (0..n).filter(|i| !members.contains(i)).collect(),
        projection,
        rms: alignment.rms,
        field_of_view,
        lens,
    })
}

/// 3×3 dilation of a 0/1 mask.
fn dilate(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] == 0 {
                continue;
            }
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        out[ny as usize * w + nx as usize] = 1;
                    }
                }
            }
        }
    }
    out
}

/// Diagnostics on standard error when `NABU_STITCH_DEBUG` is set.
pub(crate) fn debug(message: &str) {
    if std::env::var_os("NABU_STITCH_DEBUG").is_some() {
        eprintln!("[stitch] {message}");
    }
}
