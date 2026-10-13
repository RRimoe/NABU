//! Panorama stitching.
//!
//! Turns a set of overlapping photos taken from one spot into one picture.
//! Photos taken walking along a flat subject (a train, a wall) are recognised
//! and laid on the subject's plane instead ([`planar`]):
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
pub mod planar;
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
use warp::{footprint, planar_rect, rect_of, warp_planar, Rect, Surface, Warped};

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
    /// The photos are given in the order they were shot (left to right, or
    /// along the sweep). Only neighbours in that order are matched, which
    /// keeps repeated patterns (windows, waves, railings) from tying together
    /// photos that do not overlap. If that leaves a photo out, every pair is
    /// tried after all.
    pub in_order: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            projection: None,
            max_megapixels: 60.0,
            crop: true,
            registration_megapixels: 1.0,
            fine_alignment: true,
            in_order: false,
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
    /// Per photo, its place on the subject's plane (centred full-resolution
    /// pixels to the reference photo's), for photos taken walking along a
    /// flat subject. None when no plane fits.
    pub planar: Option<Vec<Matrix3<f64>>>,
    /// The planar placement explains the matches clearly better than
    /// turning on the spot: the photos were taken walking.
    pub walking: bool,
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

/// How far apart in the given order two photos may be and still be matched:
/// next to each other, or one apart (a wide lens overlaps by more than half).
const IN_ORDER_GAP: usize = 2;

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
    let mut pairs = align::match_pairs(&features, threshold, options.in_order.then_some(IN_ORDER_GAP), |v| {
        if !progress(Stage::Matching, v) {
            cancelled = true;
        }
    });
    if cancelled {
        return Err(StitchError::Cancelled);
    }
    if options.in_order && align::largest_component(n, &pairs).len() < n {
        // The order given does not hold together: let the photos find each
        // other instead.
        debug("photos given in order do not connect; matching every pair");
        pairs = align::match_pairs(&features, threshold, None, |v| {
            if !progress(Stage::Matching, v) {
                cancelled = true;
            }
        });
        if cancelled {
            return Err(StitchError::Cancelled);
        }
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

    // The other explanation: the camera moved along a flat subject. Decided
    // on the median residual of each model, so that the ground (which neither
    // model places right when the camera moves) does not decide.
    let plane = planar::fit(n, &members, &used, root, huber);
    let walking = plane.as_ref().is_some_and(|plane| {
        let flat = planar::median_residual(plane, &used);
        let turning = rotation_median_residual(&best.lens, &best.rotations, &used);
        // Turning on the spot explains a walk only with an absurdly long
        // lens: sliding past a wall looks like a tiny turn of a telephoto.
        // Phones do not shoot through a lens that narrow, so a solution that
        // needs one, where the plane fits as well, is a walk.
        let widest = features.iter().map(|f| f.width.max(f.height)).max().unwrap_or(1) as f64;
        let view = 2.0 * (widest / 2.0 / best.lens.focal).atan().to_degrees();
        debug(&format!("median residual: planar {flat:.3}px, rotation {turning:.3}px; view {view:.1}°"));
        flat < turning * WALKING_RATIO || (view < NARROWEST_VIEW && flat <= turning)
    });
    let planar = plane.map(|plane| {
        // From centred working pixels to centred full-resolution ones.
        let s = Matrix3::new(work_scale, 0.0, 0.0, 0.0, work_scale, 0.0, 0.0, 0.0, 1.0);
        let s_inv = Matrix3::new(1.0 / work_scale, 0.0, 0.0, 0.0, 1.0 / work_scale, 0.0, 0.0, 0.0, 1.0);
        plane.transforms.iter().map(|t| s_inv * t * s).collect::<Vec<_>>()
    });

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
    Ok(Alignment { lens, rotations, members, rms, planar, walking })
}

/// How much better (as a ratio of median residuals) the planar placement
/// must fit before the photos are taken to have been shot walking.
const WALKING_RATIO: f64 = 0.6;

/// Narrowest view (degrees, across the long side of a photo) that a turn on
/// the spot is believed for when the plane fits just as well.
const NARROWEST_VIEW: f64 = 25.0;

/// Median residual of the rotation model over the matches, working pixels.
fn rotation_median_residual(lens: &Lens, rotations: &[Matrix3<f64>], pairs: &[Pair]) -> f64 {
    let mut all = Vec::new();
    for p in pairs {
        // As the bundle adjustment measures it: b seen from camera i lands on a.
        let rij = rotations[p.i].transpose() * rotations[p.j];
        for (a, b) in p.a.iter().zip(&p.b) {
            if let Some(q) = lens.project(&(rij * lens.ray(*b))) {
                all.push(((q[0] - a[0]).powi(2) + (q[1] - a[1]).powi(2)).sqrt());
            }
        }
    }
    if all.is_empty() {
        return f64::INFINITY;
    }
    all.sort_by(f64::total_cmp);
    all[all.len() / 2]
}

/// Surface rectangle of one photo, if it projects at all.
fn place(surface: &Surface, lens: &Lens, rotation: &Matrix3<f64>, width: usize, height: usize) -> Option<Rect> {
    footprint(surface, lens, rotation, width, height).map(rect_of)
}

/// Widest field of view, in degrees, that is still shown flat by default.
const FLAT_LIMIT: f64 = 75.0;

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
    // A flat projection keeps straight lines straight but stretches the
    // edges: past about 75° the photos at the sides come out visibly pulled
    // and enlarged, which reads as distortion. Wider sets go on a cylinder.
    let projection = if h <= FLAT_LIMIT && v <= FLAT_LIMIT && flat_possible {
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
    let auto = if alignment.walking && alignment.planar.is_some() { Projection::Planar } else { auto };
    let mut projection = options.projection.unwrap_or(auto);
    if projection == Projection::Planar && alignment.planar.is_none() {
        projection = Projection::Rectilinear;
    }
    // Where each photo goes, at a given output scale: on the plane, or on the
    // projection surface.
    let planes = alignment.planar.as_ref().filter(|_| projection == Projection::Planar);
    let to_plane = |m: usize, scale: f64| -> Matrix3<f64> {
        let t = planes.map(|p| p[m]).unwrap_or_else(Matrix3::identity);
        let up = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
        let down = Matrix3::new(1.0 / scale, 0.0, 0.0, 0.0, 1.0 / scale, 0.0, 0.0, 0.0, 1.0);
        // Photo pixels at `scale`, centred, to plane pixels at `scale`.
        up * t * down
    };

    // Output scale: one output pixel per input pixel at the centre of the
    // projection, unless that exceeds the size limit.
    let full = Surface { projection, scale: lens.focal };
    let mut rects = Vec::new();
    for &m in members {
        let (w, h) = (images[m].width() as usize, images[m].height() as usize);
        let r = if planes.is_some() {
            planar_rect(&to_plane(m, 1.0), w, h)
        } else {
            place(&full, &lens, &alignment.rotations[m], w, h)
                .ok_or_else(|| StitchError::Layout(format!("photo {m} does not project")))?
        };
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
            if planes.is_some() {
                let t = to_plane(m, seam_scale);
                return warp_planar(&img, &t, planar_rect(&t, img.width() as usize, img.height() as usize));
            }
            let rect = place(&seam_surface, &seam_lens, &alignment.rotations[m], img.width() as usize, img.height() as usize)
                .unwrap_or(Rect { x: 0, y: 0, width: 1, height: 1 });
            warp::warp(&img, &seam_lens, &alignment.rotations[m], &seam_surface, rect)
        })
        .collect();
    let transfer = Transfer::new();
    let apply_gain = |rgb: &mut [[f32; 3]], width: usize, height: usize, gain: [f32; 3], field: Option<&exposure::GainField>| {
        rgb.par_chunks_mut(width.max(1)).enumerate().for_each(|(y, row)| {
            let fy = (y as f32 + 0.5) / height as f32;
            for (x, p) in row.iter_mut().enumerate() {
                let local = field.map_or([1.0; 3], |f| f.at((x as f32 + 0.5) / width as f32, fy));
                for c in 0..3 {
                    p[c] = transfer.srgb(transfer.linear(p[c]) * gain[c] * local[c]);
                }
            }
        });
    };
    // One gain per photo first, then what varies across each photo
    // (vignetting, the phone's local tone mapping) on top.
    let gains = exposure::gains(&small);
    for (w, g) in small.iter_mut().zip(&gains) {
        apply_gain(&mut w.rgb, w.rect.width, w.rect.height, *g, None);
    }
    let fields = exposure::block_gains(&small);
    for (w, f) in small.iter_mut().zip(&fields) {
        apply_gain(&mut w.rgb, w.rect.width, w.rect.height, [1.0; 3], Some(f));
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
        if planes.is_some() {
            out_rects.push(planar_rect(&to_plane(m, compose_scale), w, h));
            continue;
        }
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
        let mut warped = if planes.is_some() {
            warp_planar(&img, &to_plane(m, compose_scale), out_rects[k])
        } else {
            warp::warp(&img, &out_lens, &alignment.rotations[m], &out_surface, out_rects[k])
        };
        let t1 = t0.elapsed().as_secs_f64();
        apply_gain(&mut warped.rgb, warped.rect.width, warped.rect.height, gains[k], Some(&fields[k]));
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
