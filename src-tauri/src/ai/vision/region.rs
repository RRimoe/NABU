// src-tauri/src/ai/vision/region.rs
//! The shape the user drew, treated as what it is: a statement of intent.
//!
//! A lasso says two things, and the engine used to hear only the first. It says
//! "the thing I mean is in here" — which is a prompt — and it says "and it is
//! *not* out there" — which is a constraint. Passing only the bounding box to a
//! segmenter throws both away: on anything that is not roughly square, the
//! centre of the box is not even inside the drawing.
//!
//! So the outline is rasterised once, and used twice. Prompt points are chosen
//! at the places furthest from its border, where they are certain to be on the
//! object rather than on its edge. And whatever the segmenter answers is
//! clipped back to the drawing, so a model that misreads the picture can be
//! wrong about the shape of a hand but never about which half of the photograph
//! the hand is in.

use image::GrayImage;

/// Longest side of the grid the outline is reasoned about on.
///
/// Nothing here needs resolution: the prompt points are approximate by nature
/// and the constraint is deliberately generous. A fixed small grid also keeps
/// the cost independent of the size of the photograph.
const WORK_SIDE: u32 = 256;

/// How far past the drawing the segmenter is still believed, as a fraction of
/// the drawing's longest side. People undershoot when they trace, so a mask
/// that reaches a little beyond the line is usually right.
const TOLERANCE: f32 = 0.12;

/// A region the user drew, ready to prompt and to constrain with.
pub struct Region {
    /// Filled outline: 255 inside, 0 outside, on the working grid.
    filled: GrayImage,
    /// Scale from source pixels to the working grid.
    scale: f32,
    /// Bounding box in source pixels.
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

/// True when the point is inside the polygon, by the even-odd rule.
fn contains(polygon: &[[f32; 2]], x: f32, y: f32) -> bool {
    let mut inside = false;
    let mut j = polygon.len().saturating_sub(1);

    for i in 0..polygon.len() {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);

        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Chamfer distance to the nearest empty pixel using Borgefors 3-4 metric, in working-grid units.
///
/// Provides a quasi-Euclidean distance transform (<8% error) across orthogonal
/// and diagonal directions, accurately pinpointing the true poles of inaccessibility.
fn distance_inside(filled: &GrayImage) -> Vec<u16> {
    let (w, h) = filled.dimensions();
    let (wi, hi) = (w as usize, h as usize);
    let mut distance = vec![0u16; wi * hi];
    let raw = filled.as_raw();

    const FAR: u16 = u16::MAX / 2;

    for y in 0..hi {
        let row = y * wi;
        for x in 0..wi {
            distance[row + x] = if raw[row + x] > 127 {
                FAR
            } else {
                0
            };
        }
    }

    // Forward sweep: Borgefors 3-4 chamfer metric
    for y in 0..hi {
        for x in 0..wi {
            let i = y * wi + x;
            if distance[i] == 0 {
                continue;
            }
            let mut best = distance[i];
            if x > 0 {
                best = best.min(distance[i - 1].saturating_add(3));
            }
            if y > 0 {
                best = best.min(distance[i - wi].saturating_add(3));
                if x > 0 {
                    best = best.min(distance[i - wi - 1].saturating_add(4));
                }
                if x + 1 < wi {
                    best = best.min(distance[i - wi + 1].saturating_add(4));
                }
            }
            distance[i] = best;
        }
    }

    // Backward sweep: Borgefors 3-4 chamfer metric
    for y in (0..hi).rev() {
        for x in (0..wi).rev() {
            let i = y * wi + x;
            if distance[i] == 0 {
                continue;
            }
            let mut best = distance[i];
            if x + 1 < wi {
                best = best.min(distance[i + 1].saturating_add(3));
            }
            if y + 1 < hi {
                best = best.min(distance[i + wi].saturating_add(3));
                if x > 0 {
                    best = best.min(distance[i + wi - 1].saturating_add(4));
                }
                if x + 1 < wi {
                    best = best.min(distance[i + wi + 1].saturating_add(4));
                }
            }
            distance[i] = best;
        }
    }

    distance
}

impl Region {
    /// Rasterises an outline given in source pixels.
    ///
    /// Returns `None` for a degenerate drawing — fewer than three points, or a
    /// shape that encloses nothing — because there is no region to speak of and
    /// the caller should fall back to the plain bounding box.
    pub fn from_polygon(polygon: &[[f32; 2]], src_w: u32, src_h: u32) -> Option<Region> {
        if polygon.len() < 3 || src_w == 0 || src_h == 0 {
            return None;
        }

        let mut x0 = f32::MAX;
        let mut y0 = f32::MAX;
        let mut x1 = f32::MIN;
        let mut y1 = f32::MIN;
        for point in polygon {
            x0 = x0.min(point[0]);
            y0 = y0.min(point[1]);
            x1 = x1.max(point[0]);
            y1 = y1.max(point[1]);
        }
        if !(x1 > x0 && y1 > y0) {
            return None;
        }

        let scale = WORK_SIDE as f32 / src_w.max(src_h) as f32;
        let w = ((src_w as f32 * scale).round() as u32).clamp(1, WORK_SIDE);
        let h = ((src_h as f32 * scale).round() as u32).clamp(1, WORK_SIDE);

        // Only the drawing's own box is scanned: filling a 256x256 grid where
        // the shape occupies a twentieth of it would spend the whole budget on
        // pixels that are obviously outside.
        let bx0 = ((x0 * scale).floor().max(0.0)) as u32;
        let by0 = ((y0 * scale).floor().max(0.0)) as u32;
        let bx1 = ((x1 * scale).ceil() as u32).min(w.saturating_sub(1));
        let by1 = ((y1 * scale).ceil() as u32).min(h.saturating_sub(1));

        let mut filled = GrayImage::new(w, h);
        let mut any = false;
        for y in by0..=by1 {
            for x in bx0..=bx1 {
                // Sampled at the centre of the cell, in source coordinates.
                let sx = (x as f32 + 0.5) / scale;
                let sy = (y as f32 + 0.5) / scale;
                if contains(polygon, sx, sy) {
                    filled.put_pixel(x, y, image::Luma([255]));
                    any = true;
                }
            }
        }

        if !any {
            return None;
        }

        Some(Region { filled, scale, x0, y0, x1, y1 })
    }

    /// Points to prompt the segmenter with, in source pixels.
    ///
    /// Chosen at the places furthest from the outline — the poles of
    /// inaccessibility of the drawing — because a point near the border is as
    /// likely to sit on the background as on the object, and one bad point is
    /// enough to send the segmenter after the wrong thing.
    pub fn prompt_points(&self, count: usize) -> Vec<[f32; 2]> {
        let (w, h) = self.filled.dimensions();
        let (wi, hi) = (w as usize, h as usize);
        let distance = distance_inside(&self.filled);

        let mut taken: Vec<(usize, usize, u16)> = Vec::with_capacity(count);

        for _ in 0..count {
            let mut best: Option<(usize, usize, u16)> = None;

            for y in 0..hi {
                for x in 0..wi {
                    let d = distance[y * wi + x];
                    if d == 0 {
                        continue;
                    }

                    // Spread the points out: a cluster of five in the middle
                    // says no more than one point in the middle.
                    let spacing = taken.iter().all(|(tx, ty, td)| {
                        let dx = *tx as i64 - x as i64;
                        let dy = *ty as i64 - y as i64;
                        let apart = ((dx * dx + dy * dy) as f64).sqrt();
                        let min_radius = (*td as f64 / 3.0).max(2.0);
                        apart >= min_radius
                    });
                    if !spacing {
                        continue;
                    }

                    if best.map_or(true, |(_, _, bd)| d > bd) {
                        best = Some((x, y, d));
                    }
                }
            }

            match best {
                Some(point) => taken.push(point),
                None => break,
            }
        }

        taken
            .into_iter()
            .map(|(x, y, _)| {
                [
                    (x as f32 + 0.5) / self.scale,
                    (y as f32 + 0.5) / self.scale,
                ]
            })
            .collect()
    }

    /// Fraction of the drawing that a mask covers, and how far it spills out.
    ///
    /// The two numbers together say whether the segmenter answered the question
    /// that was asked: a good answer fills a decent part of the drawing and
    /// stays near it.
    pub fn agreement(&self, mask: &GrayImage, threshold: u8) -> (f32, f32) {
        let (mw, mh) = mask.dimensions();
        if mw == 0 || mh == 0 {
            return (0.0, 0.0);
        }

        let (w, h) = self.filled.dimensions();
        let mut drawn = 0u64;
        let mut inside = 0u64;
        let mut outside = 0u64;

        for y in 0..h {
            for x in 0..w {
                let is_drawn = self.filled.get_pixel(x, y)[0] > 127;
                if is_drawn {
                    drawn += 1;
                }

                // Same relative position in the mask, which is at source size.
                let mx = ((x as f32 + 0.5) / self.scale).min(mw as f32 - 1.0) as u32;
                let my = ((y as f32 + 0.5) / self.scale).min(mh as f32 - 1.0) as u32;
                if mask.get_pixel(mx, my)[0] <= threshold {
                    continue;
                }

                if is_drawn {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
        }

        if drawn == 0 {
            return (0.0, 0.0);
        }
        (inside as f32 / drawn as f32, outside as f32 / drawn as f32)
    }

    /// Clips a mask to the drawing, allowing a margin for an honest overshoot.
    ///
    /// This is the guarantee that matters: whatever the segmenter believes, it
    /// cannot erase something on the far side of the photograph. Being wrong
    /// about the exact outline of a hand is a bad result; being wrong about
    /// which half of the picture the hand is in is a broken product.
    pub fn clip(&self, mask: &mut GrayImage) {
        let (mw, mh) = mask.dimensions();
        let longest = (self.x1 - self.x0).max(self.y1 - self.y0);
        let margin = (longest * TOLERANCE).max(8.0);

        let x0 = self.x0 - margin;
        let y0 = self.y0 - margin;
        let x1 = self.x1 + margin;
        let y1 = self.y1 + margin;

        let grown = grow(&self.filled, (margin * self.scale).round().max(1.0) as u32);

        for y in 0..mh {
            for x in 0..mw {
                let fx = x as f32;
                let fy = y as f32;

                // Outside the grown box is settled without touching the grid.
                if fx < x0 || fx > x1 || fy < y0 || fy > y1 {
                    mask.put_pixel(x, y, image::Luma([0]));
                    continue;
                }

                let gx = ((fx * self.scale) as u32).min(grown.width().saturating_sub(1));
                let gy = ((fy * self.scale) as u32).min(grown.height().saturating_sub(1));
                if grown.get_pixel(gx, gy)[0] == 0 {
                    mask.put_pixel(x, y, image::Luma([0]));
                }
            }
        }
    }

    /// The drawing itself as a mask at source resolution, for the fallback.
    pub fn as_mask(&self, width: u32, height: u32) -> GrayImage {
        image::imageops::resize(
            &self.filled,
            width,
            height,
            image::imageops::FilterType::Triangle,
        )
    }
}

/// Least of the drawing a mask must cover before it is believed.
///
/// A segmenter that answers with a sliver of what was circled has latched onto
/// a detail — a fingernail instead of a hand — and the drawing itself is the
/// better answer.
const MIN_COVERAGE: f32 = 0.18;

/// Most a mask may spill beyond the drawing, as a multiple of its area.
///
/// Some overshoot is honest: people undershoot when they trace. Twice the
/// drawn area is not overshoot, it is a different object.
const MAX_SPILL: f32 = 1.0;

/// Holds a segmenter's answer to what the user actually asked about.
///
/// The segmenter is given points and returns a mask; whether that mask is an
/// answer to the right question is decided here, against the drawing. It lives
/// beside the drawing rather than inside the inference code because it is a
/// judgement about intent, not a step of pre-processing — and because
/// `tasks.rs` is included by path from the integration tests, which cannot see
/// this module at all.
///
/// Returns true when the segmenter was believed and merely clipped, false when
/// it was overruled and the drawing used instead.
pub fn constrain(mask: &mut GrayImage, region: &Region) -> bool {
    let (covered, spilled) = region.agreement(mask, 128);

    if covered < MIN_COVERAGE || spilled > MAX_SPILL {
        // The answer is about something else. The drawing was never as precise
        // as a segmenter, but it is at least about the right thing, and a
        // slightly rough removal beats removing someone's leg.
        eprintln!(
            "[nabu] segmenter disagreed with the selection (covered {covered:.2}, spilled {spilled:.2}); using the drawn shape"
        );
        let (width, height) = mask.dimensions();
        *mask = region.as_mask(width, height);
        return false;
    }

    region.clip(mask);
    true
}

/// Grows the filled region by `radius` cells, as two separable maxima.
fn grow(mask: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = mask.dimensions();
    let r = radius as i64;
    let raw = mask.as_raw();

    let mut horizontal = vec![0u8; (w * h) as usize];
    for y in 0..h {
        let row_offset = (y * w) as usize;
        for x in 0..w {
            let mut best = 0u8;
            for dx in -r..=r {
                let sx = x as i64 + dx;
                if sx >= 0 && sx < w as i64 {
                    let v = raw[row_offset + sx as usize];
                    if v > best {
                        best = v;
                        if best == 255 {
                            break;
                        }
                    }
                }
            }
            horizontal[row_offset + x as usize] = best;
        }
    }

    let mut out = GrayImage::new(w, h);
    let out_raw: &mut [u8] = out.as_mut();
    let stride = w as usize;
    for x in 0..w {
        for y in 0..h {
            let mut best = 0u8;
            for dy in -r..=r {
                let sy = y as i64 + dy;
                if sy >= 0 && sy < h as i64 {
                    let v = horizontal[sy as usize * stride + x as usize];
                    if v > best {
                        best = v;
                        if best == 255 {
                            break;
                        }
                    }
                }
            }
            out_raw[y as usize * stride + x as usize] = best;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An L shape: the classic case where the bounding-box centre is outside.
    fn l_shape() -> Vec<[f32; 2]> {
        vec![
            [0.0, 0.0],
            [40.0, 0.0],
            [40.0, 200.0],
            [200.0, 200.0],
            [200.0, 240.0],
            [0.0, 240.0],
        ]
    }

    #[test]
    fn the_centre_of_the_bounding_box_can_be_outside_the_drawing() {
        // This is the bug the old prompt had: it clicked here.
        assert!(!contains(&l_shape(), 100.0, 100.0));
        assert!(contains(&l_shape(), 20.0, 100.0), "the arm of the L is inside");
    }

    #[test]
    fn prompt_points_land_inside_the_drawing() {
        let region = Region::from_polygon(&l_shape(), 256, 256).expect("region");
        let points = region.prompt_points(5);

        assert!(!points.is_empty(), "an L must yield prompt points");
        for point in &points {
            assert!(
                contains(&l_shape(), point[0], point[1]),
                "point {point:?} fell outside the drawing"
            );
        }
    }

    #[test]
    fn prompt_points_are_spread_out_rather_than_clustered() {
        let square = vec![[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]];
        let region = Region::from_polygon(&square, 256, 256).expect("region");
        let points = region.prompt_points(4);

        assert!(points.len() >= 2, "a large square must yield several points");
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                assert!(
                    (dx * dx + dy * dy).sqrt() > 4.0,
                    "points {i} and {j} are on top of each other"
                );
            }
        }
    }

    #[test]
    fn a_degenerate_drawing_has_no_region() {
        assert!(Region::from_polygon(&[[1.0, 1.0], [2.0, 2.0]], 100, 100).is_none());
        // A line encloses nothing.
        assert!(
            Region::from_polygon(&[[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]], 100, 100).is_none()
        );
    }

    #[test]
    fn clipping_erases_everything_far_from_the_drawing() {
        let square = vec![[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]];
        let region = Region::from_polygon(&square, 200, 200).expect("region");

        // A mask covering the whole picture, as a confused segmenter produces.
        let mut mask = GrayImage::from_pixel(200, 200, image::Luma([255]));
        region.clip(&mut mask);

        assert_eq!(mask.get_pixel(35, 35)[0], 255, "inside the drawing survives");
        assert_eq!(mask.get_pixel(180, 180)[0], 0, "the far corner is erased");
        assert_eq!(mask.get_pixel(5, 35)[0], 255, "the tolerance margin survives");
    }

    #[test]
    fn agreement_reports_both_how_much_is_covered_and_how_much_spills() {
        let square = vec![[10.0, 10.0], [110.0, 10.0], [110.0, 110.0], [10.0, 110.0]];
        let region = Region::from_polygon(&square, 200, 200).expect("region");

        let empty = GrayImage::new(200, 200);
        let (covered, spilled) = region.agreement(&empty, 128);
        assert!(covered < 0.01, "an empty mask covers nothing");
        assert!(spilled < 0.01);

        let everything = GrayImage::from_pixel(200, 200, image::Luma([255]));
        let (covered, spilled) = region.agreement(&everything, 128);
        assert!(covered > 0.9, "a full mask covers the drawing");
        assert!(spilled > 1.0, "and spills far more than the drawing itself");
    }
}
