// src-tauri/src/ai/vision/shadow.rs
//! Finding the shadow an object casts, so that it leaves when the object does.
//!
//! A segmenter answers "which pixels are the bicycle", and it answers it
//! correctly: the shadow on the paving is not the bicycle. But a removal that
//! takes the bicycle and leaves its shadow has not removed anything. The shape
//! on the ground still reads *bicycle*, and the viewer sees the edit before
//! they see the photograph — which is the one thing a removal must never do.
//!
//! No shadow model is involved, and for the ordinary case none is needed. A
//! cast shadow is the same surface under less light: darker than the lit
//! ground around it, the same colour as that ground once brightness is divided
//! out, and — the part that does the real work — physically joined to the thing
//! casting it. The first two tests gather candidates; the third throws away
//! everything that is merely dark.
//!
//! The heuristic has an honest failure mode, so it is built to fail safe: when
//! the flood starts swallowing the neighbourhood the whole detection is
//! abandoned. A shadow left behind is a blemish. Half a photograph erased
//! because a hedge happened to pass the tests is a ruined picture.

use std::collections::VecDeque;

use image::{GrayImage, Luma, Rgba, RgbaImage};

use super::inpaint::MASK_THRESHOLD;

/// How far from the object a shadow may run, as a multiple of its longest side.
///
/// A shadow is about as long as the thing casting it unless the sun is very
/// low, and past that this would be guessing rather than measuring. It bounds
/// two things: the rectangle searched, and the distance travelled *through the
/// shadow itself* — which is the stricter of the two, because a flood that has
/// to snake along a thin dark band to reach somewhere is not looking at the
/// same shadow any more.
const REACH: f32 = 0.5;

/// How much ground to measure over before deciding what "lit" means here, as a
/// multiple of the object's height.
///
/// Wider than the distance walked, and necessarily so: these are two different
/// questions. How far a shadow may run is a question about this object, and
/// wants a tight answer. What the lit floor looks like is a question about the
/// floor, and wants a sample big enough to contain some of it — measure it over
/// a strip that is mostly shadow and the brightest quarter of that strip is
/// still shadow, so nothing is darker than the standard and no shadow is ever
/// found. Tying both to one number is how that happens.
const SAMPLE_REACH: f32 = 1.5;

/// How far the object's outline may fall short of the ground and still be
/// counted as touching its own shadow, as a fraction of its longest side.
///
/// A segmenter rarely traces an object down to where it rests. This one stops
/// at the moped's frame and leaves the wheels out, so between the mask and the
/// shadow lies a band of unmasked machine — bright rim, sunlit stone between
/// the spokes — and a flood that insists on strict adjacency never crosses it.
/// Four pixels are enough to bridge that on a nine-megapixel frame, and going
/// thirty times wider takes in no more, so the exact figure is not a setting
/// anyone has to get right.
const CONTACT_GAP: f32 = 0.005;

/// The part of the object's height, measured up from its lowest point, whose
/// outline may seed a shadow.
///
/// A cast shadow begins where the object meets the ground. Letting the whole
/// outline seed it is what let a moped grow a shadow out of the hedge standing
/// behind it, and from there into the trees.
const CONTACT_BAND: f32 = 0.25;

/// How far above the object's base a shadow may reach, as a fraction of the
/// object's height.
///
/// Small, but not too small: perspective puts the far end of a shadow lying on
/// the ground higher in the frame than the end at your feet, and a shadow
/// running away from the camera starts well behind the contact point.
/// Everything further up is another surface — a wall, a hedge, the sky — and
/// whatever is dark up there was not cast by this object onto this floor.
///
/// Cutting this too fine does more damage than leaving the shadow alone. The
/// part that escapes the mask is still there for the inpainter to read, and it
/// dutifully continues it back into the hole: a shadow removed by halves comes
/// back whole.
const ABOVE_BASE: f32 = 0.2;

/// How much darker than the lit ground a pixel must be to read as shadow.
const DARKER_THAN: f32 = 0.72;

/// How far a shadow's colour may drift from the surface it falls on, across the
/// axis along which one material differs from another.
///
/// Measured as chromaticity — what a pixel's colour is once its brightness has
/// been divided out — because that is the quantity a change of *light* leaves
/// alone and a change of *material* does not. It is what separates a shadow on
/// paving from the dark inside of a hedge: both are dark, only one is still the
/// colour of paving.
const CHROMA_TOLERANCE: f32 = 0.05;

/// How far it may drift towards blue, which is not a change of material at all.
///
/// Out of the sun a surface is lit by the sky instead of by the sun, and the
/// sky is far bluer — so an outdoor shadow is systematically blue-shifted, by
/// more on a clear day than any tolerance tight enough to reject foliage would
/// allow. The two directions are not the same question and do not get the same
/// answer: green is a different material, blue is the same material in the
/// shade. Getting this wrong in either direction is visible immediately —
/// too tight and no real shadow is ever found, too loose and every dark thing
/// in the picture qualifies.
const BLUE_SHIFT: f32 = 0.12;

/// Largest shadow worth believing, as a multiple of the object's own area.
///
/// A backstop rather than the main guard — the geometry above does most of the
/// work — and generous, because a low sun genuinely does throw a shadow larger
/// than the silhouette that casts it. Measured against the *object*, never
/// against the area searched: that was the mistake in the first version, where
/// the allowance grew with the very rectangle it was meant to police, so a
/// large object could never trip it.
const MAX_OF_OBJECT: f32 = 2.5;

/// Below this, there is nothing lit nearby to be darker than.
const MIN_LIT: f32 = 8.0;

fn luminance(px: &Rgba<u8>) -> f32 {
    0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32
}

/// The colour a pixel has once its brightness is divided out, as the red and
/// green shares of the total. Blue is whatever is left, so two are enough.
fn chromaticity(px: &Rgba<u8>) -> (f32, f32) {
    let sum = px[0] as f32 + px[1] as f32 + px[2] as f32;
    if sum <= f32::EPSILON {
        return (1.0 / 3.0, 1.0 / 3.0);
    }
    (px[0] as f32 / sum, px[1] as f32 / sum)
}

/// Bounding box of the object, or `None` when nothing is marked.
fn object_bounds(object: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = object.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if object.get_pixel(x, y)[0] > MASK_THRESHOLD {
                any = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if any { Some((x0, y0, x1, y1)) } else { None }
}

/// The four neighbours of `(x, y)` that lie inside the searched rectangle.
fn neighbours(
    x: u32,
    y: u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> impl Iterator<Item = (u32, u32)> {
    [
        (x > x0).then(|| (x - 1, y)),
        (x < x1).then(|| (x + 1, y)),
        (y > y0).then(|| (x, y - 1)),
        (y < y1).then(|| (x, y + 1)),
    ]
    .into_iter()
    .flatten()
}

/// The shadow `object` casts in `image`, or an empty mask when there is no
/// convincing one.
///
/// The returned mask never covers the object itself — only the ground around
/// it — so it can be added to whatever the segmenter produced.
pub fn cast_shadow(image: &RgbaImage, object: &GrayImage) -> GrayImage {
    let (w, h) = image.dimensions();
    let mut out = GrayImage::new(w, h);
    if object.dimensions() != (w, h) || w == 0 || h == 0 {
        return out;
    }
    let Some((x0, y0, x1, y1)) = object_bounds(object) else {
        return out;
    };
    let object_area = object
        .pixels()
        .filter(|p| p[0] > MASK_THRESHOLD)
        .count() as u64;

    let obj_w = x1 - x0 + 1;
    let obj_h = y1 - y0 + 1;
    // Height, not the longest side: how far a shadow runs is set by how tall
    // the thing is and how low the sun is, and a wide object does not cast a
    // wide net. Deliberately less than the full height, because where two
    // shadows merge on the same floor there is no measurement that separates
    // them — so the error is taken on the side of leaving some behind, which
    // is a blemish, rather than erasing a neighbour's, which is a fabrication.
    let walk = ((obj_h as f32 * REACH) as u32).max(8);
    let span = ((obj_h as f32 * SAMPLE_REACH) as u32).max(walk * 2);

    // The ground: from a little above where the object stands, downwards.
    //
    // This is the constraint that was missing, and the one that does most of
    // the work. Everything a cast shadow can be is on the floor the object
    // rests on; everything above the base belongs to some other surface. Take
    // that away and the flood is free to walk from the shadow into whatever
    // dark thing happens to touch it, which in a garden is the whole garden.
    let rx0 = x0.saturating_sub(span);
    let rx1 = (x1 + span).min(w - 1);
    let ry0 = y1.saturating_sub((obj_h as f32 * ABOVE_BASE) as u32);
    let ry1 = (y1 + span).min(h - 1);
    if ry0 > ry1 || rx0 > rx1 {
        return out;
    }

    // What the ground looks like where the light reaches it.
    //
    // The brightest quarter of the strip, not its average. Half of it may well
    // be in shadow already, and averaging would then take the shadow's own
    // darkness as the standard to be darker than — which finds nothing,
    // exactly when there is most to find. Confining the strip to the ground is
    // also what makes this reference mean anything: measured over a rectangle
    // wide enough to hold sky, wall and hedge, "the lit surface nearby" is an
    // average of unrelated things and tells you nothing about any of them.
    let mut levels: Vec<f32> = Vec::new();
    for y in ry0..=ry1 {
        for x in rx0..=rx1 {
            if object.get_pixel(x, y)[0] > MASK_THRESHOLD {
                continue;
            }
            levels.push(luminance(image.get_pixel(x, y)));
        }
    }
    if levels.is_empty() {
        return out;
    }
    levels.sort_by(|a, b| a.total_cmp(b));
    let lit = levels[levels.len() * 3 / 4];
    if lit < MIN_LIT {
        return out;
    }

    let (mut sum_r, mut sum_g, mut counted) = (0f32, 0f32, 0u32);
    for y in ry0..=ry1 {
        for x in rx0..=rx1 {
            if object.get_pixel(x, y)[0] > MASK_THRESHOLD {
                continue;
            }
            let px = image.get_pixel(x, y);
            if luminance(px) < lit {
                continue;
            }
            let (r, g) = chromaticity(px);
            sum_r += r;
            sum_g += g;
            counted += 1;
        }
    }
    if counted == 0 {
        return out;
    }
    let (lit_r, lit_g) = (sum_r / counted as f32, sum_g / counted as f32);

    let threshold = lit * DARKER_THAN;
    let is_shadow = |x: u32, y: u32| {
        if object.get_pixel(x, y)[0] > MASK_THRESHOLD {
            return false;
        }
        let px = image.get_pixel(x, y);
        if luminance(px) > threshold {
            return false;
        }
        let (r, g) = chromaticity(px);
        // Green is the material axis here: foliage moves along it, an
        // illuminant does not. Red is allowed to fall away much further than
        // it may rise, because falling red is what going into the shade looks
        // like and rising red is what a different surface looks like.
        (g - lit_g).abs() <= CHROMA_TOLERANCE
            && (r - lit_r) <= CHROMA_TOLERANCE
            && (lit_r - r) <= BLUE_SHIFT
    };

    // Flood outwards from where the object meets the ground, and no further
    // than `reach` steps *through the shadow*. Only what the object touches can
    // be what the object cast — and only what it touches nearby, since a dark
    // doorway reached by threading a hundred pixels of hedge is connected in
    // the arithmetic sense and in no other.
    let contact = y1.saturating_sub((obj_h as f32 * CONTACT_BAND) as u32);
    let mut foot = GrayImage::new(w, h);
    for y in contact..=y1 {
        for x in x0..=x1 {
            if object.get_pixel(x, y)[0] > MASK_THRESHOLD {
                foot.put_pixel(x, y, Luma([255]));
            }
        }
    }
    let gap = ((obj_w.max(obj_h) as f32 * CONTACT_GAP) as u32).max(4);
    let near_ground = super::inpaint::dilate(&foot, gap);

    let mut taken = vec![false; (w * h) as usize];
    let mut queue = VecDeque::new();
    for y in ry0..=ry1 {
        for x in rx0..=rx1 {
            let i = (y * w + x) as usize;
            if taken[i] || near_ground.get_pixel(x, y)[0] == 0 || !is_shadow(x, y) {
                continue;
            }
            taken[i] = true;
            queue.push_back((x, y, 0u32));
        }
    }

    let ceiling = (object_area as f32 * MAX_OF_OBJECT) as u64;
    let mut area = 0u64;

    while let Some((x, y, depth)) = queue.pop_front() {
        area += 1;
        if area > ceiling {
            return GrayImage::new(w, h);
        }
        out.put_pixel(x, y, Luma([255]));
        if depth >= walk {
            continue;
        }

        for (nx, ny) in neighbours(x, y, rx0, ry0, rx1, ry1) {
            let i = (ny * w + nx) as usize;
            if !taken[i] && is_shadow(nx, ny) {
                taken[i] = true;
                queue.push_back((nx, ny, depth + 1));
            }
        }
    }

    out
}

/// `object` together with the shadow it casts, ready for the inpainter.
pub fn with_shadow(image: &RgbaImage, object: &GrayImage) -> GrayImage {
    let shadow = cast_shadow(image, object);
    let mut out = object.clone();
    for (px, marked) in out.pixels_mut().zip(shadow.pixels()) {
        if marked[0] > 0 {
            px[0] = 255;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grey `level` everywhere: lit ground.
    fn ground(w: u32, h: u32, level: u8) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba([level, level, level, 255]))
    }

    fn fill(image: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, colour: Rgba<u8>) {
        for y in y0..y1 {
            for x in x0..x1 {
                image.put_pixel(x, y, colour);
            }
        }
    }

    fn marked(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> GrayImage {
        let mut mask = GrayImage::new(w, h);
        for y in y0..y1 {
            for x in x0..x1 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        mask
    }

    /// Bright paving, a dark object on it, and a grey patch below the object
    /// that is the same colour as the paving but less lit.
    fn scene(shadow_colour: Rgba<u8>, shadow_y0: u32) -> (RgbaImage, GrayImage) {
        let mut image = ground(200, 200, 210);
        fill(&mut image, 90, shadow_y0, 150, shadow_y0 + 30, shadow_colour);
        fill(&mut image, 90, 40, 110, 80, Rgba([30, 40, 90, 255]));
        (image, marked(200, 200, 90, 40, 110, 80))
    }

    #[test]
    fn the_shadow_below_an_object_is_found() {
        let (image, object) = scene(Rgba([130, 130, 130, 255]), 80);
        let shadow = cast_shadow(&image, &object);

        assert_eq!(shadow.get_pixel(110, 90)[0], 255, "inside the shadow");
        assert_eq!(shadow.get_pixel(95, 82)[0], 255, "where it meets the object");
        assert_eq!(shadow.get_pixel(60, 20)[0], 0, "lit paving well away from it");
        assert_eq!(shadow.get_pixel(100, 60)[0], 0, "the object is not its own shadow");
    }

    #[test]
    fn a_dark_patch_the_object_does_not_touch_is_left_alone() {
        // The same patch, moved down so a band of lit paving separates the two.
        let (image, object) = scene(Rgba([130, 130, 130, 255]), 95);
        let shadow = cast_shadow(&image, &object);

        assert!(
            shadow.pixels().all(|p| p[0] == 0),
            "an unattached dark patch is somebody else's shadow, or none"
        );
    }

    #[test]
    fn a_shadow_lit_by_the_sky_is_still_a_shadow() {
        // Out of the sun, the paving is lit by the sky and comes back
        // distinctly blue. Every sunny photograph looks like this, and a
        // tolerance that treats the blue shift as a change of material finds
        // no shadows at all in any of them.
        let (image, object) = scene(Rgba([70, 85, 115, 255]), 80);
        let shadow = cast_shadow(&image, &object);

        assert_eq!(shadow.get_pixel(110, 90)[0], 255, "inside the blue shadow");
        assert_eq!(shadow.get_pixel(60, 20)[0], 0, "and not the sunlit paving");
    }

    #[test]
    fn something_dark_of_the_wrong_colour_is_not_a_shadow() {
        // A hedge: as dark as a shadow, joined to the object, and green.
        let (image, object) = scene(Rgba([40, 90, 40, 255]), 80);
        let shadow = cast_shadow(&image, &object);

        assert!(
            shadow.pixels().all(|p| p[0] == 0),
            "shadow keeps the colour of what it falls on; foliage does not"
        );
    }

    #[test]
    fn nothing_above_the_object_is_taken_for_its_shadow() {
        // A dark band directly above the object and touching it — the hedge
        // standing behind the moped — as dark as a real shadow and the same
        // colour as the ground. It is on another surface, so whatever cast it,
        // it was not this object onto this floor.
        let mut image = ground(200, 200, 210);
        fill(&mut image, 60, 6, 160, 40, Rgba([130, 130, 130, 255]));
        fill(&mut image, 90, 40, 110, 80, Rgba([30, 40, 90, 255]));
        let object = marked(200, 200, 90, 40, 110, 80);

        let shadow = cast_shadow(&image, &object);
        assert!(
            shadow.pixels().all(|p| p[0] == 0),
            "a shadow lies on the floor the object stands on, not above its head"
        );
    }

    #[test]
    fn the_shadow_stops_at_the_distance_it_can_be_walked() {
        // A narrow band of shadow running off to both sides. The near part
        // belongs to the object; the far part is some other shadow that merely
        // touches it, and no amount of connectivity makes it this one.
        let mut image = ground(400, 400, 210);
        fill(&mut image, 0, 145, 400, 175, Rgba([130, 130, 130, 255]));
        fill(&mut image, 100, 100, 140, 145, Rgba([30, 40, 90, 255]));
        let object = marked(400, 400, 100, 100, 140, 145);

        let shadow = cast_shadow(&image, &object);
        assert_eq!(shadow.get_pixel(120, 150)[0], 255, "just under the object");
        assert_eq!(
            shadow.get_pixel(184, 170)[0],
            0,
            "too far along the band to have been cast by this object"
        );
    }

    #[test]
    fn a_flood_that_swallows_the_neighbourhood_is_abandoned() {
        // A tall, narrow object — a post — standing on a floor that is dark for
        // as far as it may walk. Being tall it is allowed to walk a long way,
        // so the geometry alone does not bound this one and the area backstop
        // has to. The safe answer is to take nothing: a shadow left behind is a
        // blemish, a floor erased is a ruined picture.
        let mut image = ground(800, 800, 210);
        fill(&mut image, 0, 400, 800, 600, Rgba([120, 120, 120, 255]));
        fill(&mut image, 200, 100, 220, 400, Rgba([30, 40, 90, 255]));
        let object = marked(800, 800, 200, 100, 220, 400);

        let shadow = cast_shadow(&image, &object);
        assert!(
            shadow.pixels().all(|p| p[0] == 0),
            "a shadow many times the object's own area is not a shadow"
        );
    }

    #[test]
    fn nothing_marked_means_nothing_to_look_for() {
        let image = ground(64, 64, 200);
        let shadow = cast_shadow(&image, &GrayImage::new(64, 64));
        assert!(shadow.pixels().all(|p| p[0] == 0));
    }

    #[test]
    fn a_scene_with_no_light_in_it_yields_no_shadow() {
        let mut image = ground(120, 120, 3);
        fill(&mut image, 40, 40, 60, 60, Rgba([1, 1, 1, 255]));
        let object = marked(120, 120, 40, 40, 60, 60);

        assert!(cast_shadow(&image, &object).pixels().all(|p| p[0] == 0));
    }

    #[test]
    fn the_union_keeps_the_object_and_adds_the_shadow() {
        let (image, object) = scene(Rgba([130, 130, 130, 255]), 80);
        let both = with_shadow(&image, &object);

        assert_eq!(both.get_pixel(100, 60)[0], 255, "the object is still marked");
        assert_eq!(both.get_pixel(110, 90)[0], 255, "and so is its shadow");
        assert_eq!(both.get_pixel(60, 20)[0], 0, "and nothing else is");
    }
}
